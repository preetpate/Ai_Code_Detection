"""
Zero-shot detector based on the intuition (from DetectGPT and related work)
that text sampled FROM a language model tends to sit in a local optimum of
that model's own probability landscape: small perturbations of AI-written
text tend to LOWER its log-probability less / more predictably than
perturbations of human text do. Here we implement a simpler but effective
proxy: raw perplexity + a curvature estimate under random token masking,
computed with a pretrained causal code LM (default: Salesforce/codegen-350M-mono).

This needs NO labeled training data at all, and generalizes across model
families better than a trained classifier — useful as a complementary signal
and as an ablation baseline ("how much does the trained classifier add over
a training-free signal?").

Usage:
    python -m src.models.perplexity_detector --data data/processed/dataset.csv
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class PerplexityDetector:
    def __init__(self, model_name: str = "Salesforce/codegen-350M-mono", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def log_likelihood(self, code: str, max_length: int = 512) -> float:
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        # negative loss * n_tokens ~ total log likelihood; we return mean log-prob per token
        return -outputs.loss.item()

    def perplexity(self, code: str) -> float:
        ll = self.log_likelihood(code)
        return float(np.exp(-ll))

    def perturb(self, code: str, n_swaps: int = 5) -> str:
        """Cheap structure-preserving perturbation: shuffle a few whitespace-
        separated tokens' order within short windows, approximating
        DetectGPT-style random rewrites without needing a second mask-filling
        model."""
        tokens = code.split(" ")
        if len(tokens) < 4:
            return code
        tokens = tokens.copy()
        for _ in range(min(n_swaps, len(tokens) // 2)):
            i = random.randint(0, len(tokens) - 2)
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
        return " ".join(tokens)

    def curvature_score(self, code: str, n_perturbations: int = 10) -> float:
        """DetectGPT-style curvature: original log-likelihood minus the mean
        log-likelihood of perturbed versions. AI-generated text tends to sit
        at a local maximum of the model's likelihood, so perturbing it drops
        likelihood more sharply than perturbing human text does -> higher
        (positive) curvature score suggests AI-generated."""
        orig_ll = self.log_likelihood(code)
        perturbed_lls = [self.log_likelihood(self.perturb(code)) for _ in range(n_perturbations)]
        mean_perturbed_ll = float(np.mean(perturbed_lls))
        std_perturbed_ll = float(np.std(perturbed_lls)) + 1e-8
        return (orig_ll - mean_perturbed_ll) / std_perturbed_ll


def evaluate_zero_shot(data_path: str, model_name: str, n_samples: int = 200, n_perturbations: int = 10):
    df = pd.read_csv(data_path)
    test_df = df[df["split"] == "test"].dropna(subset=["code"])
    if len(test_df) > n_samples:
        test_df = test_df.sample(n_samples, random_state=42)

    detector = PerplexityDetector(model_name)

    scores, labels = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Scoring"):
        score = detector.curvature_score(row["code"], n_perturbations=n_perturbations)
        scores.append(score)
        labels.append(row["label"])

    auc = roc_auc_score(labels, scores)
    # pick threshold at median for a simple accuracy readout
    threshold = float(np.median(scores))
    preds = [1 if s > threshold else 0 for s in scores]
    acc = accuracy_score(labels, preds)

    print(f"Zero-shot perplexity/curvature detector: ROC-AUC={auc:.4f}, Accuracy@median-threshold={acc:.4f}")
    return {"roc_auc": auc, "accuracy": acc, "threshold": threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model_name", default="Salesforce/codegen-350M-mono")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_perturbations", type=int, default=10)
    args = parser.parse_args()

    evaluate_zero_shot(args.data, args.model_name, args.n_samples, args.n_perturbations)
