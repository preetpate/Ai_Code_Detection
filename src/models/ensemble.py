"""
Combines the three detection signals (feature-based ML, fine-tuned
transformer, zero-shot perplexity/curvature) into one calibrated ensemble
score via logistic regression stacking. This is the model the demo app
actually serves, since it's the most robust of the four.

Usage:
    python -m src.models.ensemble --features data/processed/features.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def build_stacked_features(baseline_probs: np.ndarray, codebert_probs: np.ndarray, perplexity_scores: np.ndarray) -> np.ndarray:
    """Each input is a 1D array of per-sample scores from one detector,
    already aligned to the same sample order."""
    return np.column_stack([baseline_probs, codebert_probs, perplexity_scores])


def train_ensemble(stacked_train_X, y_train, stacked_val_X, y_val, out_path="models_saved/ensemble.pkl"):
    meta_model = LogisticRegression(class_weight="balanced")
    meta_model.fit(stacked_train_X, y_train)

    val_probs = meta_model.predict_proba(stacked_val_X)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)

    print("Ensemble validation performance:")
    print(f"  ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"  Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"  F1: {f1_score(y_val, val_preds):.4f}")
    print(f"  Learned weights (baseline, codebert, perplexity): {meta_model.coef_}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(meta_model, f)

    return meta_model


if __name__ == "__main__":
    # This script expects you've already produced per-sample probability
    # columns from each individual model and merged them into one CSV with
    # columns: label, split, baseline_prob, codebert_prob, perplexity_score
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_scores", default="data/processed/merged_scores.csv")
    parser.add_argument("--out", default="models_saved/ensemble.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.merged_scores)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    X_train = build_stacked_features(
        train_df["baseline_prob"].values, train_df["codebert_prob"].values, train_df["perplexity_score"].values
    )
    X_val = build_stacked_features(
        val_df["baseline_prob"].values, val_df["codebert_prob"].values, val_df["perplexity_score"].values
    )

    train_ensemble(X_train, train_df["label"].values, X_val, val_df["label"].values, args.out)
