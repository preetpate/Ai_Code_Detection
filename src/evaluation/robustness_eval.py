"""
Tests how much simple, semantics-preserving obfuscation degrades detector
accuracy: variable renaming, whitespace/formatting changes, and comment
stripping/insertion. A detector that collapses to chance under trivial
renaming is not measuring anything meaningful about "AI-ness" — it's
probably just fingerprinting superficial style, so this evaluation matters
for judging whether the earlier feature engineering actually captured
something robust.

Usage:
    python -m src.evaluation.robustness_eval --model models_saved/xgboost.pkl \
        --features data/processed/features.csv --dataset data/processed/dataset.csv
"""
import argparse
import ast
import pickle
import random
import string

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features.extract_features import extract_all_features


class VariableRenamer(ast.NodeTransformer):
    """Renames all local variable names to generic v0, v1, v2... to strip
    naming-style signal specifically."""

    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def _new_name(self, old_name):
        if old_name not in self.mapping:
            self.mapping[old_name] = f"v{self.counter}"
            self.counter += 1
        return self.mapping[old_name]

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)) and not node.id.startswith("__"):
            node.id = self._new_name(node.id)
        return node


def rename_variables(code: str) -> str:
    try:
        tree = ast.parse(code)
        renamer = VariableRenamer()
        new_tree = renamer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except (SyntaxError, ValueError):
        return code


def strip_comments(code: str) -> str:
    lines = [l for l in code.split("\n") if not l.strip().startswith("#")]
    return "\n".join(lines)


def reformat_whitespace(code: str) -> str:
    """Randomly adds/removes blank lines and trailing whitespace to break
    whitespace-consistency features without changing semantics."""
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(line.rstrip() + (" " * random.randint(0, 2)))
        if random.random() < 0.1:
            new_lines.append("")
    return "\n".join(new_lines)


ATTACKS = {
    "none": lambda c: c,
    "rename_vars": rename_variables,
    "strip_comments": strip_comments,
    "reformat_whitespace": reformat_whitespace,
    "all_combined": lambda c: reformat_whitespace(strip_comments(rename_variables(c))),
}


def evaluate_robustness(model_path: str, dataset_path: str, n_samples: int = 200):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_csv(dataset_path)
    test_df = df[df["split"] == "test"].dropna(subset=["code"])
    if len(test_df) > n_samples:
        test_df = test_df.sample(n_samples, random_state=42)

    results = {}
    for attack_name, attack_fn in ATTACKS.items():
        rows = []
        for code in test_df["code"]:
            try:
                attacked_code = attack_fn(code)
            except Exception:
                attacked_code = code
            rows.append(extract_all_features(attacked_code))

        feat_df = pd.DataFrame(rows).fillna(0)
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0
        X = feat_df[feature_cols].values

        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        y_true = test_df["label"].values

        results[attack_name] = {
            "accuracy": accuracy_score(y_true, preds),
            "roc_auc": roc_auc_score(y_true, probs),
        }

    print("\nRobustness under adversarial transformations:")
    for attack, metrics in results.items():
        print(f"  {attack:20s} accuracy={metrics['accuracy']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models_saved/xgboost.pkl")
    parser.add_argument("--dataset", default="data/processed/dataset.csv")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    evaluate_robustness(args.model, args.dataset, args.n_samples)
