"""
Generates SHAP explanations for the trained XGBoost baseline: a global
summary plot (which features matter most overall) and per-sample
explanations (why THIS snippet was flagged as AI-generated).

Usage:
    python -m src.explain.shap_explain --features data/processed/features.csv \
        --model models_saved/xgboost.pkl
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


def global_explanation(model, X, feature_cols, out_path="models_saved/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot -> {out_path}")
    return explainer, shap_values


def explain_single_sample(explainer, sample_row, feature_cols, top_k: int = 5):
    """Returns the top_k features that pushed this specific sample's
    prediction toward AI-generated (positive SHAP value) or human
    (negative SHAP value)."""
    shap_vals = explainer.shap_values(sample_row.reshape(1, -1))[0]
    contributions = sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))[:top_k]

    explanation = []
    for feat, val in contributions:
        direction = "toward AI-generated" if val > 0 else "toward human-written"
        explanation.append(f"{feat} (impact={val:+.3f}, pushes {direction})")
    return explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--model", default="models_saved/xgboost.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model, feature_cols = load_model(args.model)
    test_df = df[df["split"] == "test"]
    X_test = test_df[feature_cols].values

    explainer, shap_values = global_explanation(model, X_test, feature_cols)

    print("\nExample per-sample explanation (first test sample):")
    for line in explain_single_sample(explainer, X_test[0], feature_cols):
        print(f"  - {line}")
