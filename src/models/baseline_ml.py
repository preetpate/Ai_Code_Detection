"""
Trains classical ML models (XGBoost, RandomForest, SVM) on the engineered
feature matrix and reports comparative performance. This is the fast,
interpretable baseline that the deep learning model should beat.

Usage:
    python -m src.models.baseline_ml --features data/processed/features.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier


NON_FEATURE_COLS = {"id", "label", "split", "source_model", "language", "code"}


def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def evaluate(model, X, y, name: str) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    return metrics


def train_and_evaluate(features_path: str, model_out_dir: str = "models_saved"):
    df = pd.read_csv(features_path)
    feature_cols = get_feature_cols(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val, y_val = val_df[feature_cols].values, val_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    models = {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1),
        "svm": SVC(kernel="rbf", probability=True, C=2.0),
    }

    results = []
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val, f"{name}_val")
        test_metrics = evaluate(model, X_test, y_test, f"{name}_test")
        results.extend([val_metrics, test_metrics])

        with open(Path(model_out_dir) / f"{name}.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

        print(f"\n=== {name} (test set) ===")
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, target_names=["human", "ai"]))

    with open(Path(model_out_dir) / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # feature importance from XGBoost, useful for the SHAP step and report writing
    xgb_model = models["xgboost"]
    importances = sorted(
        zip(feature_cols, xgb_model.feature_importances_), key=lambda x: -x[1]
    )[:20]
    print("\nTop 20 most important features (XGBoost):")
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--model_out_dir", default="models_saved")
    args = parser.parse_args()
    train_and_evaluate(args.features, args.model_out_dir)
