"""
Measures how well a detector trained on ONE LLM's generated code generalizes
to detecting code from a DIFFERENT LLM. This is a key "advanced" evaluation:
a detector that overfits to GPT-4's style but fails on Claude's output (or
vice versa) is much less useful in practice.

Produces an NxN transfer matrix (rows = trained-on model, columns =
tested-on model), which is exactly the kind of result that makes for a
compelling plot in a project report.

Usage:
    python -m src.evaluation.cross_model_eval --features data/processed/features.csv
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.models.baseline_ml import get_feature_cols


def cross_model_matrix(df: pd.DataFrame, source_models: list, feature_cols: list) -> pd.DataFrame:
    human_df = df[df["label"] == 0]
    matrix = pd.DataFrame(index=source_models, columns=source_models, dtype=float)

    for train_model, test_model in itertools.product(source_models, source_models):
        train_ai = df[(df["label"] == 1) & (df["source_model"] == train_model) & (df["split"] == "train")]
        test_ai = df[(df["label"] == 1) & (df["source_model"] == test_model) & (df["split"] == "test")]
        train_human = human_df[human_df["split"] == "train"]
        test_human = human_df[human_df["split"] == "test"]

        if len(train_ai) < 10 or len(test_ai) < 10:
            matrix.loc[train_model, test_model] = np.nan
            continue

        train_set = pd.concat([train_ai, train_human])
        test_set = pd.concat([test_ai, test_human])

        X_train, y_train = train_set[feature_cols].values, train_set["label"].values
        X_test, y_test = test_set[feature_cols].values, test_set["label"].values

        model = XGBClassifier(n_estimators=200, max_depth=5, eval_metric="logloss")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        matrix.loc[train_model, test_model] = auc

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--out", default="models_saved/cross_model_matrix.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    feature_cols = get_feature_cols(df)
    source_models = sorted(df[df["label"] == 1]["source_model"].dropna().unique().tolist())

    print(f"Evaluating cross-model generalization across: {source_models}")
    matrix = cross_model_matrix(df, source_models, feature_cols)
    matrix.to_csv(args.out)
    print("\nCross-model ROC-AUC transfer matrix (rows=trained on, cols=tested on):")
    print(matrix)
    print(f"\nSaved -> {args.out}")
