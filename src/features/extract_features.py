"""
Runs AST + style feature extractors over every sample in the dataset and
writes a flat feature matrix ready for classical ML training.

Usage:
    python -m src.features.extract_features \
        --input data/processed/dataset.csv \
        --output data/processed/features.csv
"""
import argparse

import pandas as pd
from tqdm import tqdm

from src.features.ast_features import extract_ast_features
from src.features.style_features import extract_style_features


def extract_all_features(code: str) -> dict:
    features = {}
    features.update(extract_ast_features(code))
    features.update(extract_style_features(code))
    return features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in tqdm(df["code"], desc="Extracting features"):
        rows.append(extract_all_features(code))
    feat_df = pd.DataFrame(rows).fillna(0)
    meta_cols = [c for c in ["id", "label", "split", "source_model", "language"] if c in df.columns]
    return pd.concat([df[meta_cols].reset_index(drop=True), feat_df], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/dataset.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_df = build_feature_matrix(df)
    feature_df.to_csv(args.output, index=False)
    print(f"Saved feature matrix with {feature_df.shape[1]} columns -> {args.output}")
