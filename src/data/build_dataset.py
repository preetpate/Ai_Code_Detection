"""
Merge human_code.jsonl and ai_code.jsonl into a single labeled dataset,
dedupe, and create stratified train/val/test splits.

Usage:
    python -m src.data.build_dataset \
        --human data/raw/human_code.jsonl \
        --ai data/raw/ai_code.jsonl \
        --out data/processed/dataset.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build(human_path: str, ai_path: str, out_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    df_human = load_jsonl(human_path)
    df_ai = load_jsonl(ai_path)

    df = pd.concat([df_human, df_ai], ignore_index=True)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    df = df[df["code"].str.strip().str.len() > 0].reset_index(drop=True)

    print(f"Total samples: {len(df)} | human={len(df[df.label==0])} ai={len(df[df.label==1])}")

    # stratify by label, and by source_model if present, to keep balance across
    # both classes and across which LLM generated the AI samples
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=seed,
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")

    full = pd.concat([train, val, test], ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"Saved dataset with splits -> {out_path}")
    print(full.groupby(["split", "label"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="data/raw/human_code.jsonl")
    parser.add_argument("--ai", default="data/raw/ai_code.jsonl")
    parser.add_argument("--out", default="data/processed/dataset.csv")
    args = parser.parse_args()
    build(args.human, args.ai, args.out)
