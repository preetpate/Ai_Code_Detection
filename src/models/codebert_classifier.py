"""
Fine-tunes a pretrained code transformer (default: microsoft/codebert-base)
as a binary sequence classifier: human (0) vs AI-generated (1).

Usage:
    python -m src.models.codebert_classifier --data data/processed/dataset.csv \
        --model_name microsoft/codebert-base --epochs 4
"""
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)


def load_splits(data_path: str):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["code"])
    train_df = df[df["split"] == "train"][["code", "label"]]
    val_df = df[df["split"] == "val"][["code", "label"]]
    test_df = df[df["split"] == "test"][["code", "label"]]
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def tokenize_fn(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["code"], truncation=True, padding="max_length", max_length=max_length
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, probs),
    }


def train(
    data_path: str,
    model_name: str = "microsoft/codebert-base",
    output_dir: str = "models_saved/codebert",
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
):
    train_ds, val_ds, test_ds = load_splits(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = train_ds.map(lambda e: tokenize_fn(e, tokenizer), batched=True)
    val_ds = val_ds.map(lambda e: tokenize_fn(e, tokenizer), batched=True)
    test_ds = test_ds.map(lambda e: tokenize_fn(e, tokenizer), batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("\n=== Test set results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model_name", default="microsoft/codebert-base")
    parser.add_argument("--output_dir", default="models_saved/codebert")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(
        args.data, args.model_name, args.output_dir,
        args.epochs, args.batch_size, args.lr,
    )
