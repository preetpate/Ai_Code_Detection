"""
End-to-end pipeline runner: builds dataset -> extracts features -> trains
baseline -> runs evaluations. Fine-tuning CodeBERT is left as a separate
manual step since it benefits from GPU time and closer monitoring.

Usage:
    python train.py
"""
import subprocess
import sys


STEPS = [
    ["python", "-m", "src.data.build_dataset"],
    ["python", "-m", "src.features.extract_features"],
    ["python", "-m", "src.models.baseline_ml"],
    ["python", "-m", "src.evaluation.cross_model_eval"],
    ["python", "-m", "src.evaluation.robustness_eval"],
    ["python", "-m", "src.explain.shap_explain"],
]


def run_pipeline():
    for step in STEPS:
        print(f"\n{'=' * 60}\nRunning: {' '.join(step)}\n{'=' * 60}")
        result = subprocess.run(step)
        if result.returncode != 0:
            print(f"Step failed: {' '.join(step)}. Stopping pipeline.")
            sys.exit(1)
    print("\nPipeline complete. Trained models are in models_saved/.")
    print("Fine-tune CodeBERT separately with: python -m src.models.codebert_classifier")
    print("Launch the demo with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    run_pipeline()
