"""
Interactive demo: paste code, get a verdict (human vs AI-generated) with a
confidence score and a plain-language explanation of the top contributing
features (via the trained XGBoost baseline + SHAP).

Run with:
    streamlit run app/streamlit_app.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
import shap
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.extract_features import extract_all_features

MODEL_PATH = "models_saved/xgboost.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def predict(code: str, model, feature_cols):
    features = extract_all_features(code)
    row = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)
    prob_ai = model.predict_proba(row.values)[0, 1]
    return prob_ai, row.values[0]


def explain(explainer, sample_row, feature_cols, top_k=6):
    shap_vals = explainer.shap_values(sample_row.reshape(1, -1))[0]
    contributions = sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))[:top_k]
    return contributions


st.set_page_config(page_title="AI Code Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ AI-Generated Code Detector")
st.caption("Paste a code snippet to estimate whether it was written by a human or an LLM.")

code_input = st.text_area("Paste your code here:", height=300, placeholder="def solve(nums):\n    ...")

if st.button("Analyze", type="primary") and code_input.strip():
    try:
        model, feature_cols = load_model()
    except FileNotFoundError:
        st.error(
            "No trained model found at models_saved/xgboost.pkl. "
            "Run `python -m src.models.baseline_ml` first to train one."
        )
        st.stop()

    prob_ai, sample_row = predict(code_input, model, feature_cols)

    verdict = "🤖 Likely AI-generated" if prob_ai > 0.5 else "🧑 Likely human-written"
    st.subheader(verdict)
    st.progress(float(prob_ai))
    st.write(f"Confidence AI-generated: **{prob_ai * 100:.1f}%**")

    explainer = get_explainer(model)
    contributions = explain(explainer, sample_row, feature_cols)

    st.markdown("### Why this verdict?")
    for feat, val in contributions:
        direction = "→ pushes toward **AI-generated**" if val > 0 else "→ pushes toward **human-written**"
        st.write(f"- `{feat}` (impact {val:+.3f}) {direction}")

    st.markdown("---")
    st.caption(
        "⚠️ Detection is probabilistic, not proof. False positives/negatives are "
        "expected, especially on very short snippets or heavily edited AI output."
    )
