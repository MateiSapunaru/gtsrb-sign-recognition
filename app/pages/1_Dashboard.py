from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
import json

import streamlit as st


st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Model Dashboard")

metrics_path = Path("artifacts/metrics/test_metrics.json")
history_path = Path("artifacts/metrics/training_history.json")
cm_path = Path("artifacts/figures/confusion_matrix.png")
pca_path = Path("artifacts/figures/per_class_accuracy.png")

if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Macro F1", f"{metrics['f1_macro'] * 100:.2f}%")
    col3.metric("Weighted F1", f"{metrics['f1_weighted'] * 100:.2f}%")
else:
    st.warning("Test metrics not found yet.")

if cm_path.exists():
    st.subheader("Confusion Matrix")
    st.image(str(cm_path), width="stretch")

if pca_path.exists():
    st.subheader("Per-Class Accuracy")
    st.image(str(pca_path), width="stretch")

if history_path.exists():
    st.success("Training history found. We can add full training curves next.")