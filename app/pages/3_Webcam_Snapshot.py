from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.serving.app_inference import run_prediction_and_gradcam


st.set_page_config(page_title="Webcam Snapshot", page_icon="📷", layout="wide")

st.title("📷 Webcam Snapshot")
st.write("Capture an image from your webcam and run traffic sign recognition with Grad-CAM.")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

    with st.spinner("Running inference..."):
        result = run_prediction_and_gradcam(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original (preprocessed display)")
        st.image(result["display_image"], width="stretch")

    with col2:
        st.subheader("Grad-CAM Overlay")
        st.image(result["gradcam_overlay"], width="stretch")

    st.subheader("Prediction")
    st.metric(
        label=result["predicted_class_name"],
        value=f"{result['predicted_confidence'] * 100:.2f}%",
    )

    st.subheader("Top-K Predictions")
    top_k_df = pd.DataFrame(result["top_predictions"])
    top_k_df["confidence"] = top_k_df["confidence"].apply(lambda x: round(x * 100, 2))
    st.dataframe(
        top_k_df.rename(
            columns={
                "label": "Class ID",
                "class_name": "Class Name",
                "confidence": "Confidence (%)",
            }
        )
    )
else:
    st.info("Use the camera widget above to capture an image.")