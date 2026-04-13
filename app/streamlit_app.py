import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="GTSRB Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 GTSRB Traffic Sign Recognition")
st.markdown(
    """
    End-to-end computer vision project using:
    - Transfer Learning
    - PyTorch
    - Grad-CAM
    - Streamlit
    - ESP32-CAM live streaming
    - Docker
    """
)

st.info(
    "Use the sidebar pages to explore the dashboard, upload an image, take a snapshot, or run live ESP32 stream inference."
)
