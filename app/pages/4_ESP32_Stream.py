from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.serving.live_stream import LiveTrafficSignProcessor


DEFAULT_STREAM_URL = "http://192.168.1.144:81/stream"


st.set_page_config(page_title="ESP32 Stream", page_icon="📡", layout="wide")

st.title("📡 ESP32 Stream")
st.write(
    "Run real-time traffic sign recognition from the ESP32-CAM MJPEG stream. "
    "Only one client should access the stream at a time."
)

stream_url = st.text_input(
    "ESP32 stream URL",
    value=DEFAULT_STREAM_URL,
    help="Example: http://192.168.1.144:81/stream",
)

infer_every_n_frames = st.slider(
    "Run inference every N frames",
    min_value=1,
    max_value=20,
    value=8,
    step=1,
    help="Higher values improve smoothness but reduce responsiveness.",
)

max_display_width = st.slider(
    "Display width (px)",
    min_value=320,
    max_value=1280,
    value=720,
    step=40,
)

run_stream = st.checkbox("Start stream")

st.info(
    "Close the ESP32 camera browser page before starting the app stream, or the board may refuse the connection."
)

frame_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()


@st.cache_resource
def get_processor(infer_every_n_frames: int) -> LiveTrafficSignProcessor:
    return LiveTrafficSignProcessor(infer_every_n_frames=infer_every_n_frames)


if run_stream:
    processor = get_processor(infer_every_n_frames)
    status_placeholder.info(f"Opening stream: {stream_url}")

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        st.error("Failed to open the ESP32 stream. Check the IP address and make sure no other client is using it.")
    else:
        frame_count = 0
        fps_window_start = time.time()
        fps_window_frames = 0

        try:
            while run_stream:
                ok, frame = cap.read()
                if not ok or frame is None:
                    status_placeholder.error("Failed to read a frame from the ESP32 stream.")
                    break

                frame_count += 1
                fps_window_frames += 1

                annotated_frame, prediction_text, prediction_confidence = processor.process_bgr_frame(frame)

                display_frame = annotated_frame
                frame_height, frame_width = display_frame.shape[:2]
                if frame_width > max_display_width:
                    scale = max_display_width / frame_width
                    new_size = (int(frame_width * scale), int(frame_height * scale))
                    display_frame = cv2.resize(display_frame, new_size)

                frame_placeholder.image(display_frame, channels="BGR", use_container_width=False)

                elapsed = time.time() - fps_window_start
                if elapsed >= 1.0:
                    current_fps = fps_window_frames / elapsed
                    metrics_placeholder.markdown(
                        f"**Frames processed:** {frame_count}  \\n"
                        f"**Approx. FPS:** {current_fps:.2f}  \\n"
                        f"**Latest prediction:** {prediction_text}  \\n"
                        f"**Confidence:** {prediction_confidence * 100:.2f}%"
                    )
                    fps_window_start = time.time()
                    fps_window_frames = 0

                # Small pause to keep Streamlit responsive.
                time.sleep(0.001)

        finally:
            cap.release()
            status_placeholder.info("Stream stopped.")
else:
    st.warning("Enable 'Start stream' to begin live inference.")
