from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import av
import cv2
import numpy as np
import torch
from PIL import Image

from src.models.inference import load_trained_model, predict_image


@lru_cache(maxsize=1)
def get_live_model_bundle():
    model, config, device = load_trained_model()
    return {
        "model": model,
        "config": config,
        "device": device,
    }


@dataclass
class PredictionState:
    label: str = "No prediction yet"
    confidence: float = 0.0
    frame_count: int = 0


class LiveTrafficSignProcessor:
    def __init__(self, infer_every_n_frames: int = 8) -> None:
        self.bundle = get_live_model_bundle()
        self.model = self.bundle["model"]
        self.config = self.bundle["config"]
        self.device = self.bundle["device"]

        self.infer_every_n_frames = infer_every_n_frames
        self.state = PredictionState()
        self.lock = Lock()

    def _predict_from_bgr_frame(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        result = predict_image(
            model=self.model,
            image=pil_image,
            config=self.config,
            device=self.device,
        )

        return result["predicted_class_name"], result["predicted_confidence"]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.state.frame_count += 1

            if self.state.frame_count % self.infer_every_n_frames == 0:
                label, confidence = self._predict_from_bgr_frame(img_bgr)
                self.state.label = label
                self.state.confidence = confidence

            display_label = self.state.label
            display_conf = self.state.confidence

        overlay_text = f"{display_label} | {display_conf * 100:.2f}%"
        cv2.rectangle(img_bgr, (10, 10), (900, 60), (0, 0, 0), thickness=-1)
        cv2.putText(
            img_bgr,
            overlay_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")