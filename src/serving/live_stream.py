from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from src.serving.app_inference import run_prediction_and_gradcam


@dataclass
class PredictionState:
    class_name: str = "Waiting for inference"
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _label_style() -> dict[str, int | float | tuple[int, int, int]]:
    return {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "font_scale": 0.7,
        "thickness": 2,
        "text_color": (255, 255, 255),
        "bg_color": (0, 128, 0),
        "padding": 8,
    }


class LiveTrafficSignProcessor:
    def __init__(self, infer_every_n_frames: int = 8) -> None:
        self.infer_every_n_frames = max(1, infer_every_n_frames)
        self.frame_count = 0
        self.last_prediction = PredictionState()

    def process_bgr_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, float]:
        self.frame_count += 1
        annotated = frame.copy()

        if self.frame_count % self.infer_every_n_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            result = run_prediction_and_gradcam(pil_image)
            self.last_prediction = PredictionState(
                class_name=str(result["predicted_class_name"]),
                confidence=float(result["predicted_confidence"]),
            )

        self._draw_prediction_banner(annotated)
        return annotated, self.last_prediction.class_name, self.last_prediction.confidence

    def _draw_prediction_banner(self, frame: np.ndarray) -> None:
        style = _label_style()
        text = f"{self.last_prediction.class_name} | {self.last_prediction.confidence * 100:.1f}%"

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            style["font"],
            style["font_scale"],
            style["thickness"],
        )
        pad = int(style["padding"])

        cv2.rectangle(
            frame,
            (10, 10),
            (10 + text_w + 2 * pad, 10 + text_h + baseline + 2 * pad),
            style["bg_color"],
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (10 + pad, 10 + pad + text_h),
            style["font"],
            style["font_scale"],
            style["text_color"],
            style["thickness"],
            lineType=cv2.LINE_AA,
        )
