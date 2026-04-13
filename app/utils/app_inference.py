from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.explainability.gradcam import (
    GradCAM,
    get_target_layer,
    overlay_heatmap_on_image,
    resize_heatmap,
    tensor_to_rgb_image,
)
from src.models.inference import load_class_names, load_trained_model, predict_image


@lru_cache(maxsize=1)
def get_model_bundle():
    model, config, device = load_trained_model()
    class_names = load_class_names()
    target_layer = get_target_layer(model, config["model"]["name"])
    gradcam = GradCAM(model, target_layer)

    return {
        "model": model,
        "config": config,
        "device": device,
        "class_names": class_names,
        "gradcam": gradcam,
    }


def run_prediction_and_gradcam(image: Image.Image) -> dict:
    bundle = get_model_bundle()

    model = bundle["model"]
    config = bundle["config"]
    device = bundle["device"]
    gradcam = bundle["gradcam"]

    prediction = predict_image(
        model=model,
        image=image,
        config=config,
        device=device,
    )

    input_tensor = prediction["input_tensor"]
    heatmap, predicted_class, gradcam_confidence = gradcam.generate(input_tensor=input_tensor)

    input_tensor_cpu = input_tensor[0].detach().cpu()
    original_rgb = tensor_to_rgb_image(
        image_tensor=input_tensor_cpu,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    heatmap_resized = resize_heatmap(
        heatmap=heatmap,
        image_size=(original_rgb.shape[1], original_rgb.shape[0]),
    )

    overlay = overlay_heatmap_on_image(
        image_rgb=original_rgb,
        heatmap=heatmap_resized,
        alpha=0.4,
    )

    prediction["gradcam_predicted_class"] = predicted_class
    prediction["gradcam_confidence"] = gradcam_confidence
    prediction["display_image"] = original_rgb
    prediction["gradcam_overlay"] = overlay

    return prediction