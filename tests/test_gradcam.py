import numpy as np
import torch

from src.explainability.gradcam import (
    apply_colormap,
    overlay_heatmap_on_image,
    resize_heatmap,
    tensor_to_rgb_image,
)


def test_resize_heatmap():
    heatmap = np.random.rand(7, 7).astype(np.float32)
    resized = resize_heatmap(heatmap, (224, 224))
    assert resized.shape == (224, 224)


def test_apply_colormap():
    heatmap = np.random.rand(224, 224).astype(np.float32)
    colored = apply_colormap(heatmap)
    assert colored.shape == (224, 224, 3)


def test_overlay_heatmap_on_image():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap = np.random.rand(224, 224).astype(np.float32)
    overlay = overlay_heatmap_on_image(image, heatmap)
    assert overlay.shape == (224, 224, 3)


def test_tensor_to_rgb_image():
    tensor = torch.rand(3, 224, 224)
    image = tensor_to_rgb_image(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8