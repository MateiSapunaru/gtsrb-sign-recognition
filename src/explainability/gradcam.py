from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GradCAMResult:
    predicted_class: int
    confidence: float
    heatmap: np.ndarray
    overlay: np.ndarray


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a single input tensor.

        Args:
            input_tensor: Tensor of shape [1, C, H, W]
            class_idx: Optional target class index. If None, uses predicted class.

        Returns:
            heatmap: np.ndarray in [0, 1], shape [H, W]
            predicted_class: int
            confidence: float
        """
        self.model.eval()

        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

        target_class = predicted_class if class_idx is None else class_idx

        self.model.zero_grad()
        score = output[:, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * activations).sum(dim=0)            # [H, W]
        cam = F.relu(cam)

        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, predicted_class, confidence


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        # last feature block before pooling/classifier
        return model.features[-1]

    if model_name == "resnet18":
        return model.layer4[-1]

    raise ValueError(
        f"Unsupported model_name='{model_name}' for Grad-CAM. "
        "Supported: ['efficientnet_b0', 'resnet18']"
    )


def tensor_to_rgb_image(
    image_tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """
    Convert normalized tensor [C, H, W] to uint8 RGB image [H, W, C].
    """
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)

    image = (image * std_arr) + mean_arr
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image


def resize_heatmap(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    resized = cv2.resize(heatmap, (width, height))
    resized = np.clip(resized, 0, 1)
    return resized


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert heatmap [0,1] to RGB uint8 heatmap.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a heatmap [0,1] on an RGB uint8 image.
    """
    colored_heatmap = apply_colormap(heatmap)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)