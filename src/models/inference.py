from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.model_factory import build_model


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_class_names(classes_path: str | Path = "configs/classes.yaml") -> dict[int, str]:
    with open(classes_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if raw is None or not isinstance(raw, dict):
        raise ValueError(f"{classes_path} is empty or invalid.")

    return {int(k): str(v) for k, v in raw.items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(config_path: str | Path = "configs/config.yaml") -> tuple[torch.nn.Module, dict, torch.device]:
    config = load_config(config_path)
    device = get_device()

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["data"]["num_classes"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    ).to(device)

    best_model_path = (
        Path(config["artifacts"]["model_dir"]) / config["artifacts"]["best_model_name"]
    )

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, device


def preprocess_pil_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    transform = get_eval_transforms(image_size=image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    config: dict,
    device: torch.device,
    top_k: int | None = None,
) -> dict:
    if top_k is None:
        top_k = int(config["inference"]["top_k"])

    class_names = load_class_names()
    image_size = int(config["data"]["image_size"])

    input_tensor = preprocess_pil_image(image=image, image_size=image_size).to(device)

    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    predicted_label = int(top_indices[0][0].item())
    predicted_confidence = float(top_probs[0][0].item())

    top_predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = int(idx.item())
        top_predictions.append(
            {
                "label": label,
                "class_name": class_names[label],
                "confidence": float(prob.item()),
            }
        )

    return {
        "predicted_label": predicted_label,
        "predicted_class_name": class_names[predicted_label],
        "predicted_confidence": predicted_confidence,
        "top_predictions": top_predictions,
        "input_tensor": input_tensor,
    }