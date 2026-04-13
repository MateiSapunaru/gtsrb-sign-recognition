from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.data.dataset import GTSRBDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from src.explainability.gradcam import (
    GradCAM,
    get_target_layer,
    overlay_heatmap_on_image,
    resize_heatmap,
    save_rgb_image,
    tensor_to_rgb_image,
)
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


def main() -> None:
    config = load_config()
    class_names = load_class_names()
    device = get_device()

    image_size = config["data"]["image_size"]

    dataset = GTSRBDataset(
        csv_path=config["data"]["test_processed_csv"],
        transform=get_eval_transforms(image_size=image_size),
    )

    sample_index = 0
    image_tensor, true_label = dataset[sample_index]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["data"]["num_classes"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    ).to(device)

    best_model_path = (
        Path(config["artifacts"]["model_dir"]) / config["artifacts"]["best_model_name"]
    )
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = get_target_layer(model, config["model"]["name"])
    gradcam = GradCAM(model, target_layer)

    heatmap, predicted_class, confidence = gradcam.generate(input_tensor=input_tensor)

    original_rgb = tensor_to_rgb_image(
        image_tensor=image_tensor,
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

    output_dir = Path(config["artifacts"]["figures_dir"]) / "gradcam_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_rgb_image(original_rgb, output_dir / "sample_original.png")
    save_rgb_image(overlay, output_dir / "sample_gradcam_overlay.png")

    print(f"True label: {true_label} -> {class_names[true_label]}")
    print(f"Predicted label: {predicted_class} -> {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Saved original image to: {output_dir / 'sample_original.png'}")
    print(f"Saved Grad-CAM overlay to: {output_dir / 'sample_gradcam_overlay.png'}")

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()