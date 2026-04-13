from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

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


def sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", " "} else "_" for ch in text)
    return safe.replace(" ", "_")


def prepare_input_tensor(image_path: str | Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms(image_size=image_size)
    transformed = transform(image)
    input_tensor = transformed.unsqueeze(0)
    return transformed, input_tensor


def save_gradcam_sample(
    row: pd.Series,
    model: torch.nn.Module,
    gradcam: GradCAM,
    class_names: dict[int, str],
    image_size: int,
    output_dir: Path,
    prefix: str,
) -> dict:
    image_path = row["image_path"]
    true_label = int(row["true_label"])
    pred_label = int(row["pred_label"])
    confidence = float(row["confidence"])

    transformed_tensor, input_tensor = prepare_input_tensor(image_path, image_size=image_size)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    heatmap, predicted_class, gradcam_confidence = gradcam.generate(input_tensor=input_tensor)

    original_rgb = tensor_to_rgb_image(
        image_tensor=transformed_tensor,
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

    true_name = sanitize_filename(class_names[true_label])
    pred_name = sanitize_filename(class_names[pred_label])

    stem = Path(image_path).stem
    base_name = f"{prefix}_{stem}_true-{true_label}_{true_name}_pred-{pred_label}_{pred_name}"

    save_rgb_image(original_rgb, output_dir / f"{base_name}_original.png")
    save_rgb_image(overlay, output_dir / f"{base_name}_overlay.png")

    return {
        "image_path": image_path,
        "true_label": true_label,
        "true_class_name": class_names[true_label],
        "pred_label": pred_label,
        "pred_class_name": class_names[pred_label],
        "confidence_csv": confidence,
        "confidence_gradcam": gradcam_confidence,
        "correct": int(true_label == pred_label),
        "saved_prefix": base_name,
    }


def main() -> None:
    config = load_config()
    class_names = load_class_names()
    device = get_device()

    predictions_path = Path(config["artifacts"]["predictions_dir"]) / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}. Run evaluation first."
        )

    predictions_df = pd.read_csv(predictions_path)

    if "image_path" not in predictions_df.columns:
        raise ValueError(
            "test_predictions.csv does not contain 'image_path'. "
            "Please rerun evaluation after updating save_predictions_csv."
        )

    image_size = config["data"]["image_size"]

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

    output_root = Path(config["artifacts"]["figures_dir"]) / "gradcam_gallery"
    correct_dir = output_root / "correct"
    incorrect_dir = output_root / "incorrect"
    hard_dir = output_root / "hard_classes"

    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    correct_df = predictions_df[predictions_df["correct"] == 1].sort_values(
        by="confidence", ascending=False
    )
    incorrect_df = predictions_df[predictions_df["correct"] == 0].sort_values(
        by="confidence", ascending=False
    )

    # Pick confident correct examples
    correct_samples = correct_df.head(8)

    # Pick all incorrect examples, up to a reasonable cap
    incorrect_samples = incorrect_df.head(12)

    # Difficult classes from your evaluation
    hard_labels = [22, 27, 30]  # Bumpy road, Pedestrians, Beware of ice/snow
    hard_samples = (
        predictions_df[predictions_df["true_label"].isin(hard_labels)]
        .sort_values(by=["true_label", "confidence"], ascending=[True, False])
        .groupby("true_label", group_keys=False)
        .head(4)
    )

    summary_rows = []

    for idx, row in enumerate(correct_samples.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        summary_rows.append(
            save_gradcam_sample(
                row=row_series,
                model=model,
                gradcam=gradcam,
                class_names=class_names,
                image_size=image_size,
                output_dir=correct_dir,
                prefix=f"correct_{idx:02d}",
            )
        )

    for idx, row in enumerate(incorrect_samples.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        summary_rows.append(
            save_gradcam_sample(
                row=row_series,
                model=model,
                gradcam=gradcam,
                class_names=class_names,
                image_size=image_size,
                output_dir=incorrect_dir,
                prefix=f"incorrect_{idx:02d}",
            )
        )

    for idx, row in enumerate(hard_samples.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        summary_rows.append(
            save_gradcam_sample(
                row=row_series,
                model=model,
                gradcam=gradcam,
                class_names=class_names,
                image_size=image_size,
                output_dir=hard_dir,
                prefix=f"hard_{idx:02d}",
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "gradcam_gallery_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    gradcam.remove_hooks()

    print("Batch Grad-CAM generation complete.")
    print(f"Correct samples saved to: {correct_dir}")
    print(f"Incorrect samples saved to: {incorrect_dir}")
    print(f"Hard-class samples saved to: {hard_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Correct examples generated: {len(correct_samples)}")
    print(f"Incorrect examples generated: {len(incorrect_samples)}")
    print(f"Hard-class examples generated: {len(hard_samples)}")


if __name__ == "__main__":
    main()