from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data.dataloader import build_dataloaders
from src.models.model_factory import build_model
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import plot_confusion_matrix, plot_per_class_accuracy
from src.utils.seed import set_seed


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_class_names(classes_path: str | Path = "configs/classes.yaml") -> dict[int, str]:
    with open(classes_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return {int(k): str(v) for k, v in raw.items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    avg_loss = running_loss / total

    return avg_loss, all_labels, all_predictions, all_probabilities


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_predictions_csv(
    image_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: dict[int, str],
    output_path: str | Path,
) -> None:
    rows = []

    for idx, (image_path, true_label, pred_label, probs) in enumerate(
        zip(image_paths, y_true, y_pred, probabilities)
    ):
        confidence = float(max(probs))

        rows.append(
            {
                "sample_index": idx,
                "image_path": image_path,
                "true_label": true_label,
                "true_class_name": class_names[true_label],
                "pred_label": pred_label,
                "pred_class_name": class_names[pred_label],
                "confidence": confidence,
                "correct": int(true_label == pred_label),
            }
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_evaluation(config_path: str | Path = "configs/config.yaml") -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    class_names_map = load_class_names()
    class_names = [class_names_map[i] for i in range(config["data"]["num_classes"])]

    device = get_device()
    print(f"Using device: {device}")

    dataloaders = build_dataloaders(config_path)
    test_loader = dataloaders["test"]
    test_dataset_df = pd.read_csv(config["data"]["test_processed_csv"])
    image_paths = test_dataset_df["image_path"].tolist()

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

    criterion = nn.CrossEntropyLoss()

    test_loss, y_true, y_pred, probabilities = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
    )
    metrics["test_loss"] = float(test_loss)

    metrics_dir = Path(config["artifacts"]["metrics_dir"])
    figures_dir = Path(config["artifacts"]["figures_dir"])
    predictions_dir = Path(config["artifacts"]["predictions_dir"])

    save_json(metrics, metrics_dir / "test_metrics.json")

    confusion_matrix_np = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix_np,
        class_names=class_names,
        output_path=figures_dir / "confusion_matrix.png",
    )

    plot_per_class_accuracy(
        per_class_accuracy=metrics["per_class_accuracy"],
        class_names=class_names,
        output_path=figures_dir / "per_class_accuracy.png",
    )

    save_predictions_csv(
    image_paths=image_paths,
    y_true=y_true,
    y_pred=y_pred,
    probabilities=probabilities,
    class_names=class_names_map,
    output_path=predictions_dir / "test_predictions.csv",
    )

    print("\nEvaluation complete.")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Saved metrics to: {metrics_dir / 'test_metrics.json'}")
    print(f"Saved confusion matrix to: {figures_dir / 'confusion_matrix.png'}")
    print(f"Saved per-class accuracy to: {figures_dir / 'per_class_accuracy.png'}")
    print(f"Saved predictions to: {predictions_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    run_evaluation()