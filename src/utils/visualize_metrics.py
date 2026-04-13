from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "training_history.json"
TEST_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "test_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "plots"


def load_json(path: Path) -> dict:
    """Load a JSON file with basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_training_stages(training_history: dict) -> list[dict]:
    """Combine stage_1 and stage_2 into a single ordered epoch list."""
    combined: list[dict] = []
    global_epoch = 1

    for stage_name in ("stage_1", "stage_2"):
        stage_records = training_history.get(stage_name, [])
        for record in stage_records:
            combined.append(
                {
                    "global_epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": record["epoch"],
                    "train_loss": record["train_loss"],
                    "train_acc": record["train_acc"],
                    "val_loss": record["val_loss"],
                    "val_acc": record["val_acc"],
                }
            )
            global_epoch += 1

    if not combined:
        raise ValueError("No training records found in training_history.json")

    return combined


def plot_training_curves(records: list[dict], output_dir: Path) -> None:
    """Plot train/val loss and accuracy across both training stages."""
    epochs = [r["global_epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    train_acc = [r["train_acc"] for r in records]
    val_acc = [r["val_acc"] for r in records]

    stage_1_len = sum(1 for r in records if r["stage"] == "stage_1")

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
    plt.axvline(stage_1_len + 0.5, linestyle="--", label="Stage Boundary")
    plt.xlabel("Global Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
    plt.axvline(stage_1_len + 0.5, linestyle="--", label="Stage Boundary")
    plt.xlabel("Global Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_accuracy.png", dpi=200)
    plt.close()


def extract_per_class_metrics(test_metrics: dict) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    """Extract per-class precision/recall/F1/support from classification_report."""
    report = test_metrics.get("classification_report", {})
    excluded_keys = {"accuracy", "macro avg", "weighted avg"}

    class_names: list[str] = []
    precision: list[float] = []
    recall: list[float] = []
    f1_score: list[float] = []
    support: list[float] = []

    for label, metrics in report.items():
        if label in excluded_keys:
            continue
        if not isinstance(metrics, dict):
            continue

        class_names.append(label)
        precision.append(metrics.get("precision", 0.0))
        recall.append(metrics.get("recall", 0.0))
        f1_score.append(metrics.get("f1-score", 0.0))
        support.append(metrics.get("support", 0.0))

    if not class_names:
        raise ValueError("No per-class metrics found in classification_report")

    return class_names, precision, recall, f1_score, support


def plot_test_summary(test_metrics: dict, output_dir: Path) -> None:
    """Plot overall test metrics."""
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    metric_values = [test_metrics[name] for name in metric_names]

    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Overall Test Metrics")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_test_metrics.png", dpi=200)
    plt.close()


def plot_bottom_classes(test_metrics: dict, output_dir: Path, top_k: int = 10) -> None:
    """Plot the weakest classes by recall/F1 to highlight error-prone categories."""
    class_names, precision, recall, f1_score, _ = extract_per_class_metrics(test_metrics)

    rows = list(zip(class_names, precision, recall, f1_score))
    rows.sort(key=lambda x: x[2])  # sort by recall ascending
    weakest = rows[:top_k]

    labels = [r[0] for r in weakest]
    recalls = [r[2] for r in weakest]
    f1s = [r[3] for r in weakest]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, recalls, width, label="Recall")
    plt.bar(x + width / 2, f1s, width, label="F1-score")
    plt.ylim(0.0, 1.05)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Score")
    plt.title(f"Lowest-{top_k} Performing Classes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lowest_performing_classes.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_history = load_json(TRAINING_HISTORY_PATH)
    test_metrics = load_json(TEST_METRICS_PATH)

    combined_records = combine_training_stages(training_history)
    plot_training_curves(combined_records, OUTPUT_DIR)
    plot_test_summary(test_metrics, OUTPUT_DIR)
    plot_bottom_classes(test_metrics, OUTPUT_DIR, top_k=10)

    print("Saved plots to:")
    print(f"  - {OUTPUT_DIR / 'training_validation_loss.png'}")
    print(f"  - {OUTPUT_DIR / 'training_validation_accuracy.png'}")
    print(f"  - {OUTPUT_DIR / 'overall_test_metrics.png'}")
    print(f"  - {OUTPUT_DIR / 'lowest_performing_classes.png'}")


if __name__ == "__main__":
    main()