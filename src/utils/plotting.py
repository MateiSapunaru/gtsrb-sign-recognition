from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    figsize: tuple[int, int] = (16, 14),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confusion_matrix, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(
    per_class_accuracy: list[float],
    class_names: list[str],
    output_path: str | Path,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, per_class_accuracy)
    ax.set_title("Per-Class Accuracy")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)