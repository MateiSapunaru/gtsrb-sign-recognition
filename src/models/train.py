from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.dataloader import build_dataloaders
from src.models.model_factory import build_model, freeze_backbone, unfreeze_model
from src.utils.seed import set_seed


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def save_training_history(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def run_training(config_path: str | Path = "configs/config.yaml") -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    device = get_device()
    print(f"Using device: {device}")

    dataloaders = build_dataloaders(config_path)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model_name = config["model"]["name"]
    num_classes = config["data"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    dropout = config["model"]["dropout"]

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    ).to(device)

    label_smoothing = config["training"]["label_smoothing"]
    weight_decay = config["training"]["weight_decay"]

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {
        "stage_1": [],
        "stage_2": [],
    }

    best_val_acc = 0.0
    best_model_path = (
        Path(config["artifacts"]["model_dir"]) / config["artifacts"]["best_model_name"]
    )

    # Stage 1: train classifier head
    print("\nStage 1: training classifier head")
    freeze_backbone(model, model_name)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate_head"],
        weight_decay=weight_decay,
    )

    for epoch in range(config["training"]["epochs_head"]):
        print(f"\n[Stage 1] Epoch {epoch + 1}/{config['training']['epochs_head']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history["stage_1"].append(epoch_result)

        print(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_model_path)
            print(f"Saved new best model to {best_model_path}")

    # Stage 2: fine-tune full model
    print("\nStage 2: fine-tuning full model")
    unfreeze_model(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate_finetune"],
        weight_decay=weight_decay,
    )

    for epoch in range(config["training"]["epochs_finetune"]):
        print(
            f"\n[Stage 2] Epoch {epoch + 1}/{config['training']['epochs_finetune']}"
        )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history["stage_2"].append(epoch_result)

        print(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_model_path)
            print(f"Saved new best model to {best_model_path}")

    history_path = Path(config["artifacts"]["metrics_dir"]) / "training_history.json"
    save_training_history(history, history_path)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print(f"Training history saved at: {history_path}")


if __name__ == "__main__":
    run_training()