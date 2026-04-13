from __future__ import annotations

from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.data.dataset import GTSRBDataset
from src.data.transforms import get_eval_transforms, get_train_transforms


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataloaders(config_path: str | Path = "configs/config.yaml") -> dict:
    config = load_config(config_path)

    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_dataset = GTSRBDataset(
        csv_path=config["data"]["train_csv"],
        transform=get_train_transforms(image_size=image_size),
    )

    val_dataset = GTSRBDataset(
        csv_path=config["data"]["val_csv"],
        transform=get_eval_transforms(image_size=image_size),
    )

    test_dataset = GTSRBDataset(
        csv_path=config["data"]["test_processed_csv"],
        transform=get_eval_transforms(image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }