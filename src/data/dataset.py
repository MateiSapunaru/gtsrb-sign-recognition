from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """
    Dataset class for GTSRB using a CSV file with:
    - image_path
    - label

    The image_path can be relative or absolute.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

        required_columns = {"image_path", "label"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"CSV file {self.csv_path} is missing required columns: {missing_columns}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label