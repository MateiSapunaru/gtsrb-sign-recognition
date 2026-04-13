from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_train_dataframe(train_dir: str | Path) -> pd.DataFrame:
    train_dir = Path(train_dir)
    rows = []

    class_dirs = sorted(
        [path for path in train_dir.iterdir() if path.is_dir()],
        key=lambda x: int(x.name),
    )

    for class_dir in class_dirs:
        label = int(class_dir.name)

        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".ppm"}:
                continue

            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No training images found in {train_dir}")

    return df


def build_test_dataframe(test_dir: str | Path, test_csv_path: str | Path) -> pd.DataFrame:
    test_dir = Path(test_dir)
    test_csv_path = Path(test_csv_path)

    df = pd.read_csv(test_csv_path)

    # Normalize column names in case spacing differs
    df.columns = [col.strip() for col in df.columns]

    if "Path" not in df.columns or "ClassId" not in df.columns:
        raise ValueError(
            f"Test CSV must contain 'Path' and 'ClassId' columns. Found: {list(df.columns)}"
        )

    def resolve_test_image(relative_path: str) -> str:
        relative_path = str(relative_path).replace("\\", "/")
        filename = Path(relative_path).name
        image_path = test_dir / filename
        return str(image_path.resolve())

    processed_df = pd.DataFrame(
        {
            "image_path": df["Path"].apply(resolve_test_image),
            "label": df["ClassId"].astype(int),
        }
    )

    missing_files = [
        path for path in processed_df["image_path"] if not Path(path).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} test images listed in CSV were not found. "
            f"Example: {missing_files[0]}"
        )

    return processed_df


def main() -> None:
    config = load_config()

    train_dir = config["data"]["train_dir"]
    test_dir = config["data"]["test_dir"]
    test_csv = config["data"]["test_csv"]

    train_csv_out = Path(config["data"]["train_csv"])
    val_csv_out = Path(config["data"]["val_csv"])
    test_csv_out = Path(config["data"]["test_processed_csv"])

    val_size = config["data"]["val_size"]
    seed = config["project"]["seed"]

    train_df_full = build_train_dataframe(train_dir)

    train_df, val_df = train_test_split(
        train_df_full,
        test_size=val_size,
        random_state=seed,
        stratify=train_df_full["label"],
    )

    test_df = build_test_dataframe(test_dir, test_csv)

    train_csv_out.parent.mkdir(parents=True, exist_ok=True)

    train_df = train_df.sort_values(["label", "image_path"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label", "image_path"]).reset_index(drop=True)
    test_df = test_df.sort_values(["label", "image_path"]).reset_index(drop=True)

    train_df.to_csv(train_csv_out, index=False)
    val_df.to_csv(val_csv_out, index=False)
    test_df.to_csv(test_csv_out, index=False)

    print("Data preparation complete.")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Saved: {train_csv_out}")
    print(f"Saved: {val_csv_out}")
    print(f"Saved: {test_csv_out}")


if __name__ == "__main__":
    main()