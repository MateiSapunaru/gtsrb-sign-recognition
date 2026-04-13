from __future__ import annotations

from typing import Tuple

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int) -> transforms.Compose:
    """
    Build training transforms for transfer learning on GTSRB.

    Args:
        image_size: Final square image size expected by the model.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int) -> transforms.Compose:
    """
    Build deterministic validation/test transforms.

    Args:
        image_size: Final square image size expected by the model.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_inverse_normalization() -> transforms.Normalize:
    """
    Return a transform that approximately reverses ImageNet normalization.
    Useful for visualization.

    Returns:
        torchvision.transforms.Normalize
    """
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    inverse_mean = [-m / s for m, s in zip(mean, std)]
    inverse_std = [1 / s for s in std]

    return transforms.Normalize(mean=inverse_mean, std=inverse_std)