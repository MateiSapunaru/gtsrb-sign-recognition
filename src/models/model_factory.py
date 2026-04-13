from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Supported: ['efficientnet_b0', 'resnet18']"
    )


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    model_name = model_name.lower()

    for param in model.parameters():
        param.requires_grad = False

    if model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    if model_name == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unsupported model_name='{model_name}'")


def unfreeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True