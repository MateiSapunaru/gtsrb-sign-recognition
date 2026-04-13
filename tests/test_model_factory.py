import torch

from src.models.model_factory import build_model


def test_build_efficientnet_output_shape():
    model = build_model(
        model_name="efficientnet_b0",
        num_classes=43,
        pretrained=False,
        dropout=0.3,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert y.shape == (2, 43)


def test_build_resnet_output_shape():
    model = build_model(
        model_name="resnet18",
        num_classes=43,
        pretrained=False,
        dropout=0.3,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert y.shape == (2, 43)