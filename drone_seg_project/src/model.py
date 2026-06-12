"""
model.py

Builds a torchvision semantic segmentation model (default: DeepLabV3-ResNet101,
ImageNet/COCO pretrained) and re-heads its classifier(s) for the project's
12-class drone segmentation task.
"""

import torch.nn as nn
from torchvision.models import segmentation

from src.config import ModelConfig


def prepare_model(model_name="deeplabv3_resnet101", use_pretrained=True, num_classes=12):
    """Loads a torchvision segmentation model and replaces its classifier head(s)
    so the output channel count matches `num_classes`.

    For DeepLabV3 / FCN models, both the main classifier and (if present) the
    auxiliary classifier are re-headed. For LRASPP, the low- and high-resolution
    classifiers are replaced instead.
    """
    weights = "DEFAULT" if use_pretrained else None

    try:
        print(f"Loading model: {model_name}, pretrained={use_pretrained}")
        model = getattr(segmentation, model_name.lower())(weights=weights, aux_loss=ModelConfig.AUX_LOSS)
    except Exception as e:
        print(e)
        print("Falling back to pretrained DeepLabV3-ResNet101.")
        model = segmentation.deeplabv3_resnet101(weights="DEFAULT")

    if "lraspp" in model_name.lower():
        model.low_classifier = nn.Conv2d(in_channels=40, out_channels=num_classes, kernel_size=1, stride=1)
        model.high_classifier = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1)
    else:
        model.classifier[-1] = nn.LazyConv2d(out_channels=num_classes, kernel_size=1, stride=1)

        if use_pretrained:
            if ModelConfig.AUX_LOSS:
                model.aux_classifier[-1] = nn.LazyConv2d(out_channels=num_classes, kernel_size=1, stride=1)
            else:
                model.aux_classifier = nn.Identity()

    return model
