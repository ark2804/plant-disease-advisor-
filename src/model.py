"""
Transfer-learning model builder.

Uses a pretrained MobileNetV2 backbone (small, fast, good for deployment)
with a new classification head sized to the number of disease classes
for the given plant.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """
    Build a MobileNetV2-based classifier.

    Args:
        num_classes: number of disease classes (including 'Healthy') for this plant.
        freeze_backbone: if True, freeze the pretrained feature extractor and
            only train the new classification head. Set False (or unfreeze
            manually later) for fine-tuning once the head has converged.
        pretrained: if True, load pretrained ImageNet weights (used for training).
            Set False during inference when trained checkpoint weights are loaded.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module, num_layers_from_end: int = 20):
    """
    Unfreeze the last `num_layers_from_end` layers of the backbone for
    fine-tuning after the classification head has already converged.
    Call this, then continue training with a lower learning rate.
    """
    layers = list(model.features.children())
    for layer in layers[-num_layers_from_end:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
