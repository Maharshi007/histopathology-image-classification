"""
EfficientNet-B3 Backbone for PCam Histopathology Classification
================================================================
Drop-in replacement for PCamResNet50 using EfficientNet-B3 pretrained
on ImageNet. Exposes the same freeze_backbone / unfreeze_all / unfreeze_layer4_and_head
API so it works with the existing two-stage training pipeline in src/train.py
without any changes to train logic.

Key advantages over ResNet50:
  - Better accuracy/parameter ratio (EfficientNet family)
  - 1536-dim feature vector (vs 2048) - less overfitting risk
  - Compound scaling: depth + width + resolution optimized together
  - ~12M params vs ~25M for ResNet50

Architecture:
  EfficientNet-B3 backbone
  -> Global Average Pooling (AdaptiveAvgPool2d -> 1536-dim)
  -> Dropout(p=0.3)
  -> Linear(1536, 1)   [binary logit]

Usage:
  from src.efficientnet_model import build_efficientnet_model
  model = build_efficientnet_model(pretrained=True, dropout_rate=0.3)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from typing import Optional


class PCamEfficientNetB3(nn.Module):
    """
    EfficientNet-B3 adapted for PatchCamelyon (PCam) binary classification.

    Architecture:
      - Backbone : EfficientNet-B3 pretrained on ImageNet-1K
      - Head     : Dropout(p=0.3) -> Linear(1536, 1) -> scalar logit
      - Loss     : BCEWithLogitsLoss (or FocalLoss)
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b3(weights=weights)

        # EfficientNet-B3's classifier: [Dropout, Linear(1536, 1000)]
        in_features = self.efficientnet.classifier[1].in_features  # 1536

        # Replace ImageNet 1000-class head with binary classification head
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) raw logits."""
        return self.efficientnet(x)

    def freeze_backbone(self) -> None:
        """
        Stage 1 Transfer Learning:
        Freeze all EfficientNet feature extractor layers.
        Only the classification head remains trainable.
        """
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True

    def unfreeze_layer4_and_head(self) -> None:
        """
        Selective Fine-Tuning:
        Unfreeze the last feature block (features[-1]) and classifier.
        Equivalent to ResNet's layer4 strategy.
        """
        self.freeze_backbone()
        # EfficientNet features is an nn.Sequential; unfreeze last block
        for param in self.efficientnet.features[-1].parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Stage 2 Transfer Learning:
        Unfreeze all parameters for end-to-end fine-tuning.
        """
        for param in self.efficientnet.parameters():
            param.requires_grad = True

    def get_trainable_parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_efficientnet_model(
    pretrained:   bool  = True,
    dropout_rate: float = 0.3
) -> PCamEfficientNetB3:
    """
    Factory function to instantiate the PCam EfficientNet-B3 classifier.

    Args:
        pretrained   : Load ImageNet-pretrained weights (recommended True)
        dropout_rate : Dropout probability in the classification head

    Returns:
        PCamEfficientNetB3 model instance
    """
    model = PCamEfficientNetB3(pretrained=pretrained, dropout_rate=dropout_rate)
    total = model.get_total_parameters_count()
    print(f"[EfficientNet-B3] Total parameters: {total:,}")
    return model
