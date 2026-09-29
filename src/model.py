import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


class PCamResNet50(nn.Module):
    """
    ResNet50 model adapted for PatchCamelyon (PCam) binary histopathology classification.
    
    Architecture details:
    - Backbone: ResNet50 pretrained on ImageNet-1K.
    - Global Average Pooling: Outputs 2048-dim feature vector.
    - Classification Head: Dropout (p=0.3) -> Linear (2048 -> 1).
    - Output: Raw scalar logit for BCEWithLogitsLoss.
    """
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = resnet50(weights=weights)
        
        in_features = self.resnet.fc.in_features  # 2048
        
        # Replace 1000-class ImageNet head with binary classification head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning (B, 1) raw logits.
        """
        return self.resnet(x)

    def freeze_backbone(self) -> None:
        """
        Stage 1 Transfer Learning:
        Freeze all backbone convolutional feature extractors.
        Only the classification head (fc) parameters remain trainable.
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze_layer4_and_head(self) -> None:
        """
        Selective Fine-Tuning:
        Keep early low-level feature layers (layer1-3) frozen.
        Unfreeze high-level task-specific layer4 and the classification head.
        """
        self.freeze_backbone()
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Stage 2 Transfer Learning:
        Unfreeze all backbone parameters for end-to-end fine-tuning.
        """
        for param in self.resnet.parameters():
            param.requires_grad = True

    def get_trainable_parameters_count(self) -> int:
        """
        Return count of currently trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters_count(self) -> int:
        """
        Return total parameter count.
        """
        return sum(p.numel() for p in self.parameters())


def build_model(pretrained: bool = True, dropout_rate: float = 0.3) -> PCamResNet50:
    """
    Factory function to instantiate the PCam ResNet50 classifier.
    """
    return PCamResNet50(pretrained=pretrained, dropout_rate=dropout_rate)
