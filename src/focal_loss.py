"""
Focal Loss Implementation for PCam Histopathology Classification
================================================================
Addresses the high False-Negative rate by down-weighting easy negatives
and focusing training on hard, misclassified positive (tumor) samples.

Drop-in replacement for nn.BCEWithLogitsLoss in the existing train pipeline.

Reference:
  Lin et al. (2017) "Focal Loss for Dense Object Detection"
  https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with optional class-weighting for imbalanced datasets.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Positive-class weight in [0, 1].
                       alpha=0.75 up-weights tumor (positive) class.
                       Set to None to disable class weighting.
        gamma (float): Focusing parameter. gamma=0 reduces to BCE.
                       gamma=2 is the standard recommended value.
        pos_weight (float | None): Additional scalar multiplier on positive
                                   class loss (like BCEWithLogitsLoss.pos_weight).
                                   Useful to further penalize FNs.
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha:      float = 0.75,
        gamma:      float = 2.0,
        pos_weight: Optional[float] = None,
        reduction:  str   = "mean"
    ):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.reduction  = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : Raw model outputs, shape (B,) or (B, 1)
            targets : Binary ground-truth labels, same shape as logits
        Returns:
            Scalar focal loss (if reduction != 'none')
        """
        logits  = logits.view(-1)
        targets = targets.view(-1).float()

        # Stable BCE from logits: log(sigmoid(x)) = -softplus(-x)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t  = sigmoid(logit) for positives,  1 - sigmoid(logit) for negatives
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weight: alpha for positives, (1-alpha) for negatives
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        # Optional extra pos_weight multiplier on positive samples
        if self.pos_weight is not None:
            pw = torch.ones_like(targets)
            pw[targets == 1] = self.pos_weight
            loss = loss * pw

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_focal_loss(
    alpha:      float = 0.75,
    gamma:      float = 2.0,
    pos_weight: Optional[float] = None,
) -> BinaryFocalLoss:
    """
    Factory: returns a configured BinaryFocalLoss.

    Recommended settings for high-recall medical classification:
      alpha=0.75  - up-weights tumor (FN cost > FP cost)
      gamma=2.0   - standard focusing strength
      pos_weight  - set to ~1.5-2.0 if FNs are still too high after tuning alpha

    Example usage in training loop:
        from src.focal_loss import build_focal_loss
        criterion = build_focal_loss(alpha=0.75, gamma=2.0)
        loss = criterion(outputs, targets.float().unsqueeze(1))
    """
    return BinaryFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
