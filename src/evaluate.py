import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
from src.metrics import compute_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, Any], List[float], List[int], List[int]]:
    """
    Evaluate model on a given dataset split (validation or test).

    Args:
        model:       PyTorch neural network model
        dataloader:  DataLoader for the evaluation set
        criterion:   Loss function (e.g. BCEWithLogitsLoss)
        device:      Execution device (cuda or cpu)
        max_batches: Optional limit on the number of batches evaluated
        threshold:   Decision threshold for binary classification (default 0.5).
                     Lower values (e.g. 0.28) increase Recall at the cost of
                     Precision — preferred for medical screening tasks where
                     missing a tumour (FN) is costlier than a false alarm (FP).

    Returns:
        avg_loss:    Average scalar loss across all batches
        metrics:     Computed medical classification metrics dictionary
        all_probs:   Predicted probabilities (Sigmoid output)
        all_preds:   Binary predicted classes (0 or 1) at the given threshold
        all_targets: Ground truth classes (0 or 1)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_probs   = []
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)  # Raw logits: shape (B, 1)
            loss    = criterion(outputs, targets.float().unsqueeze(1))

            batch_size     = images.size(0)
            total_loss    += loss.item() * batch_size
            total_samples += batch_size

            # Probabilities and binary predictions at the given threshold
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics  = compute_metrics(all_targets, all_preds)

    return avg_loss, metrics, all_probs, all_preds, all_targets

