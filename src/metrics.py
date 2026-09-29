import numpy as np
from typing import Dict, Union, Tuple, Any
import torch


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7
) -> Dict[str, Union[float, Dict[str, int]]]:
    """
    Compute medical classification metrics for binary classification.
    
    Positive Class (1): Tumor
    Negative Class (0): Non-tumor
    
    Returns:
        accuracy: (TP + TN) / Total
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)  [Sensitivity / True Positive Rate for Tumor]
        specificity: TN / (TN + FP) [True Negative Rate for Non-Tumor]
        f1_score: 2 * (Precision * Recall) / (Precision + Recall)
        confusion_matrix: {'tp': TP, 'tn': TN, 'fp': FP, 'fn': FN}
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true = np.asarray(y_true, dtype=int).flatten()
    y_pred = np.asarray(y_pred, dtype=int).flatten()
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format computed metrics into a clean text summary.
    """
    cm = metrics["confusion_matrix"]
    report = (
        f"Accuracy:    {metrics['accuracy'] * 100:.2f}%\n"
        f"Precision:   {metrics['precision'] * 100:.2f}%\n"
        f"Recall/Sens: {metrics['recall'] * 100:.2f}%\n"
        f"Specificity: {metrics['specificity'] * 100:.2f}%\n"
        f"F1-Score:    {metrics['f1_score'] * 100:.2f}%\n"
        f"Confusion Matrix: [TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}]"
    )
    return report
