import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.metrics import compute_metrics, format_metrics_report
from src.utils import set_seed

def test_metrics_and_utils():
    set_seed(42)
    
    # Ground truth: 4 tumor (1), 4 non-tumor (0)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Predictions: 3 TP, 1 FN, 3 TN, 1 FP
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 6 / 8, f"Expected 0.75, got {metrics['accuracy']}"
    assert metrics["confusion_matrix"]["tp"] == 3
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert abs(metrics["precision"] - (3 / 4)) < 1e-5
    assert abs(metrics["recall"] - (3 / 4)) < 1e-5
    assert abs(metrics["specificity"] - (3 / 4)) < 1e-5
    assert abs(metrics["f1_score"] - (3 / 4)) < 1e-5
    
    print("\n--- Metrics Test Output ---")
    print(format_metrics_report(metrics))
    print("\n[SUCCESS] Metrics and Utils verified successfully!")

if __name__ == "__main__":
    test_metrics_and_utils()
