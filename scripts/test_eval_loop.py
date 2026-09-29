import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import VALID_X_PATH, VALID_Y_PATH, DEVICE
from src.dataset import PCamDataset
from src.transforms import get_eval_transforms
from src.model import build_model
from src.evaluate import evaluate_model
from src.metrics import format_metrics_report

def test_eval_loop():
    print("Testing evaluate_model loop on a small subset of validation data...")
    val_ds = PCamDataset(
        image_path=VALID_X_PATH,
        label_path=VALID_Y_PATH,
        transform=get_eval_transforms()
    )
    subset = Subset(val_ds, indices=list(range(128)))
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    val_loss, metrics, probs, preds, targets = evaluate_model(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=DEVICE
    )
    
    print(f"Loss: {val_loss:.4f}")
    print(format_metrics_report(metrics))
    assert len(probs) == 128
    assert len(preds) == 128
    assert len(targets) == 128
    print("\n[SUCCESS] evaluate_model loop verified successfully without errors!")

if __name__ == "__main__":
    test_eval_loop()
