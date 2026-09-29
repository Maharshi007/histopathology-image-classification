import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.config import (
    DEVICE,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    WEIGHT_DECAY
)
from src.metrics import compute_metrics, format_metrics_report
from src.evaluate import evaluate_model
from src.utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 200,
    max_batches: Optional[int] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Train model for one epoch.
    
    Supports automatic mixed precision (AMP) if scaler is provided.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    total_steps = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().flatten().tolist())
            all_targets.extend(targets.cpu().flatten().long().tolist())
            
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_steps:
            current_loss = total_loss / total_samples
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx + 1}/{total_steps}] - Loss: {current_loss:.4f} ({elapsed:.1f}s)")
            
    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_metrics(all_targets, all_preds)
    
    return epoch_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage1_lr: float = STAGE1_LR,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage2_lr: float = STAGE2_LR,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = DEVICE,
    use_amp: bool = False,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Two-Stage Transfer Learning Training Pipeline:
    
    Stage 1: Head Warmup (Backbone frozen, train classification head)
    Stage 2: End-to-End Fine-Tuning (Backbone unfrozen, low learning rate)
    """
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = CHECKPOINTS_DIR / "best_model.pth"
    final_model_path = CHECKPOINTS_DIR / "final_model.pth"
    
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "train_recall": [],
        "train_precision": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_recall": [],
        "val_precision": [],
        "val_f1": [],
        "lr": []
    }
    
    total_epochs = stage1_epochs + stage2_epochs
    current_epoch = 0
    
    # -------------------------------------------------------------
    # STAGE 1: Train Head (Backbone Frozen)
    # -------------------------------------------------------------
    if stage1_epochs > 0:
        print("\n" + "=" * 60)
        print(f"STAGE 1: Training Classification Head ({stage1_epochs} epochs, lr={stage1_lr})")
        print("=" * 60)
        
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage1_lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True
        )
        
        for epoch_in_stage in range(1, stage1_epochs + 1):
            current_epoch += 1
            print(f"\n--- Epoch {current_epoch}/{total_epochs} (Stage 1: {epoch_in_stage}/{stage1_epochs}) ---")
            
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler=scaler,
                max_batches=max_train_batches
            )
            
            val_loss, val_metrics, _, _, _ = evaluate_model(
                model, val_loader, criterion, device,
                max_batches=max_val_batches
            )
            
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["f1_score"])
            
            # Record metrics
            history["epoch"].append(current_epoch)
            history["stage"].append(1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_f1"].append(train_metrics["f1_score"])
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_f1"].append(val_metrics["f1_score"])
            history["lr"].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | Recall: {train_metrics['recall']*100:.2f}% | F1: {train_metrics['f1_score']*100:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | Recall: {val_metrics['recall']*100:.2f}% | F1: {val_metrics['f1_score']*100:.2f}%")
            
            # Checkpoint best model based on validation F1
            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                best_epoch = current_epoch
                print(f"[*] New best validation F1: {best_val_f1*100:.2f}% (Saving checkpoint to {best_model_path.name})")
                save_checkpoint({
                    "epoch": current_epoch,
                    "stage": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "val_metrics": val_metrics
                }, best_model_path)

    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning (Unfreeze Backbone)
    # -------------------------------------------------------------
    if stage2_epochs > 0:
        print("\n" + "=" * 60)
        print(f"STAGE 2: Fine-Tuning Backbone ({stage2_epochs} epochs, lr={stage2_lr})")
        print("=" * 60)
        
        model.unfreeze_all()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage2_lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True
        )
        
        for epoch_in_stage in range(1, stage2_epochs + 1):
            current_epoch += 1
            print(f"\n--- Epoch {current_epoch}/{total_epochs} (Stage 2: {epoch_in_stage}/{stage2_epochs}) ---")
            
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler=scaler,
                max_batches=max_train_batches
            )
            
            val_loss, val_metrics, _, _, _ = evaluate_model(
                model, val_loader, criterion, device,
                max_batches=max_val_batches
            )
            
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["f1_score"])
            
            # Record metrics
            history["epoch"].append(current_epoch)
            history["stage"].append(2)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_f1"].append(train_metrics["f1_score"])
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_f1"].append(val_metrics["f1_score"])
            history["lr"].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | Recall: {train_metrics['recall']*100:.2f}% | F1: {train_metrics['f1_score']*100:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | Recall: {val_metrics['recall']*100:.2f}% | F1: {val_metrics['f1_score']*100:.2f}%")
            
            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                best_epoch = current_epoch
                print(f"[*] New best validation F1: {best_val_f1*100:.2f}% (Saving checkpoint to {best_model_path.name})")
                save_checkpoint({
                    "epoch": current_epoch,
                    "stage": 2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "val_metrics": val_metrics
                }, best_model_path)

    # Save final model
    save_checkpoint({
        "epoch": current_epoch,
        "model_state_dict": model.state_dict(),
        "final_val_f1": history["val_f1"][-1] if history["val_f1"] else None
    }, final_model_path)
    
    # Save training history JSON
    history_path = METRICS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation F1: {best_val_f1*100:.2f}% (Epoch {best_epoch})")
    print(f"Checkpoints saved in: {CHECKPOINTS_DIR}")
    print(f"History saved in: {history_path}")
    print("=" * 60)
    
    return history
