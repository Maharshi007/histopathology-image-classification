"""
Experiment Training Script
===========================
Extends the base training pipeline with:
  - Choice of backbone: ResNet50 | EfficientNet-B3
  - Choice of loss: BCE | FocalLoss (alpha, gamma configurable)
  - Optional pos_weight for BCE to further penalise FNs
  - Enhanced stain augmentation (color jitter for H&E variation)
  - All CLI args so you can grid-search from a shell script

Examples:
  # Baseline (reproduce existing run)
  .venv\\Scripts\\python.exe scripts\\train_experiment.py --backbone resnet50 --loss bce

  # Focal Loss experiment (recommended for improving Recall)
  .venv\\Scripts\\python.exe scripts\\train_experiment.py --backbone resnet50 --loss focal --alpha 0.75 --gamma 2.0

  # EfficientNet-B3 with Focal Loss
  .venv\\Scripts\\python.exe scripts\\train_experiment.py --backbone efficientnet_b3 --loss focal --alpha 0.75 --gamma 2.0

  # BCE + pos_weight (simple class re-weighting, no focal)
  .venv\\Scripts\\python.exe scripts\\train_experiment.py --backbone resnet50 --loss bce --pos-weight 2.0

  # EfficientNet-B3 baseline
  .venv\\Scripts\\python.exe scripts\\train_experiment.py --backbone efficientnet_b3 --loss bce --batch-size 32
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from src.config import (
    TRAIN_X_PATH, TRAIN_Y_PATH,
    VALID_X_PATH, VALID_Y_PATH,
    TEST_X_PATH,  TEST_Y_PATH,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR,
    STAGE1_EPOCHS, STAGE1_LR,
    STAGE2_EPOCHS, STAGE2_LR,
    WEIGHT_DECAY, SEED
)
from src.dataset    import PCamDataset
from src.transforms import get_train_transforms, get_eval_transforms
from src.metrics    import compute_metrics, format_metrics_report
from src.evaluate   import evaluate_model
from src.utils      import save_checkpoint, load_checkpoint, set_seed
from src.focal_loss import build_focal_loss


# ---- Enhanced augmentation with stain/color jitter --------------------------

def get_enhanced_train_transforms():
    """
    Adds ColorJitter to the standard transforms to simulate H&E stain variation.
    Histopathology slides can vary significantly in staining intensity between
    labs and slide preparations, making color robustness essential.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
        # Simulate H&E stain variation: subtle brightness/contrast/saturation shifts
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ---- Model factory ----------------------------------------------------------

def build_model_from_args(args) -> nn.Module:
    if args.backbone == "resnet50":
        from src.model import build_model
        model = build_model(pretrained=True, dropout_rate=args.dropout)
    elif args.backbone == "efficientnet_b3":
        from src.efficientnet_model import build_efficientnet_model
        model = build_efficientnet_model(pretrained=True, dropout_rate=args.dropout)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{args.backbone}] Total parameters: {total:,}")
    return model


# ---- Loss factory -----------------------------------------------------------

def build_criterion(args) -> nn.Module:
    if args.loss == "bce":
        if args.pos_weight is not None:
            pw = torch.tensor([args.pos_weight], device=DEVICE)
            print(f"  [Loss] BCEWithLogitsLoss  pos_weight={args.pos_weight:.2f}")
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        print("  [Loss] BCEWithLogitsLoss (standard)")
        return nn.BCEWithLogitsLoss()
    elif args.loss == "focal":
        print(f"  [Loss] BinaryFocalLoss  alpha={args.alpha:.2f}  gamma={args.gamma:.2f}")
        return build_focal_loss(alpha=args.alpha, gamma=args.gamma,
                                pos_weight=args.pos_weight)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


# ---- One epoch ---------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, criterion,
                    device, scaler=None, log_interval=200,
                    max_batches: Optional[int] = None):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets   = [], []
    start = time.time()
    total_steps = min(len(dataloader), max_batches) if max_batches else len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss    = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        bs           = images.size(0)
        total_loss  += loss.item() * bs
        total_samples += bs

        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().flatten().tolist())
            all_targets.extend(targets.cpu().flatten().long().tolist())

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_steps:
            elapsed = time.time() - start
            print(f"  Batch [{batch_idx+1}/{total_steps}]  Loss: {total_loss/total_samples:.4f}  ({elapsed:.1f}s)")

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics    = compute_metrics(all_targets, all_preds)
    return epoch_loss, metrics


# ---- Two-Stage Training ------------------------------------------------------

def train_experiment(model, train_loader, val_loader, criterion, args):
    best_val_f1   = -1.0
    best_epoch    = -1
    run_name      = f"{args.backbone}_{args.loss}"
    if args.loss == "focal":
        run_name += f"_a{args.alpha}_g{args.gamma}"
    if args.pos_weight:
        run_name += f"_pw{args.pos_weight}"

    ckpt_dir   = CHECKPOINTS_DIR
    best_ckpt  = ckpt_dir / f"best_{run_name}.pth"
    final_ckpt = ckpt_dir / f"final_{run_name}.pth"

    scaler = torch.cuda.amp.GradScaler() if (args.amp and DEVICE.type == "cuda") else None

    history = {k: [] for k in [
        "epoch", "stage",
        "train_loss", "train_acc", "train_recall", "train_precision", "train_f1",
        "val_loss",   "val_acc",   "val_recall",   "val_precision",   "val_f1",
        "lr"
    ]}
    total_epochs  = args.stage1_epochs + args.stage2_epochs
    current_epoch = 0

    def record(stage, ep, t_loss, t_m, v_loss, v_m, lr):
        history["epoch"].append(ep)
        history["stage"].append(stage)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_m["accuracy"])
        history["train_recall"].append(t_m["recall"])
        history["train_precision"].append(t_m["precision"])
        history["train_f1"].append(t_m["f1_score"])
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_m["accuracy"])
        history["val_recall"].append(v_m["recall"])
        history["val_precision"].append(v_m["precision"])
        history["val_f1"].append(v_m["f1_score"])
        history["lr"].append(lr)

    def maybe_save(current_epoch, stage, val_metrics, optimizer):
        nonlocal best_val_f1, best_epoch
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_epoch  = current_epoch
            print(f"  [*] New best val F1: {best_val_f1*100:.2f}%  -> {best_ckpt.name}")
            save_checkpoint({
                "epoch": current_epoch, "stage": stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "val_metrics": val_metrics,
                "run_name": run_name
            }, best_ckpt)

    # ------------------------------------------------------------------
    # STAGE 1: Head Warmup
    # ------------------------------------------------------------------
    if args.stage1_epochs > 0:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Head Warmup  ({args.stage1_epochs} epochs, lr={args.stage1_lr})")
        print("=" * 60)
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.stage1_lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True
        )
        for ep in range(1, args.stage1_epochs + 1):
            current_epoch += 1
            print(f"\n--- Epoch {current_epoch}/{total_epochs}  (Stage 1: {ep}/{args.stage1_epochs}) ---")
            t_loss, t_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
            v_loss, v_m, _, _, _ = evaluate_model(model, val_loader, nn.BCEWithLogitsLoss(), DEVICE)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(v_m["f1_score"])
            record(1, current_epoch, t_loss, t_m, v_loss, v_m, lr)
            print(f"  Train | Loss:{t_loss:.4f}  Acc:{t_m['accuracy']*100:.2f}%  Recall:{t_m['recall']*100:.2f}%  F1:{t_m['f1_score']*100:.2f}%")
            print(f"  Val   | Loss:{v_loss:.4f}  Acc:{v_m['accuracy']*100:.2f}%  Recall:{v_m['recall']*100:.2f}%  F1:{v_m['f1_score']*100:.2f}%")
            maybe_save(current_epoch, 1, v_m, optimizer)

    # ------------------------------------------------------------------
    # STAGE 2: End-to-End Fine-Tuning
    # ------------------------------------------------------------------
    if args.stage2_epochs > 0:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Full Fine-Tuning  ({args.stage2_epochs} epochs, lr={args.stage2_lr})")
        print("=" * 60)
        model.unfreeze_all()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True
        )
        for ep in range(1, args.stage2_epochs + 1):
            current_epoch += 1
            print(f"\n--- Epoch {current_epoch}/{total_epochs}  (Stage 2: {ep}/{args.stage2_epochs}) ---")
            t_loss, t_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
            v_loss, v_m, _, _, _ = evaluate_model(model, val_loader, nn.BCEWithLogitsLoss(), DEVICE)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(v_m["f1_score"])
            record(2, current_epoch, t_loss, t_m, v_loss, v_m, lr)
            print(f"  Train | Loss:{t_loss:.4f}  Acc:{t_m['accuracy']*100:.2f}%  Recall:{t_m['recall']*100:.2f}%  F1:{t_m['f1_score']*100:.2f}%")
            print(f"  Val   | Loss:{v_loss:.4f}  Acc:{v_m['accuracy']*100:.2f}%  Recall:{v_m['recall']*100:.2f}%  F1:{v_m['f1_score']*100:.2f}%")
            maybe_save(current_epoch, 2, v_m, optimizer)

    # Save final checkpoint
    save_checkpoint({"epoch": current_epoch, "model_state_dict": model.state_dict()}, final_ckpt)

    # Save history
    hist_path = METRICS_DIR / f"history_{run_name}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Best Val F1: {best_val_f1*100:.2f}%  (epoch {best_epoch})")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  History: {hist_path}")
    print("=" * 60)
    return best_ckpt, history


# ---- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PCam experiment training script")

    p.add_argument("--backbone",      type=str,   default="resnet50",
                   choices=["resnet50", "efficientnet_b3"],
                   help="Model backbone architecture")
    p.add_argument("--loss",          type=str,   default="focal",
                   choices=["bce", "focal"],
                   help="Loss function")
    p.add_argument("--alpha",         type=float, default=0.75,
                   help="[Focal] Alpha weighting for positive class (default: 0.75)")
    p.add_argument("--gamma",         type=float, default=2.0,
                   help="[Focal] Focusing gamma (default: 2.0)")
    p.add_argument("--pos-weight",    type=float, default=None,
                   help="[BCE/Focal] Extra pos_weight scalar for positive class")
    p.add_argument("--dropout",       type=float, default=0.3,
                   help="Dropout rate in classification head")
    p.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--stage1-epochs", type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--stage2-epochs", type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--stage1-lr",     type=float, default=STAGE1_LR)
    p.add_argument("--stage2-lr",     type=float, default=STAGE2_LR)
    p.add_argument("--weight-decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--aug",           type=str,   default="enhanced",
                   choices=["standard", "enhanced"],
                   help="Training augmentation strategy (enhanced adds color jitter)")
    p.add_argument("--amp",           action="store_true",
                   help="Enable Automatic Mixed Precision (CUDA only)")
    p.add_argument("--eval-test",     action="store_true",
                   help="After training, evaluate best checkpoint on test set")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 62)
    print("  PCam EXPERIMENT TRAINING")
    print("=" * 62)
    print(f"  Backbone      : {args.backbone}")
    print(f"  Loss          : {args.loss}" +
          (f"  alpha={args.alpha}  gamma={args.gamma}" if args.loss == "focal" else "") +
          (f"  pos_weight={args.pos_weight}" if args.pos_weight else ""))
    print(f"  Augmentation  : {args.aug}")
    print(f"  Device        : {DEVICE}")
    print(f"  Stage 1       : {args.stage1_epochs} epochs @ lr={args.stage1_lr}")
    print(f"  Stage 2       : {args.stage2_epochs} epochs @ lr={args.stage2_lr}")
    print()

    # Transforms
    train_tf = get_enhanced_train_transforms() if args.aug == "enhanced" else get_train_transforms()
    eval_tf  = get_eval_transforms()

    # Datasets
    train_ds = PCamDataset(TRAIN_X_PATH, TRAIN_Y_PATH, transform=train_tf)
    val_ds   = PCamDataset(VALID_X_PATH, VALID_Y_PATH, transform=eval_tf)
    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val samples   : {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())

    # Model + loss
    model     = build_model_from_args(args).to(DEVICE)
    criterion = build_criterion(args)

    # Train
    best_ckpt, history = train_experiment(model, train_loader, val_loader, criterion, args)

    # Optional test-set evaluation
    if args.eval_test:
        print("\n[Test Set Evaluation]")
        test_ds     = PCamDataset(TEST_X_PATH, TEST_Y_PATH, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=torch.cuda.is_available())

        # Reload best checkpoint for clean eval
        model_eval = build_model_from_args(args).to(DEVICE)
        load_checkpoint(best_ckpt, model_eval, device=DEVICE)

        t_loss, t_metrics, _, _, _ = evaluate_model(
            model_eval, test_loader, nn.BCEWithLogitsLoss(), DEVICE
        )
        run_name = best_ckpt.stem.replace("best_", "")
        print(f"\nTest Loss: {t_loss:.4f}")
        print(format_metrics_report(t_metrics))

        result_path = METRICS_DIR / f"test_metrics_{run_name}.json"
        with open(result_path, "w") as f:
            json.dump({"run": run_name, "test_loss": t_loss, "metrics": t_metrics}, f, indent=2)
        print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
