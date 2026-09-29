import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_X_PATH,
    TRAIN_Y_PATH,
    VALID_X_PATH,
    VALID_Y_PATH,
    TEST_X_PATH,
    TEST_Y_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    WEIGHT_DECAY,
    CHECKPOINTS_DIR,
    METRICS_DIR
)
from src.dataset import PCamDataset
from src.transforms import get_train_transforms, get_eval_transforms
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import set_seed, load_checkpoint
from src.metrics import format_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on PatchCamelyon (PCam)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for DataLoader")
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS, help="Epochs for Stage 1 (Head warmup)")
    parser.add_argument("--stage1-lr", type=float, default=STAGE1_LR, help="Learning rate for Stage 1")
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS, help="Epochs for Stage 2 (Fine-tuning)")
    parser.add_argument("--stage2-lr", type=float, default=STAGE2_LR, help="Learning rate for Stage 2")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Optimizer weight decay")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Limit number of train batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Limit number of validation batches per epoch")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test split evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 65)
    print("PATCHCAMELYON (PCAM) HISTOPATHOLOGY CLASSIFICATION")
    print("=" * 65)
    print(f"Device:          {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Num Workers:     {args.num_workers}")
    print(f"Random Seed:     {args.seed}")
    print(f"Stage 1 Epochs:  {args.stage1_epochs} (LR: {args.stage1_lr})")
    print(f"Stage 2 Epochs:  {args.stage2_epochs} (LR: {args.stage2_lr})")
    if args.limit_train_batches:
        print(f"Limit Train Batches: {args.limit_train_batches}")
    if args.limit_val_batches:
        print(f"Limit Val Batches:   {args.limit_val_batches}")
    print("=" * 65)
    
    # 1. Datasets & Transforms
    print("\n[1/4] Initializing datasets and transforms...")
    train_dataset = PCamDataset(
        image_path=TRAIN_X_PATH,
        label_path=TRAIN_Y_PATH,
        transform=get_train_transforms()
    )
    val_dataset = PCamDataset(
        image_path=VALID_X_PATH,
        label_path=VALID_Y_PATH,
        transform=get_eval_transforms()
    )
    
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    
    # 2. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # 3. Model Architecture
    print("\n[2/4] Building ResNet50 model with ImageNet pretrained weights...")
    model = build_model(pretrained=True).to(DEVICE)
    print(f"  Total parameters:     {model.get_total_parameters_count():,}")
    
    # 4. Training
    print("\n[3/4] Starting training pipeline...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        device=DEVICE,
        use_amp=False,
        max_train_batches=args.limit_train_batches,
        max_val_batches=args.limit_val_batches
    )
    
    # 5. Final Evaluation on Test Set
    if not args.skip_test:
        print("\n[4/4] Evaluating best model checkpoint on untouched TEST set...")
        best_model_path = CHECKPOINTS_DIR / "best_model.pth"
        
        test_dataset = PCamDataset(
            image_path=TEST_X_PATH,
            label_path=TEST_Y_PATH,
            transform=get_eval_transforms()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"  Test samples: {len(test_dataset):,}")
        
        # Load best model checkpoint
        load_checkpoint(best_model_path, model, device=DEVICE)
        
        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_metrics, test_probs, test_preds, test_targets = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=DEVICE
        )
        
        print("\n" + "=" * 60)
        print("FINAL TEST SET EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}\n")
        print(format_metrics_report(test_metrics))
        print("=" * 60)
        
        # Save test metrics
        test_metrics_file = METRICS_DIR / "test_metrics.json"
        with open(test_metrics_file, "w") as f:
            json.dump({
                "test_loss": test_loss,
                "metrics": test_metrics
            }, f, indent=2)
        print(f"Test metrics saved to: {test_metrics_file}")


if __name__ == "__main__":
    main()
