import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    VALID_X_PATH,
    VALID_Y_PATH,
    TEST_X_PATH,
    TEST_Y_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    CHECKPOINTS_DIR,
    METRICS_DIR
)
from src.dataset import PCamDataset
from src.transforms import get_eval_transforms
from src.model import build_model
from src.evaluate import evaluate_model
from src.utils import load_checkpoint
from src.metrics import format_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet50 model on PCam dataset")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_model.pth"),
                        help="Path to model checkpoint .pth file")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test",
                        help="Dataset split to evaluate on ('valid' or 'test')")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for binary classification (default: 0.5). "
                             "Use 0.28 for optimal F1/Recall based on threshold sweep.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print("=" * 60)
    print(f"EVALUATING MODEL ON {args.split.upper()} SET")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {DEVICE}")
    print(f"Split:      {args.split}")
    
    # Select split paths
    if args.split == "valid":
        x_path, y_path = VALID_X_PATH, VALID_Y_PATH
    else:
        x_path, y_path = TEST_X_PATH, TEST_Y_PATH
        
    dataset = PCamDataset(
        image_path=x_path,
        label_path=y_path,
        transform=get_eval_transforms()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Total samples: {len(dataset):,}")
    print(f"Threshold:     {args.threshold}" +
          ("  (optimal — from threshold sweep)" if abs(args.threshold - 0.5) > 0.01 else "  (default)"))
    
    # Load model
    model = build_model(pretrained=False).to(DEVICE)
    load_checkpoint(checkpoint_path, model, device=DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    loss, metrics, probs, preds, targets = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=DEVICE,
        threshold=args.threshold
    )
    
    print("\n" + "=" * 60)
    print(f"RESULTS ON {args.split.upper()} SET")
    print("=" * 60)
    print(f"Average Loss: {loss:.4f}\n")
    print(format_metrics_report(metrics))
    print("=" * 60)
    
    # Save results
    threshold_tag = f"_t{args.threshold:.2f}".replace(".", "p") if abs(args.threshold - 0.5) > 0.01 else ""
    output_file = METRICS_DIR / f"{args.split}_evaluation_results{threshold_tag}.json"
    with open(output_file, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "threshold": args.threshold,
            "loss": loss,
            "metrics": metrics
        }, f, indent=2)
    print(f"Evaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()
