import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_X_PATH,
    TRAIN_Y_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)
from src.dataset import PCamDataset
from src.transforms import get_train_transforms
from src.model import build_model

def test_dataloader_and_training_step():
    print(f"Testing DataLoader & Step on Device: {DEVICE} (Batch Size: {BATCH_SIZE})")
    
    dataset = PCamDataset(
        image_path=TRAIN_X_PATH,
        label_path=TRAIN_Y_PATH,
        transform=get_train_transforms()
    )
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    model = build_model(pretrained=True).to(DEVICE)
    model.freeze_backbone()  # Stage 1
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
    
    print("\nRunning 5 training steps...")
    start_time = time.time()
    
    for i, (images, targets) in enumerate(loader):
        if i >= 5:
            break
            
        step_start = time.time()
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True).float().unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            
        step_dur = time.time() - step_start
        print(f"  Batch {i+1}: Loss = {loss.item():.4f} | Batch latency: {step_dur*1000:.1f} ms | Batch shape: {images.shape}")
        
    total_time = time.time() - start_time
    print(f"\n5 batches completed in {total_time:.2f}s ({total_time / 5 * 1000:.1f} ms/batch)")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak VRAM Allocated: {max_allocated:.2f} MB | Reserved: {reserved:.2f} MB")
        
    print("\n[SUCCESS] DataLoader and training step verified successfully!")

if __name__ == "__main__":
    test_dataloader_and_training_step()
