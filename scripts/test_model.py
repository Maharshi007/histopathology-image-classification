import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model import build_model
from src.config import DEVICE

def test_model():
    print(f"Testing Model Initialization on Device: {DEVICE}")
    model = build_model(pretrained=True).to(DEVICE)
    
    total_params = model.get_total_parameters_count()
    print(f"Total Parameters: {total_params:,}")
    
    # Test Stage 1: Frozen Backbone
    model.freeze_backbone()
    trainable_stage1 = model.get_trainable_parameters_count()
    print(f"Stage 1 Trainable Parameters (Head only): {trainable_stage1:,}")
    assert trainable_stage1 == 2048 * 1 + 1, f"Expected 2049, got {trainable_stage1}"
    
    # Test Stage 2a: Selective Unfreeze (Layer 4 + Head)
    model.unfreeze_layer4_and_head()
    trainable_stage2a = model.get_trainable_parameters_count()
    print(f"Selective Stage Trainable Parameters (Layer 4 + Head): {trainable_stage2a:,}")
    
    # Test Stage 2b: Unfreeze All
    model.unfreeze_all()
    trainable_stage2b = model.get_trainable_parameters_count()
    print(f"Stage 2 Trainable Parameters (All): {trainable_stage2b:,}")
    assert trainable_stage2b == total_params
    
    # Test Forward Pass
    dummy_input = torch.randn(4, 3, 96, 96, device=DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Dummy Forward Pass Output Shape: {output.shape}")
    assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU VRAM Allocated: {allocated_mb:.2f} MB | Reserved: {reserved_mb:.2f} MB")
        
    print("\n[SUCCESS] ResNet50 model architecture verified successfully!")

if __name__ == "__main__":
    test_model()
