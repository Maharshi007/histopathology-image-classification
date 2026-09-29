import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import build_model
from src.gradcam import GradCAM, apply_colormap_on_image
from src.config import DEVICE

def test_gradcam():
    print(f"Testing Grad-CAM on Device: {DEVICE}")
    model = build_model(pretrained=True).to(DEVICE)
    gradcam = GradCAM(model=model)
    
    # Dummy input (1, 3, 96, 96)
    dummy_input = torch.randn(1, 3, 96, 96, device=DEVICE)
    cam, prob, pred = gradcam.generate_cam(dummy_input)
    
    print("CAM shape:", cam.shape)
    print("CAM min/max:", cam.min(), cam.max())
    print("Predicted prob:", prob, "Pred class:", pred)
    
    assert cam.shape == (96, 96), f"Expected (96, 96), got {cam.shape}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0, "CAM must be normalized in [0, 1]"
    
    # Test colormap overlay
    dummy_raw_img = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    overlay = apply_colormap_on_image(dummy_raw_img, cam, alpha=0.5)
    
    print("Overlay shape:", overlay.shape, "Dtype:", overlay.dtype)
    assert overlay.shape == (96, 96, 3)
    assert overlay.dtype == np.uint8
    
    print("\n[SUCCESS] Grad-CAM module verified successfully!")

if __name__ == "__main__":
    test_gradcam()
