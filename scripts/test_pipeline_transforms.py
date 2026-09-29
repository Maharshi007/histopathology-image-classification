import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import TRAIN_X_PATH, TRAIN_Y_PATH, VALID_X_PATH, VALID_Y_PATH
from src.dataset import PCamDataset
from src.transforms import get_train_transforms, get_eval_transforms


def test_transforms():
    train_ds = PCamDataset(
        image_path=TRAIN_X_PATH,
        label_path=TRAIN_Y_PATH,
        transform=get_train_transforms()
    )
    
    val_ds = PCamDataset(
        image_path=VALID_X_PATH,
        label_path=VALID_Y_PATH,
        transform=get_eval_transforms()
    )
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    train_img, train_lbl = train_ds[0]
    val_img, val_lbl = val_ds[0]
    
    print("\n--- Train Sample ---")
    print("Shape:", train_img.shape)
    print("Dtype:", train_img.dtype)
    print("Min:", train_img.min().item(), "Max:", train_img.max().item())
    print("Mean per channel:", train_img.mean(dim=[1, 2]).tolist())
    print("Std per channel:", train_img.std(dim=[1, 2]).tolist())
    print("Label:", train_lbl, "Type:", type(train_lbl))
    
    print("\n--- Validation Sample ---")
    print("Shape:", val_img.shape)
    print("Dtype:", val_img.dtype)
    print("Min:", val_img.min().item(), "Max:", val_img.max().item())
    print("Mean per channel:", val_img.mean(dim=[1, 2]).tolist())
    print("Std per channel:", val_img.std(dim=[1, 2]).tolist())
    print("Label:", val_lbl, "Type:", type(val_lbl))
    
    assert isinstance(train_img, torch.Tensor), "train_img must be a torch.Tensor"
    assert train_img.shape == (3, 96, 96), f"Expected shape (3, 96, 96), got {train_img.shape}"
    assert train_img.dtype == torch.float32, f"Expected float32, got {train_img.dtype}"
    assert train_lbl in (0, 1), f"Unexpected label {train_lbl}"
    
    print("\n[SUCCESS] Transforms and Dataset verified successfully!")


if __name__ == "__main__":
    test_transforms()
