import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw

from src.config import (
    TEST_X_PATH,
    TEST_Y_PATH,
    VALID_X_PATH,
    VALID_Y_PATH,
    DEVICE,
    CHECKPOINTS_DIR,
    FIGURES_DIR
)
from src.dataset import PCamDataset
from src.transforms import get_eval_transforms
from src.model import build_model
from src.utils import load_checkpoint
from src.gradcam import GradCAM, apply_colormap_on_image


def draw_center_box(img_pil: Image.Image) -> Image.Image:
    """
    Draw bounding box around the central 32x32 diagnostic region on a 96x96 patch.
    """
    draw = ImageDraw.Draw(img_pil)
    # Coordinates for central 32x32 region in 96x96: (32, 32) to (64, 64)
    draw.rectangle([32, 32, 64, 64], outline=(0, 255, 0), width=1)
    return img_pil


def save_gradcam_panel(
    raw_img_hwc: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    true_label: int,
    pred_label: int,
    pred_prob: float,
    case_name: str,
    output_path: Path
) -> None:
    """
    Create and save a 3-panel comparison image:
    [Original Patch with 32x32 ROI] | [Grad-CAM Heatmap] | [Overlay]
    """
    # Panel 1: Original with ROI Box
    img_pil = Image.fromarray(raw_img_hwc)
    img_with_roi = draw_center_box(img_pil.copy())
    
    # Panel 2: Heatmap
    heatmap_rgb = (np.stack([heatmap, heatmap, heatmap], axis=-1) * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)
    
    # Panel 3: Overlay
    overlay_pil = Image.fromarray(overlay)
    
    # Combine horizontally: 96*3 = 288 width, 96 height + banner
    panel_w = 96 * 3 + 16
    panel_h = 96 + 32
    
    combined = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    combined.paste(img_with_roi, (4, 28))
    combined.paste(heatmap_pil, (96 + 8, 28))
    combined.paste(overlay_pil, (96 * 2 + 12, 28))
    
    # Simple header drawing
    draw = ImageDraw.Draw(combined)
    label_str = "Tumor (1)" if true_label == 1 else "Non-Tumor (0)"
    pred_str = f"{'Tumor (1)' if pred_label == 1 else 'Non-Tumor (0)'} ({pred_prob*100:.1f}%)"
    header = f"[{case_name.upper()}] True: {label_str} | Pred: {pred_str}"
    draw.text((6, 6), header, fill=(0, 0, 0))
    
    combined.save(output_path)
    print(f"Saved {case_name} visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations for PCam")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_model.pth"))
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--max-per-case", type=int, default=2)
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    x_path = TEST_X_PATH if args.split == "test" else VALID_X_PATH
    y_path = TEST_Y_PATH if args.split == "test" else VALID_Y_PATH
    
    # Raw dataset without normalization for visualization
    raw_dataset = PCamDataset(image_path=x_path, label_path=y_path, transform=None)
    eval_transform = get_eval_transforms()
    
    # Load Model & Grad-CAM
    model = build_model(pretrained=False).to(DEVICE)
    load_checkpoint(checkpoint_path, model, device=DEVICE)
    gradcam = GradCAM(model=model)
    
    # Find TP, TN, FP, FN samples
    cases = {"tp": [], "tn": [], "fp": [], "fn": []}
    target_count = args.max_per_case
    
    print("Searching dataset for representative TP, TN, FP, FN cases...")
    for idx in range(len(raw_dataset)):
        if all(len(v) >= target_count for v in cases.values()):
            break
            
        raw_img, true_label = raw_dataset[idx]
        tensor_img = eval_transform(raw_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = model(tensor_img)
            prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob >= 0.5 else 0
            
        if true_label == 1 and pred_label == 1 and len(cases["tp"]) < target_count:
            cases["tp"].append((idx, raw_img, true_label, pred_label, prob))
        elif true_label == 0 and pred_label == 0 and len(cases["tn"]) < target_count:
            cases["tn"].append((idx, raw_img, true_label, pred_label, prob))
        elif true_label == 0 and pred_label == 1 and len(cases["fp"]) < target_count:
            cases["fp"].append((idx, raw_img, true_label, pred_label, prob))
        elif true_label == 1 and pred_label == 0 and len(cases["fn"]) < target_count:
            cases["fn"].append((idx, raw_img, true_label, pred_label, prob))
            
    print("\nGenerating Grad-CAM overlays...")
    for case_type, samples in cases.items():
        for i, (idx, raw_img, true_lbl, pred_lbl, prob) in enumerate(samples):
            tensor_img = eval_transform(raw_img).unsqueeze(0).to(DEVICE)
            heatmap, _, _ = gradcam.generate_cam(tensor_img, target_class=1)
            overlay = apply_colormap_on_image(raw_img, heatmap, alpha=0.45)
            
            output_name = f"gradcam_{case_type}_{i+1}_idx{idx}.png"
            output_path = FIGURES_DIR / output_name
            save_gradcam_panel(
                raw_img_hwc=raw_img,
                heatmap=heatmap,
                overlay=overlay,
                true_label=true_lbl,
                pred_label=pred_lbl,
                pred_prob=prob,
                case_name=f"{case_type}_{i+1}",
                output_path=output_path
            )


if __name__ == "__main__":
    main()
