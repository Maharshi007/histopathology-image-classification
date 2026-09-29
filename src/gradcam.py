import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    
    Targeted at ResNet50's final convolutional layer (resnet.layer4[-1]).
    Computes class activation maps for binary classification outputs.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            # Default to the final bottleneck block in ResNet50 layer4
            self.target_layer = self.model.resnet.layer4[-1]
        else:
            self.target_layer = target_layer
            
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, float, int]:
        """
        Generate Grad-CAM heatmap for a single input image tensor (1, C, H, W).
        
        Args:
            input_tensor: Preprocessed tensor (1, 3, 96, 96) on the correct device.
            target_class: Target class index. For binary classification with single logit,
                         None defaults to the model's predicted class.
                         
        Returns:
            cam: 2D numpy array (96, 96) normalized in range [0, 1].
            prob: Model predicted probability for tumor (class 1).
            pred_class: Binary predicted class (0 or 1).
        """
        self.model.zero_grad()
        
        # Forward pass
        logit = self.model(input_tensor)  # shape (1, 1)
        prob = torch.sigmoid(logit).item()
        pred_class = 1 if prob >= 0.5 else 0
        
        # Score for backprop: if target_class is 1, backprop logit; if 0, backprop -logit
        if target_class is None:
            score = logit if pred_class == 1 else -logit
        else:
            score = logit if target_class == 1 else -logit
            
        score.backward(retain_graph=True)
        
        # Global Average Pooling of gradients: shape (1, C, 1, 1)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on features with positive influence on class score
        cam = F.relu(cam)
        
        # Upsample to original image resolution (96x96)
        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False
        )
        
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Min-max normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, prob, pred_class


def apply_colormap_on_image(
    org_img: np.ndarray,
    activation_map: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a heatmap onto an RGB image using simple RGB interpolation.
    
    Args:
        org_img: uint8 array (H, W, 3) in range [0, 255]
        activation_map: float array (H, W) in range [0.0, 1.0]
        alpha: Heatmap blend intensity (0.0 to 1.0)
        
    Returns:
        overlay: uint8 RGB array (H, W, 3) in range [0, 255]
    """
    # Simple Jet-like colormap mapping without external heavy dependencies
    # Map [0, 1] to pseudo-colored RGB: Blue -> Cyan -> Yellow -> Red
    val = np.clip(activation_map, 0, 1)
    
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0, 1)
    
    heatmap_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.float32)
    org_float = org_img.astype(np.float32)
    
    overlay = (alpha * heatmap_rgb + (1.0 - alpha) * org_float)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay
