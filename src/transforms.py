from torchvision import transforms
from src.config import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """
    Data augmentation and preprocessing for training.
    
    Histopathology patches are rotationally and reflectionally invariant under 
    microscopy (no canonical 'up' or 'down'). Horizontal and vertical flips
    preserve structural and cellular morphology while teaching the model invariance.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms():
    """
    Deterministic preprocessing for validation and test sets.
    Converts uint8 [0, 255] HWC array to float32 [0.0, 1.0] CHW Tensor
    and applies ImageNet standardization.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
