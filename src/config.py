import torch
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# Ensure output directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths
TRAIN_X_PATH = DATA_DIR / "camelyonpatch_level_2_split_train_x.h5"
TRAIN_Y_PATH = DATA_DIR / "camelyonpatch_level_2_split_train_y.h5"
VALID_X_PATH = DATA_DIR / "camelyonpatch_level_2_split_valid_x.h5"
VALID_Y_PATH = DATA_DIR / "camelyonpatch_level_2_split_valid_y.h5"
TEST_X_PATH = DATA_DIR / "camelyonpatch_level_2_split_test_x.h5"
TEST_Y_PATH = DATA_DIR / "camelyonpatch_level_2_split_test_y.h5"

# Hardware and System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Conservative setting for Windows 11 + h5py compatibility
SEED = 42

# Image Dimensions
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
NUM_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification (BCEWithLogitsLoss: 0=non-tumor, 1=tumor)

# Normalization Constants (ImageNet pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Hyperparameters (GTX 1650 4GB VRAM profile)
BATCH_SIZE = 64
STAGE1_EPOCHS = 3
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 5
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
