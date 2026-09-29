# Deep Learning-Based Histopathology Image Classification on PatchCamelyon (PCam)

A modular, reproducible, and production-quality medical computer vision system using **ResNet50 Transfer Learning** to classify histopathology image patches into tumor metastasis vs. normal lymph node tissue.

---

## 1. Problem Formulation & Dataset Task Definition

- **Dataset**: [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) benchmark derived from the CAMELYON16 Challenge.
- **Input Patch Dimensions**: $96 \times 96 \times 3$ RGB pixels ($0.243\,\mu\text{m}$ resolution, $10\times$ equivalent magnification).
- **Target Label ($y \in \{0, 1\}$)**:
  - `0` = Non-tumor (normal lymph node stroma, lymphocytes, adipose tissue)
  - `1` = Tumor (metastatic breast carcinoma)
- **Task Specificity**: A patch is labeled as **positive (tumor = 1)** if and only if the central $32 \times 32$ pixel region contains at least one pixel of tumor tissue. The outer $32\,\text{px}$ margin provides spatial and cellular context without requiring padding.
- **Dataset Scale**:
  - **Train**: 262,144 patches (strictly balanced: 131,072 class 0, 131,072 class 1)
  - **Validation**: 32,768 patches (16,384 class 0, 16,384 class 1)
  - **Test**: 32,768 patches (16,384 class 0, 16,384 class 1)

> **Important Domain Boundary**: This project operates on extracted **image patches**, not whole-slide gigapixel images (WSIs). It serves as the patch-level feature extraction and classification stage in modern computational pathology workflows.

---

## 2. Model Architecture & Transfer Learning Strategy

### Primary Model: ResNet50
- **Feature Extractor**: Deep residual network with 50 layers pretrained on ImageNet-1K.
- **Global Average Pooling**: Compresses spatial feature maps into a 2048-dimensional dense representation vector.
- **Classification Head**: `Dropout(p=0.3) -> Linear(2048, 1)` producing raw scalar logits for numerical stability.

### Two-Stage Transfer Learning
1. **Stage 1 (Head Warmup)**:
   - All convolutional backbone weights are frozen (`requires_grad = False`).
   - Only the binary classification head (2,049 parameters) is trained using `AdamW` ($\text{LR} = 10^{-3}$).
   - **Why**: Prevents large random initial gradients from destroying pretrained low- and mid-level visual features.
2. **Stage 2 (End-to-End / Selective Fine-Tuning)**:
   - Backbone layers are unfrozen and trained end-to-end with a conservative learning rate ($\text{LR} = 10^{-4}$).
   - `ReduceLROnPlateau` scheduler dynamically lowers learning rate when validation $F_1$-score plateaus.

---

## 3. Histopathology-Specific Preprocessing & Augmentation

Histopathology tissue sections under brightfield microscopy possess **$D_4$ dihedral rotational and reflectional symmetry** (there is no canonical "up" or "down" orientation on a slide).

- **Training Pipeline**:
  - Convert `HWC uint8 [0, 255]` $\rightarrow$ `CHW float32 [0.0, 1.0]` (`transforms.ToTensor()`)
  - Random Horizontal Flip ($p = 0.5$)
  - Random Vertical Flip ($p = 0.5$)
  - ImageNet Channel Standardization: $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$
- **Evaluation Pipeline (Validation / Test)**:
  - Deterministic conversion and ImageNet standardization (no stochastic augmentations).

---

## 4. Hardware Optimization (GTX 1650 4GB VRAM)

- **Memory Efficiency**: Peak VRAM allocation during training is $< 250\,\text{MB}$ ($< 6\%$ of total 4 GB capacity).
- **HDF5 Data Streaming**: Reads patches directly from disk via HDF5 slicing, eliminating RAM bottlenecks ($< 500\,\text{MB}$ RAM overhead).
- **Precision**: Deterministic, full-precision (FP32) training guarantees stability without numerical underflow.

---

## 5. Project Directory Structure

```text
histopathology-image-classification/
├── data/
│   └── raw/                   # 6 PCam HDF5 dataset files (.h5)
├── src/
│   ├── __init__.py            # Package interface
│   ├── config.py              # Centralized hyperparameters & paths
│   ├── dataset.py             # Memory-efficient HDF5 PCamDataset
│   ├── transforms.py          # Domain-specific augmentation & normalization
│   ├── model.py               # ResNet50 with stage-wise freezing controls
│   ├── metrics.py             # Medical classification evaluation metrics
│   ├── train.py               # Two-stage training engine & checkpointing
│   ├── evaluate.py            # Evaluation & validation logic
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping
│   └── utils.py               # Reproducibility seed & checkpoint helpers
├── scripts/
│   ├── train.py               # Training entrypoint script
│   ├── evaluate.py            # Standalone evaluation entrypoint script
│   ├── generate_gradcam.py    # Grad-CAM visualization generator
│   └── download_weights.py    # Pretrained weights helper
├── results/
│   ├── checkpoints/           # Best and final model .pth weights
│   ├── metrics/               # Evaluation JSONs & training history
│   └── figures/               # Grad-CAM overlays (TP, TN, FP, FN)
└── requirements.txt
```

---

## 6. Usage & Execution

### 1. Verification of Environment & Subsystems
```powershell
.venv\Scripts\python.exe scripts\test_pipeline_transforms.py
.venv\Scripts\python.exe scripts\test_model.py
.venv\Scripts\python.exe scripts\test_dataloader.py
.venv\Scripts\python.exe scripts\test_gradcam.py
```

### 2. Run Full Training Pipeline
```powershell
.venv\Scripts\python.exe scripts\train.py --batch-size 64 --stage1-epochs 3 --stage2-epochs 5
```

### 3. Evaluate on Test Set
```powershell
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint results/checkpoints/best_model.pth --split test
```

### 4. Generate Grad-CAM Explainability Visualizations
```powershell
.venv\Scripts\python.exe scripts\generate_gradcam.py --checkpoint results/checkpoints/best_model.pth --split test
```

---

## 7. Medical Metrics & Error Analysis

The system tracks domain-essential evaluation metrics:
- **Recall / Sensitivity**: $\frac{TP}{TP + FN}$ (Critical for minimizing missed tumor metastases)
- **Precision / PPV**: $\frac{TP}{TP + FP}$
- **Specificity**: $\frac{TN}{TN + FP}$
- **$F_1$-Score**: Harmonic mean of Precision and Sensitivity
- **Confusion Matrix**: Full tally of $TP, TN, FP, FN$

### Explainability with Grad-CAM
Grad-CAM computes gradients of the tumor class logit with respect to the feature activations of the last convolutional block (`resnet.layer4[-1]`). The resulting spatial heatmap highlights cellular regions (e.g. enlarged pleomorphic nuclei, hyperchromatic clusters) driving model decisions.

---

## 8. Disclaimer
This is an academic research and benchmark classification project on the PatchCamelyon dataset. It is not approved or intended for clinical diagnostics or direct patient care.
