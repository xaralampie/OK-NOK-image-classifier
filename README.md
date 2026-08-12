# OK-NOK Image Classifier

An end-to-end, deep learning-based automated visual inspection pipeline designed for quality control in manufacturing line assembly. This project leverages transfer learning with **ResNet-18 (Convolutional Neural Network)**, **Stratified K-Fold Cross-Validation**, and **OpenCV** to automatically classify industrial components (e.g., connectors) as **OK** (pass) or **NOK** (fail / defect / wrong orientation) with confidence scoring and real-time visual overlays.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works & Pipeline Architecture](#how-it-works--pipeline-architecture)
- [Deep Learning & CNN Details](#deep-learning--cnn-details)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training & Visual Inspection Inference](#2-model-training--visual-inspection-inference)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Detailed Code Walkthrough](#detailed-code-walkthrough)
- [License](#license)

---

## Overview

In modern manufacturing, quality assurance often relies on visual inspection to ensure parts are installed correctly and defect-free. Manual inspection is slow and error-prone. This repository provides a complete industrial-grade machine vision pipeline that:

1. **Parses and indexes** visual inspection datasets automatically from filenames (`OK_*`, `NOK_*`).
2. **Trains a specialized ResNet-18 Convolutional Neural Network (CNN)** using transfer learning and partial backbone fine-tuning.
3. **Validates robustly** using Stratified K-Fold Cross Validation and early stopping to avoid overfitting on small datasets.
4. **Executes real-time inference** on raw images/frames and outputs annotated inspection overlays (`inspection_result.png`) featuring status bounding banners, text labels, and confidence percentage.

---

## Key Features

- **Convolutional Neural Network (CNN)**: Uses a pre-trained ResNet-18 model fine-tuned specifically for industrial defect detection.
- **Robust Transfer Learning**: Freezes early feature extractor layers while fine-tuning high-level semantic layers (`layer4`) and custom multi-layer perceptron (MLP) classification heads with Dropout.
- **Data Augmentation**: Incorporates spatial and color transforms (Random Rotation, Resized Crop, Color Jitter) to simulate factory light variation and alignment shifts.
- **Stratified K-Fold CV**: Ensures balanced class distribution across training and validation splits, dynamically adapting fold counts to small dataset sizes.
- **Adaptive Early Stopping**: Monitors validation loss and saves the best-performing model checkpoint per fold while preventing overtraining.
- **Automated OpenCV Visual Overlay**: Draws green (`OK`) or red (`NOK`) visual indicators and confidence scores directly onto output inspection images.

---

## How It Works & Pipeline Architecture

The pipeline consists of four distinct stages executed end-to-end within `train_custom.py`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA & AUGMENTATION                    │
│  - Dataset parsing (OK / NOK prefixes in data/)                         │
│  - Augmentation pipeline (Rotation, Crop, Color Jitter, Normalization)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: RESNET-18 MODEL BUILD                     │
│  - Load ImageNet pre-trained ResNet-18 backbone                         │
│  - Freeze layers 1–3, unfreeze Layer 4                                 │
│  - Custom classifier head (Dropout -> FC 128 -> ReLU -> Dropout -> FC 2)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               STAGE 3: STRATIFIED K-FOLD TRAINING & EVAL               │
│  - Stratified K-Fold split (e.g., K=2 to 5 depending on sample size)   │
│  - Adam Optimizer + ReduceLROnPlateau scheduler                         │
│  - Early stopping & checkpoint saving (`best_model_foldX.pt`)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STAGE 4: VISUAL INSPECTION INFERENCE               │
│  - Load best model checkpoint                                          │
│  - Inference with Softmax probability calculation                      │
│  - OpenCV visual annotation (Green frame for OK, Red for NOK)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Deep Learning & CNN Details

### What Type of CNN is Used?
This project uses **ResNet-18 (Residual Network with 18 layers)**, originally introduced by He et al. (Microsoft Research).

#### Why ResNet-18?
1. **Residual Connections (Skip Connections)**: ResNet uses identity shortcut connections that allow gradients to flow directly through the network, solving the vanishing gradient problem and enabling fast, stable convergence.
2. **Lightweight & High Speed**: With only ~11.7 million parameters, ResNet-18 is fast enough for real-time edge deployment on manufacturing line vision systems or embedded PCs.
3. **Effective Feature Representation**: Pre-trained on ImageNet, its early layers capture fundamental visual features (edges, textures, gradients) which translate exceptionally well to industrial surface and component inspection.

### Fine-Tuning Strategy (Transfer Learning)
- **Frozen Backbone (`conv1` through `layer3`)**: Retains universal visual feature detectors without updating weights during gradient updates (`requires_grad = False`).
- **Trainable High-Level Features (`layer4`)**: Unfrozen (`requires_grad = True`) to adapt complex spatial patterns specific to the target component geometry.
- **Custom Classification Head**:
  - `Dropout(p=0.2)`
  - `Linear(in_features=512, out_features=128)`
  - `ReLU()` activation
  - `Dropout(p=0.1)`
  - `Linear(in_features=128, out_features=num_classes)`

---

## Repository Structure

```
OK-NOK-image-classifier/
├── data/                       # Folder containing industrial training/testing images
│   ├── OK_01.jpg
│   ├── OK_02.png
│   ├── NOK_01.jpg
│   └── NOK_02.png
├── train_custom.py             # Complete PyTorch training & OpenCV inspection script
├── requirements.txt            # Project dependencies list
└── README.md                   # Repository documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch (CUDA recommended for GPU acceleration)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/OK-NOK-image-classifier.git
   cd OK-NOK-image-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Linux/macOS
   # or
   .\venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Data Preparation
Place your target inspection images inside the `data/` directory. The pipeline automatically extracts ground truth labels based on filename prefixes:
- Filenames starting with **`OK`** (e.g., `OK_part1.png`, `ok_sample.jpg`) are assigned to the **OK** class.
- Filenames starting with **`NOK`** (e.g., `NOK_defect1.png`, `nok_wrong_side.jpg`) are assigned to the **NOK** class.

### 2. Model Training & Visual Inspection Inference
To run the full end-to-end pipeline (dataset loading, cross-validation training, checkpoint saving, and sample image visual inspection), run:

```bash
python train_custom.py
```

During execution, the script will:
1. Load and parse images inside `data/`.
2. Train models across Stratified K-Fold splits and save the top checkpoint (`best_model_fold0.pt`).
3. Reload the saved model checkpoint and execute visual inspection on a sample image.
4. Export the annotated image (`inspection_result.png`):
   - **OK Result**: Solid Green border with label `OK: Correct Side (Confidence %)`.
   - **NOK Result**: Solid Red border with label `NOK: WRONG SIDE (Confidence %)`.

---

## Configuration & Hyperparameters

Key hyperparameter settings used in `train_custom.py`:

| Parameter | Value | Description |
|---|---|---|
| **Backbone** | ResNet-18 | Pre-trained weights (`ResNet18_Weights.DEFAULT`) |
| **Image Resolution** | 224 x 224 | Input dimensions for standard ResNet backbone |
| **Batch Size** | Dynamic (`min(8, len)`) | Optimized for small industrial datasets |
| **Optimizer** | Adam (`lr=1e-3`) | Updates only trainable parameters (`layer4` + `fc`) |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor reduction on validation loss plateau (patience=2) |
| **Loss Function** | CrossEntropyLoss | Multi-class standard cross-entropy |
| **Max Epochs** | 15 | Maximum training duration per fold |
| **Early Stopping** | 5 epochs | Triggers when validation loss stops improving |

---

## Detailed Code Walkthrough

### Stage 1: Custom Dataset & Transforms
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
Applies geometric transformations and light jittering to teach the network invariance to real-world camera alignment variations and lighting changes.

### Stage 2: ResNet-18 Backbone Customization
```python
def build_model(num_classes=2):
    backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.layer4.parameters():
        param.requires_grad = True
    # Custom classifier head replacing default fc layer...
```
Selectively unfreezes deep residual blocks (`layer4`) while freezing early low-level feature layers (`layer1` to `layer3`).

### Stage 3: K-Fold Evaluator & Training Loop
```python
val_loss, val_acc = evaluate(model, val_loader, criterion)
scheduler.step(val_loss)
```
Evaluates accuracy/loss after each epoch, updates learning rate adaptively, and saves model state dictionaries to disk when new validation loss minima are hit.

### Stage 4: OpenCV Annotation
```python
cv2.rectangle(annotated_img, (0, 0), (w, h), color, thickness=10)
cv2.putText(annotated_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
```
Applies visual indicators directly to BGR numpy arrays and saves the rendered visual feedback file (`inspection_result.png`).


### HELPING TOOLS FOLDER CONSISTS OF PYTHON SCRIPTS THAT MAY BE OF USE

---

## License

This project is open-source and available under the [MIT License](LICENSE).
