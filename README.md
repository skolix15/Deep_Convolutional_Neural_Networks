# Deep Convolutional Neural Networks — Feature Extraction vs Fine-tuning

> Transfer learning with ResNet18 on CIFAR-10 & STL10  
> MSc Artificial Intelligence — Deep Learning & Multimedia Data Analysis

---

## Overview

This project investigates two transfer learning strategies for adapting a pretrained **ResNet18** model to new image classification tasks:

- **Feature Extraction (FE):** All pretrained layers are frozen. Only a new fully connected head (512 → 10) is trained — just **5,130 trainable parameters**.
- **Fine-tuning (FT):** The last two residual blocks (layer3, layer4) are unfrozen alongside the new FC head, allowing the model to adapt its higher-level representations — approximately **10.5M trainable parameters**.

Both strategies are evaluated on **CIFAR-10** and **STL10**, two datasets sharing the same 10 semantic categories but differing significantly in image resolution and training set size.

---

## Results

| | FE — CIFAR-10 | FT — CIFAR-10 | FE — STL10 | FT — STL10 |
|---|---|---|---|---|
| **Test Accuracy** | 79.19% | 94.95% | 93.16% | 95.54% |
| **Accuracy Gain (FT vs FE)** | — | +15.76% | — | +2.37% |
| **Trainable Params** | 5,130 | 10,498,570 | 5,130 | 10,498,570 |
| **Training Time** | 25.9 min | 27.5 min | 3.2 min | 3.4 min |
| **Training Images** | 45,000 | 45,000 | 4,500 | 4,500 |
| **Image Resolution** | 32×32 | 32×32 | 96×96 | 96×96 |

---

## Key Findings

- Fine-tuning outperforms Feature Extraction on both datasets, but the gain is much larger on CIFAR-10 (+15.76%) than STL10 (+2.37%), due to CIFAR-10's significant resolution mismatch with ImageNet.
- Feature Extraction on STL10 (93.16%, 4,500 images) outperforms Feature Extraction on CIFAR-10 (79.19%, 45,000 images), demonstrating that **domain similarity matters more than dataset size**.
- Fine-tuning additional layers adds only ~6% to training time, making it a low-cost improvement when domain shift is present.
- Grad-CAM visualizations show that fine-tuning produces more object-focused attention maps, especially on CIFAR-10.

---

## Model Architecture

ResNet18 pretrained on ImageNet (1.2M images, 1000 classes). The original classification head is replaced with:

```
Dropout(p=0.3) → Linear(512 → 10)
```

In fine-tuning, `layer3` and `layer4` are additionally unfrozen.

---

## Datasets

**CIFAR-10**
- 60,000 color images at 32×32 pixels across 10 classes
- Split: 45,000 train / 5,000 val / 10,000 test

**STL10**
- 13,000 labeled images at 96×96 pixels across 10 classes
- Split: 4,500 train / 500 val / 8,000 test

All images are resized to 224×224 and normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Training data is augmented with random horizontal flips and random crops.

---

## Training Setup

| Setting | Feature Extraction | Fine-tuning |
|---|---|---|
| Epochs | 15 | 15 |
| Optimizer | Adam | Adam |
| Learning Rate | 1e-3 | 1e-4 |
| LR Scheduler | StepLR (×0.5 every 5 epochs) | StepLR (×0.5 every 5 epochs) |
| Loss | Cross-Entropy | Cross-Entropy |
| Checkpointing | Best val accuracy | Best val accuracy |

The lower learning rate for fine-tuning is intentional — it avoids disrupting the pretrained weights (catastrophic forgetting).

---

## Repository Structure

```
├── dcnn_cifar10_resnet18.ipynb   # CIFAR-10 experiment
├── dcnn_stl10_resnet18.ipynb     # STL10 experiment
├── DCNN_Report.pdf               # Full report
└── README.md
```

---

## Requirements

```bash
torch
torchvision
numpy
matplotlib
scikit-learn
```

---

## Visualizations

The notebooks include:
- Training/validation accuracy and loss curves
- Normalized confusion matrices
- Grad-CAM attention maps for both Feature Extraction and Fine-tuning models
