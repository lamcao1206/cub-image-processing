# Transfer Learning for Bird Species Classification

## Overview

We apply **transfer learning** using a **ResNet-50** convolutional neural network, pretrained on ImageNet (1.2M images, 1000 classes), to classify 200 bird species from the **CUB-200-2011** dataset. Instead of training a deep network from scratch — which would require far more data and compute — we leverage the rich visual features already learned by ResNet-50 and adapt them to our fine-grained bird classification task.

Two approaches are implemented:

| Approach | Method                            | Description                                                                     |
| -------- | --------------------------------- | ------------------------------------------------------------------------------- |
| **A**    | Fine-Tuning                       | Replace the classifier head and retrain the full network end-to-end             |
| **B**    | Feature Extraction + Classical ML | Freeze the CNN, extract feature vectors, and train classical classifiers on top |

---

## Preprocessing

All images are preprocessed with **bounding box cropping** (15% padding) to focus on the bird region, then resized to 224×224 pixels — the standard input size for ResNet-50.

**Training augmentations** are applied to improve generalization:

- Resize to 256×256, then random crop to 224×224
- Random horizontal flip
- Color jitter (brightness, contrast, saturation ±20%)
- ImageNet normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

**Validation/Test transforms** use center crop instead of random crop, with no augmentation.

---

## Approach A — Fine-Tuning ResNet-50

The pretrained ResNet-50 has its final fully connected layer replaced with a new linear layer mapping the 2048-dimensional feature vector to 200 output classes. Training proceeds in **two stages**:

### Stage 1: Classifier Head Warmup (5 epochs)

- The entire backbone (convolutional layers) is **frozen** — gradients are not computed for these parameters.
- Only the new classifier head is trained using **Adam** optimizer with a learning rate of **1×10⁻³**.
- This prevents the randomly initialized head from sending large, noisy gradients through the pretrained backbone and destroying its learned features.

### Stage 2: Full Network Fine-Tuning (up to 20 epochs)

- All layers are **unfrozen** and trained end-to-end.
- **Differential learning rates** are applied:
  - Backbone layers: **1×10⁻⁵** (small updates to preserve pretrained knowledge)
  - Classifier head: **1×10⁻³** (faster adaptation for the task-specific layer)
- A **cosine annealing** learning rate scheduler smoothly decays the learning rate to zero over the training period.
- **Early stopping** with patience of 5 epochs monitors validation accuracy to prevent overfitting.
- The model checkpoint with the best validation accuracy is saved and used for final evaluation.

### Loss Function

**Cross-Entropy Loss** — the standard loss for multi-class classification, which combines log-softmax and negative log-likelihood.

---

## Approach B — Feature Extraction + Classical ML

The pretrained ResNet-50 is used as a **fixed feature extractor**:

1. The final classification layer (`fc`) is replaced with an identity function, so the network outputs a **2048-dimensional feature vector** for each image.
2. All model weights are frozen — no gradient computation or backpropagation occurs.
3. Feature vectors are extracted for every image in the train, validation, and test splits and cached to disk as `.npy` files.

These feature vectors are then used to train three classical machine learning classifiers:

| Classifier              | Key Hyperparameters                       |
| ----------------------- | ----------------------------------------- |
| **Logistic Regression** | C=1.0, L-BFGS solver, max 2000 iterations |
| **Random Forest**       | 500 trees                                 |
| **K-Nearest Neighbors** | k=5, cosine distance                      |

This approach demonstrates how powerful the representations learned by a pretrained CNN are — even without any fine-tuning, classical classifiers can achieve strong accuracy on the extracted features.

---

## Why Transfer Learning?

Fine-grained bird classification is challenging: many species look nearly identical except for subtle differences in plumage color, beak shape, or wing patterns. The CUB-200-2011 dataset contains only ~11,788 images across 200 classes (~59 images per class on average), which is insufficient to train a deep CNN from scratch.

Transfer learning solves this by:

- **Reusing low-level features** (edges, textures, color gradients) learned from ImageNet's 1.2 million images
- **Reusing mid-level features** (patterns, shapes, parts) that generalize across visual recognition tasks
- **Adapting high-level features** to the specific domain of bird species through fine-tuning
- **Reducing training time** from days/weeks to hours
- **Requiring less data** — the pretrained backbone provides strong regularization

---

## Evaluation

All models are evaluated on a held-out test set using:

- **Top-1 Accuracy** — percentage of images where the predicted class matches the ground truth
- **Top-5 Accuracy** — percentage of images where the ground truth class appears in the model's top 5 predictions (available for models that output probability distributions)

Results from both approaches are compared side-by-side to quantify the benefit of end-to-end fine-tuning over classical classifiers on frozen features.

---

## Architecture Summary

```
Input Image (224×224×3)
        │
        ▼
┌──────────────────────────┐
│   ResNet-50 Backbone     │  ← Pretrained on ImageNet
│   (Convolutional Layers) │
│                          │
│   conv1 → layer1 → ...  │
│   ... → layer4 → avgpool│
└──────────┬───────────────┘
           │
     2048-dim vector
           │
     ┌─────┴─────┐
     │            │
     ▼            ▼
 Approach A   Approach B
     │            │
  FC Layer     Classical ML
 (200 units)  (LogReg/RF/KNN)
     │            │
     ▼            ▼
  200 class    200 class
 predictions  predictions
```
