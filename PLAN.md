# Image Classification Plan — CUB-200-2011 Bird Species

## Overview

- **Dataset**: CUB-200-2011 — 200 bird species, ~11,788 images
- **Split**: Train / Val / Test (stratified, seed=42, 80/20 from original train + original test)
- **Goal**: Build and evaluate image classification models using **transfer learning** and compare with **traditional ML on extracted features**

---

## 0. Prerequisites

```
pip install torch torchvision scikit-learn matplotlib pandas tqdm pillow
```

- GPU recommended (CUDA) but CPU works for small experiments
- EDA already completed in `src/classification.ipynb` and `src/core_eda.ipynb`

---

## 1. Data Loading & Preprocessing

### 1.1 Custom Dataset

- Parse `data/CUB_200_2011/images.txt` → image paths
- Parse `data/CUB_200_2011/image_class_labels.txt` → class labels (0-indexed)
- Parse `data/CUB_200_2011/train_val_test_split.txt` → split assignment
- Build a PyTorch `Dataset` class that reads images by split (train/val/test)

### 1.2 Transforms

| Split        | Transforms                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| **Train**    | `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ColorJitter(0.2, 0.2, 0.2)`, `Normalize(ImageNet)` |
| **Val/Test** | `Resize(256)`, `CenterCrop(224)`, `Normalize(ImageNet)`                                                 |

### 1.3 DataLoaders

- `batch_size = 32` (adjust based on GPU memory)
- `num_workers = 4`
- Shuffle train only

---

## 2. Approach A — Transfer Learning (Fine-Tuning Pretrained CNN)

This is the **required** approach.

### 2.1 Model Selection

Use **ResNet-50** pretrained on ImageNet as the backbone. Other options: EfficientNet-B0, MobileNetV2.

### 2.2 Strategy: Two-Stage Fine-Tuning

**Stage 1 — Train classifier head only (feature extraction)**

1. Load `resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`
2. Freeze all layers (`param.requires_grad = False`)
3. Replace `model.fc` with `nn.Linear(2048, 200)`
4. Train for **5–10 epochs** with `lr=1e-3`, Adam optimizer
5. Purpose: warm up the new classification head

**Stage 2 — Unfreeze and fine-tune entire network**

1. Unfreeze all (or last few) layers
2. Lower learning rate to `lr=1e-4` or use differential LR:
   - Backbone: `1e-5`
   - Classifier head: `1e-3`
3. Train for **15–25 epochs**
4. Use `ReduceLROnPlateau` or `CosineAnnealingLR` scheduler
5. Apply early stopping on validation loss (patience=5)

### 2.3 Training Loop

```
For each epoch:
    Train phase:
        - Forward pass → CrossEntropyLoss
        - Backward pass → optimizer.step()
        - Track loss and accuracy
    Validation phase:
        - Forward pass only (no grad)
        - Track loss and accuracy
        - Save best model checkpoint
    Log: epoch, train_loss, train_acc, val_loss, val_acc
```

### 2.4 Hyperparameters Summary

| Parameter      | Value                      |
| -------------- | -------------------------- |
| Optimizer      | Adam / AdamW               |
| Loss           | CrossEntropyLoss           |
| LR (head)      | 1e-3                       |
| LR (backbone)  | 1e-5                       |
| Batch size     | 32                         |
| Epochs         | 5 (stage 1) + 20 (stage 2) |
| Scheduler      | CosineAnnealingLR          |
| Early stopping | patience=5 on val_loss     |

---

## 3. Approach B — Traditional ML on Extracted Features

This is the **comparison** approach.

### 3.1 Feature Extraction

1. Load same pretrained ResNet-50 (no fine-tuning)
2. Remove the final FC layer → use output of `avgpool` as features
3. Pass all images through the model → get **2048-dim feature vectors**
4. Save features to disk (numpy arrays) to avoid re-computation

### 3.2 Classical Classifiers

Train and evaluate multiple classifiers on extracted features:

| Classifier              | Key Hyperparameters                                   |
| ----------------------- | ----------------------------------------------------- |
| **Logistic Regression** | `C=1.0`, `max_iter=1000`, `multi_class='multinomial'` |
| **SVM (Linear)**        | `C=1.0`, `kernel='linear'`                            |
| **SVM (RBF)**           | `C=10`, `gamma='scale'`                               |
| **Random Forest**       | `n_estimators=500`, `max_depth=None`                  |
| **KNN**                 | `n_neighbors=5`, `metric='cosine'`                    |

### 3.3 Optional: Dimensionality Reduction

- Apply PCA (e.g., 256 or 512 components) before classical ML
- Compare performance with/without PCA

---

## 4. Evaluation

### 4.1 Metrics (for both approaches)

- **Top-1 Accuracy** (primary metric)
- **Top-5 Accuracy**
- **Per-class Precision, Recall, F1** (classification report)
- **Confusion Matrix** (heatmap, at least top-20 most confused pairs)

### 4.2 Comparison Table

| Model                                 | Top-1 Acc | Top-5 Acc | Training Time | Notes             |
| ------------------------------------- | --------- | --------- | ------------- | ----------------- |
| ResNet-50 Fine-Tuned                  | —         | —         | —             | Transfer learning |
| Logistic Regression (ResNet features) | —         | —         | —             | Traditional ML    |
| SVM Linear (ResNet features)          | —         | —         | —             | Traditional ML    |
| SVM RBF (ResNet features)             | —         | —         | —             | Traditional ML    |
| Random Forest (ResNet features)       | —         | —         | —             | Traditional ML    |
| KNN (ResNet features)                 | —         | —         | —             | Traditional ML    |

### 4.3 Visualizations

- Training/validation loss and accuracy curves (fine-tuning)
- Confusion matrix heatmap
- Per-class accuracy bar chart (top/bottom 10 classes)
- t-SNE of learned features (fine-tuned model) colored by class

---

## 5. Implementation Order (Step by Step)

```
Step 1: Data loading
   └── Build CUBDataset class, verify data loading, check shape/label correctness

Step 2: Feature extraction (Approach B first — faster to iterate)
   └── Extract ResNet-50 features for all images
   └── Save to .npy files

Step 3: Train classical classifiers
   └── Logistic Regression, SVM, RF, KNN
   └── Evaluate on test set, record metrics

Step 4: Fine-tuning (Approach A)
   └── Stage 1: freeze backbone, train head
   └── Stage 2: unfreeze, fine-tune with low LR
   └── Save best checkpoint

Step 5: Evaluate fine-tuned model
   └── Test accuracy, confusion matrix, classification report

Step 6: Compare all models
   └── Build comparison table
   └── Generate plots

Step 7: Report / Notebook write-up
   └── Combine EDA + modeling + evaluation
   └── Add analysis of results
```

---

## 6. Expected Results (Rough Estimates)

| Model                                  | Expected Top-1 Accuracy |
| -------------------------------------- | ----------------------- |
| Random baseline                        | ~0.5%                   |
| KNN on ResNet features                 | ~40–55%                 |
| Logistic Regression on ResNet features | ~55–65%                 |
| SVM on ResNet features                 | ~55–65%                 |
| ResNet-50 fine-tuned (head only)       | ~50–65%                 |
| ResNet-50 fine-tuned (full)            | ~70–82%                 |

> CUB-200 is a **fine-grained** classification task — 200 visually similar bird species. Even state-of-the-art models find this challenging. Results in the 60–80% range are reasonable for a course assignment.

---

## 7. File Structure (Proposed)

```
src/
  classification.ipynb     ← existing EDA + add training/eval cells
  (or new training.ipynb)
models/                    ← saved model checkpoints
  resnet50_finetuned.pth
precompute/
  features/
    train_features.npy
    train_labels.npy
    val_features.npy
    val_labels.npy
    test_features.npy
    test_labels.npy
csv/
  model_comparison.csv     ← final comparison results
```

---

## 8. Key Tips

1. **Start with feature extraction + Logistic Regression** — fastest way to get a working baseline
2. **Use `torch.no_grad()` during feature extraction** — saves memory
3. **Monitor validation accuracy** to avoid overfitting (200 classes with limited data per class)
4. **Use `torchvision.models` weights enum** (new API) instead of deprecated `pretrained=True`
5. **Normalize images with ImageNet stats** — critical for pretrained models
6. If GPU memory is tight, reduce batch size or use `fp16` (mixed precision)
7. Log everything — you'll need it for the report
