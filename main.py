"""
CUB-200-2011 Bird Classification Pipeline
- Approach A: Fine-tune ResNet-50 (Transfer Learning)
- Approach B: Extract ResNet-50 features + Classical ML (LogReg, RF, KNN)

Usage:
    python main.py --approach both   # Run both approaches (default)
    python main.py --approach A      # Fine-tuning only
    python main.py --approach B      # Classical ML only
"""

import os
import csv
import time
import pickle
import tempfile
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    precision_score, recall_score, f1_score,
)
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "CUB_200_2011")
IMAGES_TXT = os.path.join(DATA_DIR, "images.txt")
LABELS_TXT = os.path.join(DATA_DIR, "image_class_labels.txt")
SPLIT_TXT = os.path.join(DATA_DIR, "train_val_test_split.txt")
CLASSES_TXT = os.path.join(DATA_DIR, "classes.txt")
IMAGE_DIR = os.path.join(BASE_DIR, "precompute", "preprocessed_images")
FEATURE_DIR = os.path.join(BASE_DIR, "precompute", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Dataset
# ============================================================
class CUBDataset(Dataset):
    """
    Loads preprocessed 224x224 images from precompute/preprocessed_images/,
    using images.txt, image_class_labels.txt, and train_val_test_split.txt
    to determine which images belong to which split.
    """

    def __init__(self, image_dir, images_txt, labels_txt, split_txt, split="train", transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Parse metadata files
        image_paths = {}
        with open(images_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_paths[int(parts[0])] = parts[1]

        image_labels = {}
        with open(labels_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_labels[int(parts[0])] = int(parts[1]) - 1  # 0-indexed

        image_splits = {}
        with open(split_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_splits[int(parts[0])] = parts[1]

        # Filter by split
        self.samples = []
        for img_id, rel_path in image_paths.items():
            if image_splits.get(img_id) == split:
                full_path = os.path.join(image_dir, rel_path)
                label = image_labels[img_id]
                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_class_names():
    """Load class names from classes.txt"""
    names = {}
    with open(CLASSES_TXT, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            names[int(parts[0]) - 1] = parts[1]
    return names


# ============================================================
# 2. Transforms (images are already 224x224)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dataloaders():
    """Build train/val/test DataLoaders."""
    train_ds = CUBDataset(IMAGE_DIR, IMAGES_TXT, LABELS_TXT, SPLIT_TXT, split="train", transform=train_transform)
    val_ds = CUBDataset(IMAGE_DIR, IMAGES_TXT, LABELS_TXT, SPLIT_TXT, split="val", transform=val_transform)
    test_ds = CUBDataset(IMAGE_DIR, IMAGES_TXT, LABELS_TXT, SPLIT_TXT, split="test", transform=val_transform)

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader


# ============================================================
# 3. Approach B — Feature Extraction + Classical ML
# ============================================================
def extract_features():
    """Extract 2048-dim features from ResNet-50 (frozen) for all splits."""
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # Check if features already extracted
    if all(os.path.exists(os.path.join(FEATURE_DIR, f"{s}_features.npy")) for s in ["train", "val", "test"]):
        print("Features already extracted, loading from disk...")
        return {
            s: (np.load(os.path.join(FEATURE_DIR, f"{s}_features.npy")),
                np.load(os.path.join(FEATURE_DIR, f"{s}_labels.npy")))
            for s in ["train", "val", "test"]
        }

    print("Extracting features with frozen ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # Remove classifier → outputs 2048-dim
    model = model.to(DEVICE)
    model.eval()

    features_dict = {}
    for split_name in ["train", "val", "test"]:
        ds = CUBDataset(IMAGE_DIR, IMAGES_TXT, LABELS_TXT, SPLIT_TXT, split=split_name, transform=val_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Extracting {split_name}"):
                images = images.to(DEVICE)
                feats = model(images).cpu().numpy()
                all_features.append(feats)
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
        np.save(os.path.join(FEATURE_DIR, f"{split_name}_features.npy"), features)
        np.save(os.path.join(FEATURE_DIR, f"{split_name}_labels.npy"), labels)
        features_dict[split_name] = (features, labels)
        print(f"  {split_name}: {features.shape[0]} samples, feature dim={features.shape[1]}")

    return features_dict


def run_classical_ml(features_dict):
    """Train and evaluate classical ML classifiers on extracted features."""
    X_train, y_train = features_dict["train"]
    X_val, y_val = features_dict["val"]
    X_test, y_test = features_dict["test"]

    classifiers = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # Inference timing
        start = time.time()
        y_pred = clf.predict(X_test)
        infer_time = time.time() - start
        infer_per_image = infer_time / len(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Top-5 accuracy (only for classifiers with predict_proba)
        top5 = None
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)
            top5 = top_k_accuracy_score(y_test, y_proba, k=5)

        # Model size (serialized with pickle)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            pickle.dump(clf, tmp)
            tmp_path = tmp.name
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.remove(tmp_path)

        # Parameter count
        if hasattr(clf, "coef_"):
            n_params = clf.coef_.size + (clf.intercept_.size if hasattr(clf, "intercept_") else 0)
        elif hasattr(clf, "estimators_"):
            n_params = sum(est.tree_.node_count for est in clf.estimators_)
        else:
            n_params = None  # KNN has no learnable parameters

        results[name] = {
            "accuracy": acc,
            "top5": top5,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time": train_time,
            "infer_time_per_image": infer_per_image,
            "flops": None,  # N/A for classical ML
            "parameters": n_params,
            "model_size_mb": model_size_mb,
        }
        print(f"  {name}: Acc={acc:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
              + (f", Top-5={top5:.4f}" if top5 else "")
              + f" ({train_time:.1f}s)")

    return results


# ============================================================
# 4. Approach A — Fine-Tuning ResNet-50
# ============================================================
def build_resnet50_finetune():
    """Build ResNet-50 with new classifier head for 200 classes."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def finetune_resnet50(train_loader, val_loader):
    """
    Two-stage fine-tuning:
      Stage 1: Freeze backbone, train only fc head (5 epochs)
      Stage 2: Unfreeze all, fine-tune with lower LR (20 epochs)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_resnet50_finetune()
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    total_train_start = time.time()

    # ---- Stage 1: Freeze backbone ----
    print("\n=== Stage 1: Training classifier head only ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch+1}/5 — Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet50_best.pth"))

    # ---- Stage 2: Unfreeze all layers ----
    print("\n=== Stage 2: Fine-tuning entire network ===")
    for param in model.parameters():
        param.requires_grad = True

    # Differential learning rates
    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    head_params = list(model.fc.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-3},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch+1}/20 — Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet50_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    total_train_time = time.time() - total_train_start
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_train_time:.1f}s")

    # Load best model for test evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet50_best.pth"), weights_only=True))
    return model, history, total_train_time


@torch.no_grad()
def test_finetuned_model(model, test_loader, train_time):
    """Evaluate fine-tuned model on test set with all metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    start = time.time()
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs)
    infer_time = time.time() - start

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs)
    n_test = len(all_labels)

    top1 = accuracy_score(all_labels, all_preds)
    top5 = top_k_accuracy_score(all_labels, all_probs, k=5)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())

    # Model size on disk
    model_path = os.path.join(MODEL_DIR, "resnet50_best.pth")
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else None

    # FLOPs estimation (using thop if available, else known ResNet-50 value)
    flops = None
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except ImportError:
        flops = 4.1e9  # ResNet-50 ~4.1 GFLOPs (well-known value)

    print(f"\nFine-tuned ResNet-50 Test Results:")
    print(f"  Accuracy:  {top1:.4f}")
    print(f"  Top-5:     {top5:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Params:    {n_params:,}")
    print(f"  FLOPs:     {flops:.2e}" if flops else "  FLOPs:     N/A")
    print(f"  Model size:{model_size_mb:.1f} MB" if model_size_mb else "  Model size: N/A")

    return {
        "accuracy": top1,
        "top5": top5,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time": train_time,
        "infer_time_per_image": infer_time / n_test,
        "flops": flops,
        "parameters": n_params,
        "model_size_mb": model_size_mb,
    }


# ============================================================
# 5. Visualization
# ============================================================
def plot_training_history(history):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="Stage 1→2")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="Stage 1→2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.join(BASE_DIR, "showcase"), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, "showcase", "training_curves.png"), dpi=150)
    plt.show()


def print_comparison(classical_results, finetune_results):
    """Print a comparison table of all models."""
    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'Acc':>7} {'Top-5':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Train(s)':>9} {'Infer(ms)':>10}")
    print("=" * 100)

    all_results = {}
    if classical_results:
        all_results.update(classical_results)
    if finetune_results:
        all_results["ResNet-50 Fine-Tuned"] = finetune_results

    for name, r in all_results.items():
        top5_str = f"{r['top5']:.4f}" if r.get("top5") else "N/A"
        infer_ms = r["infer_time_per_image"] * 1000
        print(f"{name:<25} {r['accuracy']:>7.4f} {top5_str:>7} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['train_time']:>9.1f} {infer_ms:>10.3f}")

    print("=" * 100)


def save_results_csv(classical_results, finetune_results):
    """Save results to two CSV files: performance and resource."""
    csv_dir = os.path.join(BASE_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    all_results = {}
    if classical_results:
        all_results.update(classical_results)
    if finetune_results:
        all_results["ResNet-50 Fine-Tuned"] = finetune_results

    # --- Performance CSV ---
    perf_path = os.path.join(csv_dir, "classification_performance.csv")
    perf_fields = ["Model", "Precision", "Recall", "F1-Score", "Accuracy", "Top-5 Accuracy"]

    with open(perf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for name, r in all_results.items():
            writer.writerow({
                "Model": name,
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1-Score": f"{r['f1']:.4f}",
                "Accuracy": f"{r['accuracy']:.4f}",
                "Top-5 Accuracy": f"{r['top5']:.4f}" if r.get("top5") else "N/A",
            })

    # --- Resource CSV ---
    res_path = os.path.join(csv_dir, "classification_resource.csv")
    res_fields = ["Model", "Training Time (s)", "Inference Time (ms/image)", "FLOPs", "Parameters", "Model Size (MB)"]

    with open(res_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_fields)
        writer.writeheader()
        for name, r in all_results.items():
            writer.writerow({
                "Model": name,
                "Training Time (s)": f"{r['train_time']:.1f}",
                "Inference Time (ms/image)": f"{r['infer_time_per_image'] * 1000:.3f}",
                "FLOPs": f"{r['flops']:.2e}" if r.get("flops") else "N/A",
                "Parameters": f"{r['parameters']:,}" if r.get("parameters") else "N/A",
                "Model Size (MB)": f"{r['model_size_mb']:.1f}" if r.get("model_size_mb") else "N/A",
            })

    print(f"\nResults saved to:\n  {perf_path}\n  {res_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUB-200-2011 Bird Classification")
    parser.add_argument("--approach", type=str, default="both", choices=["A", "B", "both"],
                        help="Which approach to run: A (fine-tuning), B (classical ML), or both")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Approach: {args.approach}")
    print()

    # --- Build DataLoaders ---
    train_loader, val_loader, test_loader = build_dataloaders()

    classical_results = None
    finetune_results = None

    # --- Approach B: Feature Extraction + Classical ML ---
    if args.approach in ("B", "both"):
        print("\n" + "=" * 60)
        print("APPROACH B: Feature Extraction + Classical ML")
        print("=" * 60)
        features_dict = extract_features()
        classical_results = run_classical_ml(features_dict)

    # --- Approach A: Fine-Tuning ---
    if args.approach in ("A", "both"):
        print("\n" + "=" * 60)
        print("APPROACH A: Fine-Tuning ResNet-50")
        print("=" * 60)
        model, history, train_time = finetune_resnet50(train_loader, val_loader)
        finetune_results = test_finetuned_model(model, test_loader, train_time)
        plot_training_history(history)

    # --- Compare & Save ---
    print_comparison(classical_results, finetune_results)
    save_results_csv(classical_results, finetune_results)