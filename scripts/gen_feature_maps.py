"""
Visualize ResNet-50 feature extraction at multiple layers.

Picks 5 sample images from different classes, passes each through
a pretrained ResNet-50, and saves EACH filter as a separate image file.

Output structure:
  showcase/feature_maps/
    sample_0_black_footed_albatross/
        original.png
        conv1_filter_0.png
        conv1_filter_1.png
        ...
        conv1_filter_7.png
        layer1_filter_0.png
        ...
        layer4_filter_7.png
        summary.png
    sample_1_laysan_albatross/
        ...

Usage:
    python scripts/gen_feature_maps.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "CUB_200_2011")
IMAGE_DIR = os.path.join(BASE_DIR, "precompute", "preprocessed_images")
IMAGES_TXT = os.path.join(DATA_DIR, "images.txt")
LABELS_TXT = os.path.join(DATA_DIR, "image_class_labels.txt")
CLASSES_TXT = os.path.join(DATA_DIR, "classes.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "showcase", "feature_maps")

NUM_SAMPLES = 5
NUM_FILTERS_TO_SHOW = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same as main.py val_transform
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_class_names():
    names = {}
    with open(CLASSES_TXT, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            names[int(parts[0]) - 1] = parts[1]
    return names


def pick_sample_images(n=5):
    """Pick n sample images from different classes."""
    image_paths = {}
    with open(IMAGES_TXT, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_paths[int(parts[0])] = parts[1]

    image_labels = {}
    with open(LABELS_TXT, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_labels[int(parts[0])] = int(parts[1]) - 1

    seen_classes = set()
    samples = []
    for img_id, rel_path in image_paths.items():
        label = image_labels[img_id]
        if label not in seen_classes:
            full_path = os.path.join(IMAGE_DIR, rel_path)
            if os.path.exists(full_path):
                samples.append((full_path, label))
                seen_classes.add(label)
        if len(samples) >= n:
            break
    return samples


def get_feature_maps(model, image_tensor):
    """Hook into ResNet-50 layers and capture feature maps."""
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = [
        model.conv1.register_forward_hook(make_hook("conv1")),
        model.layer1.register_forward_hook(make_hook("layer1")),
        model.layer2.register_forward_hook(make_hook("layer2")),
        model.layer3.register_forward_hook(make_hook("layer3")),
        model.layer4.register_forward_hook(make_hook("layer4")),
    ]

    with torch.no_grad():
        model(image_tensor)

    for h in hooks:
        h.remove()

    return activations


def save_single_filter(fmap_2d, save_path, title=None):
    """Save a single filter activation as its own image file."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(fmap_2d, cmap="viridis")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    samples = pick_sample_images(NUM_SAMPLES)
    class_names = load_class_names()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(DEVICE)
    model.eval()

    layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]

    for idx, (img_path, label) in enumerate(samples):
        breed = class_names.get(label, f"class_{label}")
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx}_{breed}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\nSample {idx}: {breed}")

        # Save original
        orig = Image.open(img_path).convert("RGB").resize((224, 224), Image.LANCZOS)
        orig.save(os.path.join(sample_dir, "original.png"))
        print(f"  Saved original.png")

        # Extract feature maps
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        activations = get_feature_maps(model, tensor)

        # Save each filter as a separate file
        for layer_name in layers:
            feat = activations[layer_name][0]  # (C, H, W)
            total_filters = feat.shape[0]
            n = min(NUM_FILTERS_TO_SHOW, total_filters)
            print(f"  {layer_name}: {total_filters} total filters, {feat.shape[1]}x{feat.shape[2]}, saving {n}")

            for fi in range(n):
                fmap = feat[fi].numpy()

                plain_path = os.path.join(sample_dir, f"{layer_name}_filter_{fi}.png")
                save_single_filter(fmap, plain_path, title=f"{layer_name} - Filter {fi}")

        # Summary: original + first filter from each layer
        fig, axes = plt.subplots(1, len(layers) + 1, figsize=(3.5 * (len(layers) + 1), 3.5))
        axes[0].imshow(np.array(orig))
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        for i, layer_name in enumerate(layers):
            feat = activations[layer_name][0]
            axes[i + 1].imshow(feat[0].numpy(), cmap="viridis")
            axes[i + 1].set_title(f"{layer_name}\n{feat.shape[1]}x{feat.shape[2]}", fontsize=9)
            axes[i + 1].axis("off")

        fig.suptitle(f"ResNet-50 Feature Extraction — {breed}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved summary.png")

    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
