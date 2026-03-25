"""
Generate t-SNE embeddings for CUB images using ResNet50 ImageNet-pretrained features.

Output DataFrame schema:
	image_name, breed, tsne_x, tsne_y
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"

IMAGES_TXT = DATA_DIR / "images.txt"
IMAGE_CLASS_LABELS_TXT = DATA_DIR / "image_class_labels.txt"
CLASSES_TXT = DATA_DIR / "classes.txt"
IMAGE_ROOT = DATA_DIR / "images"

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "csv" / "tsne_embeddings.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate t-SNE embedding from ResNet50 ImageNet features."
	)
	parser.add_argument("--images-txt", type=Path, default=IMAGES_TXT)
	parser.add_argument("--labels-txt", type=Path, default=IMAGE_CLASS_LABELS_TXT)
	parser.add_argument("--classes-txt", type=Path, default=CLASSES_TXT)
	parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--max-images", type=int, default=0, help="0 means use all images")
	parser.add_argument("--pca-dim", type=int, default=50, help="0 disables PCA before t-SNE")
	parser.add_argument("--perplexity", type=float, default=30.0)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def load_mapping(path: Path) -> dict[int, str]:
	mapping: dict[int, str] = {}
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			idx, value = line.split(" ", 1)
			mapping[int(idx)] = value
	return mapping


def load_int_mapping(path: Path) -> dict[int, int]:
	mapping: dict[int, int] = {}
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			idx, value = line.split()
			mapping[int(idx)] = int(value)
	return mapping


def build_records(
	images_map: dict[int, str],
	labels_map: dict[int, int],
	classes_map: dict[int, str],
	image_root: Path,
	max_images: int,
) -> List[Tuple[Path, str, str]]:
	records: List[Tuple[Path, str, str]] = []
	for image_idx in sorted(images_map.keys()):
		rel_path = images_map[image_idx]
		class_idx = labels_map.get(image_idx)
		if class_idx is None:
			continue

		breed = classes_map.get(class_idx, "unknown")
		image_path = image_root / rel_path
		if not image_path.exists():
			continue

		image_name = Path(rel_path).stem
		records.append((image_path, image_name, breed))

		if max_images > 0 and len(records) >= max_images:
			break

	return records


class CUBDataset(Dataset):
	def __init__(self, records: List[Tuple[Path, str, str]], transform):
		self.records = records
		self.transform = transform

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int):
		image_path, image_name, breed = self.records[index]
		with Image.open(image_path) as img:
			img = img.convert("RGB")
			tensor = self.transform(img)
		return tensor, image_name, breed


def build_feature_model(device: str) -> tuple[nn.Module, object]:
	weights = ResNet50_Weights.IMAGENET1K_V2
	model = resnet50(weights=weights)
	model.fc = nn.Identity()
	model.eval()
	model.to(device)
	return model, weights.transforms()


def extract_features(
	model: nn.Module,
	dataloader: DataLoader,
	device: str,
) -> tuple[np.ndarray, List[str], List[str]]:
	all_features: List[np.ndarray] = []
	image_names: List[str] = []
	breeds: List[str] = []

	with torch.no_grad():
		for images, batch_names, batch_breeds in tqdm(dataloader, desc="Extracting features"):
			images = images.to(device)
			feats = model(images)
			feats_np = feats.cpu().numpy()

			all_features.append(feats_np)
			image_names.extend(list(batch_names))
			breeds.extend(list(batch_breeds))

	features_np = np.concatenate(all_features, axis=0) if all_features else np.zeros((0, 2048), dtype=np.float32)
	return features_np, image_names, breeds


def main() -> None:
	args = parse_args()

	images_map = load_mapping(args.images_txt)
	labels_map = load_int_mapping(args.labels_txt)
	classes_map = load_mapping(args.classes_txt)

	records = build_records(
		images_map=images_map,
		labels_map=labels_map,
		classes_map=classes_map,
		image_root=args.image_root,
		max_images=args.max_images,
	)
	if not records:
		raise ValueError("No valid image records found.")

	model, transform = build_feature_model(args.device)
	dataset = CUBDataset(records=records, transform=transform)
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=(args.device.startswith("cuda")),
	)

	features, image_names, breeds = extract_features(model, dataloader, args.device)
	if features.shape[0] < 2:
		raise ValueError("Need at least 2 samples for t-SNE.")

	tsne_input = features
	if args.pca_dim > 0 and args.pca_dim < features.shape[1]:
		pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
		tsne_input = pca.fit_transform(features)

	tsne = TSNE(
		n_components=2,
		perplexity=args.perplexity,
		random_state=args.random_state,
		init="pca",
		learning_rate="auto",
	)
	tsne_result = tsne.fit_transform(tsne_input)

	df = pd.DataFrame(
		{
			"image_name": image_names,
			"breed": breeds,
			"tsne_x": tsne_result[:, 0],
			"tsne_y": tsne_result[:, 1],
		}
	)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output_csv, index=False)

	print(f"Saved: {args.output_csv}")
	print(f"Rows: {len(df):,}")


if __name__ == "__main__":
	main()

