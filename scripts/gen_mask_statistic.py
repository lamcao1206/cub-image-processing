"""
Generate mask statistics CSV from CUB segmentation masks.

Default input:
- data/CUB_200_2011/segmentations

Default output:
- csv/mask_statistics.csv

Output columns:
image_id,breed,species,split,total_pixels,fg_pixels,boundary_pixels,bg_pixels,
fg_percentage,boundary_percentage,bg_percentage,mask_coverage,mask_height,mask_width
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"

DEFAULT_SEGMENTATIONS_DIR = DATA_DIR / "segmentations"
DEFAULT_IMAGES_FILE = DATA_DIR / "images.txt"
DEFAULT_SPLIT_FILE = DATA_DIR / "train_val_test_split.txt"
FALLBACK_SPLIT_FILE = DATA_DIR / "train_test_split.txt"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "csv" / "mask_statistics.csv"

LOGGER = logging.getLogger("gen_mask_statistic")

OUTPUT_FIELDS = [
    "image_id",
    "breed",
    "species",
    "split",
    "total_pixels",
    "fg_pixels",
    "boundary_pixels",
    "bg_pixels",
    "fg_percentage",
    "boundary_percentage",
    "bg_percentage",
    "mask_coverage",
    "mask_height",
    "mask_width",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mask statistics CSV from CUB segmentation masks."
    )
    parser.add_argument(
        "--segmentations-dir",
        type=Path,
        default=DEFAULT_SEGMENTATIONS_DIR,
        help="Path to segmentation root folder",
    )
    parser.add_argument(
        "--images-file",
        type=Path,
        default=DEFAULT_IMAGES_FILE,
        help="Path to images.txt for id-to-name mapping",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="Path to split file (default: train_val_test_split.txt)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="bird",
        help="Species label to write in output (default: bird)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N masks (default: 1000)",
    )
    return parser.parse_args()


def read_images_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find images file: {path}")

    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            idx, rel = line.split(" ", 1)
            stem = Path(rel).stem.lower()
            out[int(idx)] = stem
    return out


def read_split_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find split file: {path}")

    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                key = int(parts[0])
                split = parts[1]
            else:
                key = i
                split = parts[0]
            out[key] = split.lower()
    return out


def build_imageid_to_split(
    images_map: Dict[int, str], split_map: Dict[int, str]
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for idx, image_id in images_map.items():
        split = split_map.get(idx, "unknown")
        out[image_id] = split
    return out


def calc_mask_stats(mask_path: Path) -> Dict[str, float | int]:
    arr = np.array(Image.open(mask_path), dtype=np.uint8)
    if arr.ndim > 2:
        arr = arr[..., 0]

    h, w = arr.shape
    total = int(h * w)

    fg = int((arr == 255).sum())
    bg = int((arr == 0).sum())
    boundary = int(((arr > 0) & (arr < 255)).sum())

    if total <= 0:
        return {
            "total_pixels": 0,
            "fg_pixels": 0,
            "boundary_pixels": 0,
            "bg_pixels": 0,
            "fg_percentage": 0.0,
            "boundary_percentage": 0.0,
            "bg_percentage": 0.0,
            "mask_coverage": 0.0,
            "mask_height": int(h),
            "mask_width": int(w),
        }

    fg_pct = 100.0 * fg / total
    boundary_pct = 100.0 * boundary / total
    bg_pct = 100.0 * bg / total
    coverage = (fg + boundary) / total

    return {
        "total_pixels": total,
        "fg_pixels": fg,
        "boundary_pixels": boundary,
        "bg_pixels": bg,
        "fg_percentage": fg_pct,
        "boundary_percentage": boundary_pct,
        "bg_percentage": bg_pct,
        "mask_coverage": coverage,
        "mask_height": int(h),
        "mask_width": int(w),
    }


def pick_split_file(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    if FALLBACK_SPLIT_FILE.exists():
        return FALLBACK_SPLIT_FILE
    raise FileNotFoundError(
        f"Could not find split file: {preferred} or fallback {FALLBACK_SPLIT_FILE}"
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    seg_root = args.segmentations_dir
    if not seg_root.exists():
        raise FileNotFoundError(f"Could not find segmentations directory: {seg_root}")

    split_file = pick_split_file(args.split_file)

    images_map = read_images_map(args.images_file)
    split_map = read_split_map(split_file)
    imageid_to_split = build_imageid_to_split(images_map, split_map)

    LOGGER.info("segmentations dir: %s", seg_root)
    LOGGER.info("split file       : %s", split_file)

    class_dirs = sorted([p for p in seg_root.iterdir() if p.is_dir()])
    rows = []
    processed = 0

    for class_dir in class_dirs:
        breed = class_dir.name.lower()
        for mask_path in sorted(class_dir.glob("*.png")):
            image_id = mask_path.stem.lower()
            split = imageid_to_split.get(image_id, "unknown")
            stats = calc_mask_stats(mask_path)

            rows.append(
                {
                    "image_id": image_id,
                    "breed": breed,
                    "species": args.species,
                    "split": split,
                    **stats,
                }
            )
            processed += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                LOGGER.info("Progress: %d masks processed", processed)

    if not rows:
        raise ValueError(f"No PNG files found under: {seg_root}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Done. rows=%d | output=%s", len(rows), args.output_csv)


if __name__ == "__main__":
    main()
