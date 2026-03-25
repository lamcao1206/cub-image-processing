"""
Generate bounding-box statistics CSV from CUB-200-2011 metadata.

For every image in the dataset, computes bbox geometry, coverage,
normalized aspect ratio, center position, size category, etc.
Outputs a single CSV to csv/bbox_statistic.csv.

Reference: scripts/hbox/generate_bbox_analytics.py
"""

import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ── Paths (no argparse – project convention) ─────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"

IMAGES_FILE = DATA_DIR / "images.txt"
CLASSES_FILE = DATA_DIR / "classes.txt"
IMAGE_CLASS_LABELS_FILE = DATA_DIR / "image_class_labels.txt"
BOUNDING_BOX_FILE = DATA_DIR / "bounding_boxes.txt"
TRAIN_VAL_TEST_SPLIT_FILE = DATA_DIR / "train_val_test_split.txt"

CSV_DIR = PROJECT_ROOT / "csv"
OUTPUT_FILE = CSV_DIR / "bbox_statistic.csv"

# ── Thresholds ───────────────────────────────────────────────────────────
COVERAGE_SMALL_THR = 0.10  # coverage < 10 % → "small"
COVERAGE_MEDIUM_THR = 0.20  # coverage < 20 % → "medium", else "large"
LOG_EVERY = 500

LOGGER = logging.getLogger("gen_bbox_statistic")


# ── Loaders ──────────────────────────────────────────────────────────────
def load_kv_pairs(path: Path) -> Dict[int, str]:
    """Load a file of '<int_key> <string_value>' lines."""
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            mapping[int(key)] = value
    return mapping


def load_int_pairs(path: Path) -> Dict[int, int]:
    """Load a file of '<int_key> <int_value>' lines."""
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split()
            mapping[int(key)] = int(value)
    return mapping


def load_bboxes(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    """Load bounding_boxes.txt → {image_id: (x, y, w, h)}."""
    mapping: Dict[int, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid bbox row in {path}: {line}")
            img_id = int(parts[0])
            x, y, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            mapping[img_id] = (x, y, w, h)
    return mapping


# ── Helpers ──────────────────────────────────────────────────────────────
def size_category(coverage: float) -> str:
    if coverage < COVERAGE_SMALL_THR:
        return "small"
    if coverage < COVERAGE_MEDIUM_THR:
        return "medium"
    return "large"


def clean_number(value: float) -> float | int:
    """Return int when the float is whole, else the float itself."""
    return int(value) if float(value).is_integer() else round(value, 4)


def build_row(
    image_id: int,
    rel_path: str,
    class_name: str,
    split_label: str,
    bbox: Tuple[float, float, float, float],
    image_root: Path,
) -> Optional[Dict[str, object]]:
    """Build one CSV row for the given image + bbox combo."""
    image_path = image_root / rel_path
    if not image_path.exists():
        return None

    xmin, ymin, bbox_w, bbox_h = bbox
    xmax = xmin + bbox_w
    ymax = ymin + bbox_h
    bbox_area = bbox_w * bbox_h
    diagonal = math.sqrt(bbox_w**2 + bbox_h**2)

    max_side = max(bbox_w, bbox_h)
    normalized_aspect_ratio = (min(bbox_w, bbox_h) / max_side) if max_side > 0 else 0.0

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    if img_w <= 0 or img_h <= 0:
        return None

    img_area = img_w * img_h
    coverage = bbox_area / float(img_area)

    center_x = (xmin + bbox_w / 2.0) / img_w
    center_y = (ymin + bbox_h / 2.0) / img_h

    return {
        "image_id": Path(rel_path).stem,
        "breed": class_name,
        "split": split_label,
        "width": clean_number(bbox_w),
        "height": clean_number(bbox_h),
        "area": clean_number(bbox_area),
        "normalized_aspect_ratio": round(normalized_aspect_ratio, 4),
        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),
        "xmin": clean_number(xmin),
        "ymin": clean_number(ymin),
        "xmax": clean_number(xmax),
        "ymax": clean_number(ymax),
        "size_category": size_category(coverage),
        "diagonal": round(diagonal, 2),
        "image_width": img_w,
        "image_height": img_h,
    }


FIELDNAMES = [
    "image_id",
    "breed",
    "split",
    "width",
    "height",
    "area",
    "normalized_aspect_ratio",
    "center_x",
    "center_y",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "size_category",
    "diagonal",
    "image_width",
    "image_height",
]


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Data directory : %s", DATA_DIR)
    LOGGER.info("Output file    : %s", OUTPUT_FILE)

    # Load metadata
    images_map = load_kv_pairs(IMAGES_FILE)
    class_names = load_kv_pairs(CLASSES_FILE)
    class_by_image = load_int_pairs(IMAGE_CLASS_LABELS_FILE)
    split_by_image = load_kv_pairs(TRAIN_VAL_TEST_SPLIT_FILE)
    bboxes = load_bboxes(BOUNDING_BOX_FILE)

    LOGGER.info(
        "Loaded %d images, %d classes, %d bounding boxes",
        len(images_map),
        len(class_names),
        len(bboxes),
    )

    image_root = DATA_DIR / "images"
    image_ids: List[int] = sorted(bboxes.keys())
    total = len(image_ids)

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    started_at = time.time()

    LOGGER.info("Starting CSV generation for %d images …", total)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for idx, image_id in enumerate(image_ids, start=1):
            rel_path = images_map.get(image_id)
            class_id = class_by_image.get(image_id)
            split_val = split_by_image.get(image_id, "unknown")
            bbox = bboxes.get(image_id)

            if rel_path is None or class_id is None or bbox is None:
                skipped += 1
                continue

            class_name = class_names.get(class_id, "unknown")

            row = build_row(
                image_id=image_id,
                rel_path=rel_path,
                class_name=class_name,
                split_label=split_val,
                bbox=bbox,
                image_root=image_root,
            )

            if row is None:
                skipped += 1
                continue

            writer.writerow(row)
            written += 1

            if LOG_EVERY > 0 and (idx % LOG_EVERY == 0 or idx == total):
                elapsed = max(time.time() - started_at, 1e-9)
                rate = idx / elapsed
                LOGGER.info(
                    "Progress: %d/%d (%.1f%%) | written=%d | skipped=%d | %.1f img/s",
                    idx,
                    total,
                    idx / total * 100.0 if total else 100.0,
                    written,
                    skipped,
                    rate,
                )

    elapsed_total = time.time() - started_at
    LOGGER.info(
        "Done. output=%s | written=%d | skipped=%d | elapsed=%.2fs",
        OUTPUT_FILE,
        written,
        skipped,
        elapsed_total,
    )


if __name__ == "__main__":
    main()
