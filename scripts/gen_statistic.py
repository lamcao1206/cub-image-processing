import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# ── Project root is two levels up from this script (scripts/core_eda/) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── Hardcoded paths relative to the project root ──
DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"
IMAGES_FILE = DATA_DIR / "images.txt"
CLASSES_FILE = DATA_DIR / "classes.txt"
IMAGE_CLASS_LABELS_FILE = DATA_DIR / "image_class_labels.txt"
TRAIN_VAL_TEST_SPLIT_FILE = DATA_DIR / "train_val_test_split.txt"

# ── Output goes into the csv/ folder at the project root ──
CSV_DIR = PROJECT_ROOT / "csv"
OUTPUT_FILE = CSV_DIR / "image_statistic.csv"
SMALL_MAX_AREA = 100_000
MEDIUM_MAX_AREA = 500_000
LOG_EVERY = 200

LOGGER = logging.getLogger("data_preprocessing")


def load_kv_pairs(path: Path) -> Dict[int, str]:
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
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split()
            mapping[int(key)] = int(value)
    return mapping


def orientation(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def area_bucket(area: int, small_max: int, medium_max: int) -> str:
    if area <= small_max:
        return "small"
    if area <= medium_max:
        return "medium"
    return "large"


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    core = gray[1:-1, 1:-1]
    lap = (
        -4.0 * core
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def image_stats(image_path: Path) -> Dict[str, float]:
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        rgb = np.asarray(img_rgb, dtype=np.float32)
        gray = np.asarray(img_rgb.convert("L"), dtype=np.float32)

    means = rgb.mean(axis=(0, 1))
    stds = rgb.std(axis=(0, 1))

    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": laplacian_variance(gray),
    }


def build_row(
    image_id: int,
    rel_path: str,
    class_names: Dict[int, str],
    class_by_image: Dict[int, int],
    split_by_image: Dict[int, str],
) -> Optional[Dict[str, object]]:
    image_path = DATA_DIR / "images" / rel_path

    if not image_path.exists():
        return None

    file_size_by = image_path.stat().st_size
    file_size_kb = file_size_by / 1024.0

    with Image.open(image_path) as img:
        width, height = img.size

    area = width * height
    class_id = class_by_image[image_id]
    breed = class_names[class_id]
    split = split_by_image.get(image_id, "unknown")

    stats = image_stats(image_path)

    return {
        "image_id": Path(rel_path).stem,
        "breed": breed,
        "split": split,
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": width / height if height else 0.0,
        "file_size_by": file_size_by,
        "file_size_kb": file_size_kb,
        **stats,
        "size_category": area_bucket(area, SMALL_MAX_AREA, MEDIUM_MAX_AREA),
        "orientation": orientation(width, height),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Data directory : %s", DATA_DIR)
    LOGGER.info("Output file    : %s", OUTPUT_FILE)

    images_map = load_kv_pairs(IMAGES_FILE)
    class_names = load_kv_pairs(CLASSES_FILE)
    class_by_image = load_int_pairs(IMAGE_CLASS_LABELS_FILE)
    split_by_image = load_kv_pairs(TRAIN_VAL_TEST_SPLIT_FILE)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "breed",
        "split",
        "width",
        "height",
        "area",
        "aspect_ratio",
        "file_size_by",
        "file_size_kb",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "brightness",
        "contrast",
        "sharpness",
        "size_category",
        "orientation",
    ]

    image_ids: List[int] = sorted(images_map.keys())

    total = len(image_ids)
    written_rows: List[int] = []
    skipped_ids: List[int] = []
    started_at = time.time()

    LOGGER.info("Starting CSV generation for %d images", total)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for processed, image_id in enumerate(image_ids, start=1):
            rel_path = images_map[image_id]
            row = build_row(
                image_id=image_id,
                rel_path=rel_path,
                class_names=class_names,
                class_by_image=class_by_image,
                split_by_image=split_by_image,
            )

            if row is None:
                skipped_ids.append(image_id)
            else:
                writer.writerow(row)
                written_rows.append(image_id)

            if LOG_EVERY > 0 and (processed % LOG_EVERY == 0 or processed == total):
                elapsed = max(time.time() - started_at, 1e-9)
                rate = processed / elapsed
                LOGGER.info(
                    "Progress: %d/%d (%.2f%%) | written=%d | skipped_missing=%d | %.2f img/s",
                    processed,
                    total,
                    (processed / total * 100.0) if total else 100.0,
                    len(written_rows),
                    len(skipped_ids),
                    rate,
                )

    elapsed_total = time.time() - started_at
    LOGGER.info(
        "Finished. output=%s | total=%d | written=%d | skipped_missing=%d | elapsed=%.2fs",
        OUTPUT_FILE,
        total,
        len(written_rows),
        len(skipped_ids),
        elapsed_total,
    )


if __name__ == "__main__":
    main()
