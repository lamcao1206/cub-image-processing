import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


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
    data_dir: Path,
    image_id: int,
    rel_path: str,
    class_names: Dict[int, str],
    class_by_image: Dict[int, int],
    split_by_image: Dict[int, int],
    small_max: int,
    medium_max: int,
) -> Optional[Dict[str, object]]:
    image_path = data_dir / "images" / rel_path

    if not image_path.exists():
        return None

    file_size_by = image_path.stat().st_size
    file_size_kb = file_size_by / 1024.0

    with Image.open(image_path) as img:
        width, height = img.size

    area = width * height
    class_id = class_by_image[image_id]
    breed = class_names[class_id]
    split = "train" if split_by_image.get(image_id, 0) == 1 else "test"

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
        "size_category": area_bucket(area, small_max, medium_max),
        "orientation": orientation(width, height),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-image statistics CSV from CUB metadata and image files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to CUB data directory (default: data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "image_statistics.csv",
        help="Output CSV path (default: data/image_statistics.csv)",
    )
    parser.add_argument(
        "--small-max-area",
        type=int,
        default=100_000,
        help="Max pixel area for 'small' images (default: 100000)",
    )
    parser.add_argument(
        "--medium-max-area",
        type=int,
        default=500_000,
        help="Max pixel area for 'medium' images (default: 500000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N images for quick tests (default: 0 = all)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N processed images (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    images_map = load_kv_pairs(data_dir / "images.txt")
    class_names = load_kv_pairs(data_dir / "classes.txt")
    class_by_image = load_int_pairs(data_dir / "image_class_labels.txt")
    split_by_image = load_int_pairs(data_dir / "train_test_split.txt")

    args.output.parent.mkdir(parents=True, exist_ok=True)

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
    if args.limit > 0:
        image_ids = image_ids[: args.limit]

    total = len(image_ids)
    processed = 0
    written = 0
    skipped_missing = 0
    started_at = time.time()

    logging.info("Starting CSV generation for %d images", total)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in image_ids:
            rel_path = images_map[image_id]
            row = build_row(
                data_dir=data_dir,
                image_id=image_id,
                rel_path=rel_path,
                class_names=class_names,
                class_by_image=class_by_image,
                split_by_image=split_by_image,
                small_max=args.small_max_area,
                medium_max=args.medium_max_area,
            )
            processed += 1

            if row is None:
                skipped_missing += 1
            else:
                writer.writerow(row)
                written += 1

            if args.log_every > 0 and (processed % args.log_every == 0 or processed == total):
                elapsed = max(time.time() - started_at, 1e-9)
                rate = processed / elapsed
                logging.info(
                    "Progress: %d/%d (%.2f%%) | written=%d | skipped_missing=%d | %.2f img/s",
                    processed,
                    total,
                    (processed / total * 100.0) if total else 100.0,
                    written,
                    skipped_missing,
                    rate,
                )

    elapsed_total = time.time() - started_at
    logging.info(
        "Finished. output=%s | total=%d | written=%d | skipped_missing=%d | elapsed=%.2fs",
        args.output,
        total,
        written,
        skipped_missing,
        elapsed_total,
    )


if __name__ == "__main__":
    main()
