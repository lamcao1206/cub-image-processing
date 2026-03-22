import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

def load_id_str(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            mapping[int(key)] = value
    return mapping


def load_id_int(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            key, value = line.split()
            mapping[int(key)] = int(value)
    return mapping


def load_bboxes(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    mapping: Dict[int, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            items = line.split()
            if len(items) != 5:
                raise ValueError(f"Invalid bbox row in {path}: {line}")
            image_id = int(items[0])
            x, y, width, height = (float(items[1]), float(items[2]), float(items[3]), float(items[4]))
            mapping[image_id] = (x, y, width, height)
    return mapping


def size_category_by_coverage(coverage: float, small_thr: float, medium_thr: float) -> str:
    if coverage < small_thr:
        return "small"
    if coverage < medium_thr:
        return "medium"
    return "large"


def to_clean_number(value: float) -> float | int:
    if float(value).is_integer():
        return int(value)
    return value


def build_row(
    image_id: int,
    rel_path: str,
    class_name: str,
    split_value: int,
    bbox: Tuple[float, float, float, float],
    image_root: Path,
    coverage_small_thr: float,
    coverage_medium_thr: float,
) -> Optional[Dict[str, object]]:
    image_path = image_root / rel_path
    if not image_path.exists():
        return None

    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    area = width * height
    diagonal = math.sqrt(width * width + height * height)
    max_side = max(width, height)
    normalized_aspect_ratio = (min(width, height) / max_side) if max_side > 0 else 0.0

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    if image_width <= 0 or image_height <= 0:
        return None

    center_x = (xmin + (width / 2.0)) / image_width
    center_y = (ymin + (height / 2.0)) / image_height
    coverage = area / float(image_width * image_height)

    return {
        "image_id": Path(rel_path).stem,
        "breed": class_name,
        "split": "train" if split_value == 1 else "test",
        "width": to_clean_number(width),
        "height": to_clean_number(height),
        "area": to_clean_number(area),
        "normalized_aspect_ratio": normalized_aspect_ratio,
        "center_x": center_x,
        "center_y": center_y,
        "xmin": to_clean_number(xmin),
        "ymin": to_clean_number(ymin),
        "xmax": to_clean_number(xmax),
        "ymax": to_clean_number(ymax),
        "size_category": size_category_by_coverage(
            coverage=coverage,
            small_thr=coverage_small_thr,
            medium_thr=coverage_medium_thr,
        ),
        "diagonal": diagonal,
        "image_width": image_width,
        "image_height": image_height,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bounding-box analytics CSV from CUB txt metadata."
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
        default=Path("data") / "bbox_analytics.csv",
        help="Output CSV path (default: data/bbox_analytics.csv)",
    )
    parser.add_argument(
        "--coverage-small-threshold",
        type=float,
        default=0.10,
        help="Coverage ratio threshold below which size_category is small (default: 0.10)",
    )
    parser.add_argument(
        "--coverage-medium-threshold",
        type=float,
        default=0.20,
        help="Coverage ratio threshold below which size_category is medium (default: 0.20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N bboxes for quick tests (default: 0 means all)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N rows (default: 500)",
    )
    return parser.parse_args()


def validate_thresholds(small: float, medium: float) -> None:
    if not (0.0 <= small <= 1.0 and 0.0 <= medium <= 1.0):
        raise ValueError("Coverage thresholds must be in [0, 1].")
    if small >= medium:
        raise ValueError("coverage-small-threshold must be lower than coverage-medium-threshold.")


def iter_image_ids(ids: Iterable[int], limit: int) -> List[int]:
    ordered = sorted(ids)
    if limit > 0:
        return ordered[:limit]
    return ordered


def main() -> None:
    args = parse_args()
    validate_thresholds(args.coverage_small_threshold, args.coverage_medium_threshold)

    data_dir = args.data_dir
    image_root = data_dir / "images"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    images = load_id_str(data_dir / "images.txt")
    classes = load_id_str(data_dir / "classes.txt")
    image_class = load_id_int(data_dir / "image_class_labels.txt")
    image_split = load_id_int(data_dir / "train_test_split.txt")
    bboxes = load_bboxes(data_dir / "bounding_boxes.txt")

    image_ids = iter_image_ids(bboxes.keys(), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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

    written = 0
    skipped = 0

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_id in enumerate(image_ids, start=1):
            rel_path = images.get(image_id)
            class_id = image_class.get(image_id)
            split_value = image_split.get(image_id)
            bbox = bboxes.get(image_id)

            if rel_path is None or class_id is None or split_value is None or bbox is None:
                skipped += 1
                continue

            class_name = classes.get(class_id, "unknown")
            row = build_row(
                image_id=image_id,
                rel_path=rel_path,
                class_name=class_name,
                split_value=split_value,
                bbox=bbox,
                image_root=image_root,
                coverage_small_thr=args.coverage_small_threshold,
                coverage_medium_thr=args.coverage_medium_threshold,
            )

            if row is None:
                skipped += 1
                continue

            writer.writerow(row)
            written += 1

            if args.log_every > 0 and (idx % args.log_every == 0 or idx == len(image_ids)):
                logging.info(
                    "Progress: %d/%d | written=%d | skipped=%d",
                    idx,
                    len(image_ids),
                    written,
                    skipped,
                )

    logging.info("Done. Wrote %d rows to %s (skipped=%d)", written, args.output, skipped)


if __name__ == "__main__":
    main()