"""
Generate per-breed shape distribution statistics from segmentation masks.

Default input:
- data/CUB_200_2011/segmentations

Default output:
- csv/shape_distribution.csv

Output columns:
breed,
convexity_mean,convexity_std,convexity_min,convexity_max,
compactness_mean,compactness_std,compactness_min,compactness_max,
eccentricity_mean,eccentricity_std,eccentricity_min,eccentricity_max,
count
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SEGMENTATIONS_DIR = PROJECT_ROOT / "data" / "CUB_200_2011" / "segmentations"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "csv" / "shape_distribution.csv"

LOGGER = logging.getLogger("gen_shape_distribution")

OUTPUT_FIELDS = [
    "breed",
    "convexity_mean",
    "convexity_std",
    "convexity_min",
    "convexity_max",
    "compactness_mean",
    "compactness_std",
    "compactness_min",
    "compactness_max",
    "eccentricity_mean",
    "eccentricity_std",
    "eccentricity_min",
    "eccentricity_max",
    "count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-breed shape distribution CSV from segmentation masks."
    )
    parser.add_argument(
        "--segmentations-dir",
        type=Path,
        default=DEFAULT_SEGMENTATIONS_DIR,
        help="Path to segmentation root folder",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N masks (default: 500)",
    )
    return parser.parse_args()


def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def convex_hull(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def mask_boundary(binary_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(binary_mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = center & up & down & left & right
    return center & (~interior)


def calc_shape_metrics(mask_path: Path) -> Dict[str, float]:
    arr = np.array(Image.open(mask_path), dtype=np.uint8)
    if arr.ndim > 2:
        arr = arr[..., 0]

    # Object region: any non-background label.
    mask = arr > 0
    area = int(mask.sum())
    if area <= 0:
        return {"convexity": 0.0, "compactness": 0.0, "eccentricity": 0.0}

    boundary = mask_boundary(mask)
    perimeter = float(boundary.sum())

    # Compactness: 4*pi*A / P^2 (higher means more compact / circular).
    compactness = 0.0
    if perimeter > 0:
        compactness = float(4.0 * math.pi * area / (perimeter * perimeter))

    # Eccentricity from covariance eigenvalues of foreground coordinates.
    coords = np.argwhere(mask)
    if coords.shape[0] <= 2:
        eccentricity = 0.0
    else:
        xy = np.stack([coords[:, 1], coords[:, 0]], axis=1).astype(np.float64)
        cov = np.cov(xy, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        major = float(eigvals[-1])
        minor = float(eigvals[0])
        if major <= 0:
            eccentricity = 0.0
        else:
            eccentricity = float(math.sqrt(max(0.0, 1.0 - (minor / major))))

    # Convexity approximation via convex hull area ratio.
    boundary_points_idx = np.argwhere(boundary)
    points = [(float(c), float(r)) for r, c in boundary_points_idx]
    hull = convex_hull(points)
    hull_area = polygon_area(hull)
    if hull_area <= 0:
        convexity = 1.0
    else:
        convexity = float(min(1.0, max(0.0, area / hull_area)))

    compactness = float(min(1.0, max(0.0, compactness)))
    eccentricity = float(min(1.0, max(0.0, eccentricity)))

    return {
        "convexity": convexity,
        "compactness": compactness,
        "eccentricity": eccentricity,
    }


def summarize_breed(breed: str, rows: List[Dict[str, float]]) -> Dict[str, float | int | str]:
    convexity_vals = [r["convexity"] for r in rows]
    compactness_vals = [r["compactness"] for r in rows]
    eccentricity_vals = [r["eccentricity"] for r in rows]

    return {
        "breed": breed,
        "convexity_mean": float(statistics.fmean(convexity_vals)),
        "convexity_std": safe_std(convexity_vals),
        "convexity_min": float(min(convexity_vals)),
        "convexity_max": float(max(convexity_vals)),
        "compactness_mean": float(statistics.fmean(compactness_vals)),
        "compactness_std": safe_std(compactness_vals),
        "compactness_min": float(min(compactness_vals)),
        "compactness_max": float(max(compactness_vals)),
        "eccentricity_mean": float(statistics.fmean(eccentricity_vals)),
        "eccentricity_std": safe_std(eccentricity_vals),
        "eccentricity_min": float(min(eccentricity_vals)),
        "eccentricity_max": float(max(eccentricity_vals)),
        "count": len(rows),
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    seg_root = args.segmentations_dir
    if not seg_root.exists():
        raise FileNotFoundError(f"Could not find segmentations directory: {seg_root}")

    class_dirs = sorted([p for p in seg_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in: {seg_root}")

    LOGGER.info("segmentations dir: %s", seg_root)
    LOGGER.info("class folders     : %d", len(class_dirs))

    per_breed_rows: Dict[str, List[Dict[str, float]]] = {}
    processed = 0

    for class_dir in class_dirs:
        breed = class_dir.name
        mask_paths = sorted(class_dir.glob("*.png"))
        if not mask_paths:
            continue

        bucket = per_breed_rows.setdefault(breed, [])
        for mask_path in mask_paths:
            bucket.append(calc_shape_metrics(mask_path))
            processed += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                LOGGER.info(
                    "Progress: %d masks processed | breeds=%d",
                    processed,
                    len(per_breed_rows),
                )

    if processed == 0:
        raise ValueError(f"No PNG mask files found under: {seg_root}")

    out_rows = [
        summarize_breed(breed=breed, rows=rows)
        for breed, rows in sorted(per_breed_rows.items())
        if rows
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    LOGGER.info("Done. masks=%d | breeds=%d | output=%s", processed, len(out_rows), args.output_csv)


if __name__ == "__main__":
    main()
