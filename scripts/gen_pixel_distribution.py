"""
Generate per-breed pixel-distribution statistics from segmentation masks.

Default input:
- data/CUB_200_2011/segmentations

Default output:
- csv/pixel_distribution.csv

Output columns:
breed,
fg_percentage_mean,fg_percentage_std,fg_percentage_min,fg_percentage_max,
boundary_percentage_mean,boundary_percentage_std,boundary_percentage_min,boundary_percentage_max,
bg_percentage_mean,bg_percentage_std,bg_percentage_min,bg_percentage_max,
mask_coverage_mean,mask_coverage_std,count

Mask convention (CUB trimap-like values):
- background: 0
- foreground: 255
- boundary/uncertain: 1..254
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SEGMENTATIONS_DIR = PROJECT_ROOT / "data" / "CUB_200_2011" / "segmentations"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "csv" / "pixel_distribution.csv"

LOGGER = logging.getLogger("gen_pixel_distribution")

OUTPUT_FIELDS = [
	"breed",
	"fg_percentage_mean",
	"fg_percentage_std",
	"fg_percentage_min",
	"fg_percentage_max",
	"boundary_percentage_mean",
	"boundary_percentage_std",
	"boundary_percentage_min",
	"boundary_percentage_max",
	"bg_percentage_mean",
	"bg_percentage_std",
	"bg_percentage_min",
	"bg_percentage_max",
	"mask_coverage_mean",
	"mask_coverage_std",
	"count",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate per-breed pixel distribution CSV from segmentation masks."
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
		default=1000,
		help="Log progress every N masks (default: 1000)",
	)
	return parser.parse_args()


def safe_std(values: List[float]) -> float:
	# Population std: stable when count is small and avoids NaN for count=1.
	if len(values) <= 1:
		return 0.0
	return statistics.pstdev(values)


def calc_mask_percentages(mask_path: Path) -> Dict[str, float]:
	arr = np.array(Image.open(mask_path), dtype=np.uint8)
	if arr.ndim > 2:
		arr = arr[..., 0]

	h, w = arr.shape
	total = h * w
	if total <= 0:
		return {
			"fg_percentage": 0.0,
			"boundary_percentage": 0.0,
			"bg_percentage": 0.0,
			"mask_coverage": 0.0,
		}

	fg = int((arr == 255).sum())
	bg = int((arr == 0).sum())
	boundary = int(((arr > 0) & (arr < 255)).sum())

	fg_pct = 100.0 * fg / total
	boundary_pct = 100.0 * boundary / total
	bg_pct = 100.0 * bg / total
	coverage = (fg + boundary) / total

	return {
		"fg_percentage": fg_pct,
		"boundary_percentage": boundary_pct,
		"bg_percentage": bg_pct,
		"mask_coverage": coverage,
	}


def summarize_breed(breed: str, rows: List[Dict[str, float]]) -> Dict[str, float | int | str]:
	fg_vals = [r["fg_percentage"] for r in rows]
	boundary_vals = [r["boundary_percentage"] for r in rows]
	bg_vals = [r["bg_percentage"] for r in rows]
	coverage_vals = [r["mask_coverage"] for r in rows]

	return {
		"breed": breed,
		"fg_percentage_mean": float(statistics.fmean(fg_vals)),
		"fg_percentage_std": float(safe_std(fg_vals)),
		"fg_percentage_min": float(min(fg_vals)),
		"fg_percentage_max": float(max(fg_vals)),
		"boundary_percentage_mean": float(statistics.fmean(boundary_vals)),
		"boundary_percentage_std": float(safe_std(boundary_vals)),
		"boundary_percentage_min": float(min(boundary_vals)),
		"boundary_percentage_max": float(max(boundary_vals)),
		"bg_percentage_mean": float(statistics.fmean(bg_vals)),
		"bg_percentage_std": float(safe_std(bg_vals)),
		"bg_percentage_min": float(min(bg_vals)),
		"bg_percentage_max": float(max(bg_vals)),
		"mask_coverage_mean": float(statistics.fmean(coverage_vals)),
		"mask_coverage_std": float(safe_std(coverage_vals)),
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
			bucket.append(calc_mask_percentages(mask_path))
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
