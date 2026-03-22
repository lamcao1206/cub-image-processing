from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SEGMENTATIONS_DIR = REPO_ROOT / "data" / "segmentations"
IMAGE_STATS_CSV = REPO_ROOT / "data" / "image_statistics.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "mask_statistics.csv"


def build_split_lookup(image_stats_csv: Path) -> dict[str, str]:
	if not image_stats_csv.exists():
		raise FileNotFoundError(f"Could not find split source CSV: {image_stats_csv}")

	df = pd.read_csv(image_stats_csv)
	required = {"image_id", "split"}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Missing columns in {image_stats_csv}: {sorted(missing)}")

	split_map: dict[str, str] = {}
	for _, row in df[["image_id", "split"]].dropna().iterrows():
		split_map[str(row["image_id"]) ] = str(row["split"])
	return split_map


def calculate_mask_stats(mask_path: Path) -> dict[str, float | int]:
	mask_arr = np.array(Image.open(mask_path), dtype=np.uint8)
	if mask_arr.ndim > 2:
		# If multi-channel, use the first channel which carries the label values.
		mask_arr = mask_arr[..., 0]

	h, w = mask_arr.shape
	total_pixels = int(h * w)

	bg_pixels = int((mask_arr == 0).sum())
	fg_pixels = int((mask_arr == 255).sum())
	boundary_pixels = int(((mask_arr > 0) & (mask_arr < 255)).sum())

	fg_pct = (100.0 * fg_pixels / total_pixels) if total_pixels else 0.0
	boundary_pct = (100.0 * boundary_pixels / total_pixels) if total_pixels else 0.0
	bg_pct = (100.0 * bg_pixels / total_pixels) if total_pixels else 0.0

	# Coverage means any non-background mask area (foreground + boundary).
	mask_coverage = ((fg_pixels + boundary_pixels) / total_pixels) if total_pixels else 0.0

	return {
		"total_pixels": total_pixels,
		"fg_pixels": fg_pixels,
		"boundary_pixels": boundary_pixels,
		"bg_pixels": bg_pixels,
		"fg_percentage": fg_pct,
		"boundary_percentage": boundary_pct,
		"bg_percentage": bg_pct,
		"mask_coverage": mask_coverage,
		"mask_height": int(h),
		"mask_width": int(w),
	}


def main() -> None:
	if not SEGMENTATIONS_DIR.exists():
		raise FileNotFoundError(f"Could not find segmentations directory: {SEGMENTATIONS_DIR}")

	split_lookup = build_split_lookup(IMAGE_STATS_CSV)

	rows: list[dict[str, object]] = []
	class_dirs = sorted([p for p in SEGMENTATIONS_DIR.iterdir() if p.is_dir()])

	for class_dir in class_dirs:
		breed = class_dir.name
		for mask_path in sorted(class_dir.glob("*.png")):
			image_id = mask_path.stem
			split = split_lookup.get(image_id, "unknown")

			stats = calculate_mask_stats(mask_path)
			rows.append(
				{
					"image_id": image_id,
					"breed": breed,
					"species": "bird",
					"split": split,
					**stats,
				}
			)

	if not rows:
		raise ValueError(f"No mask files found under: {SEGMENTATIONS_DIR}")

	out_df = pd.DataFrame(rows)
	ordered_cols = [
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
	out_df = out_df[ordered_cols]

	OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(OUTPUT_CSV, index=False)

	print(f"Saved: {OUTPUT_CSV}")
	print(f"Rows: {len(out_df):,}")
	print("Tri-map interpretation: background=0, boundary=1..254, foreground=255")


if __name__ == "__main__":
	main()
