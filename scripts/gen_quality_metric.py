"""
Generate per-breed quality metrics CSV from hbox and image CSV data.

Default input:
- csv/bbox_analytics.csv
- csv/image_statistics.csv (optional fallback for image dimensions)

Default output:
- csv/quality_metrics.csv

Output columns:
breed,count,avg_width,avg_height,avg_area,avg_aspect_ratio,
avg_coverage,area_cv,aspect_cv,pct_small,pct_medium,pct_large
"""

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

DEFAULT_HBOX_CSV = CSV_DIR / "bbox_analytics.csv"
DEFAULT_IMAGE_CSV = CSV_DIR / "image_statistics.csv"
LEGACY_IMAGE_CSV = CSV_DIR / "image_statistic.csv"
DEFAULT_OUTPUT_CSV = CSV_DIR / "quality_metrics.csv"

LOGGER = logging.getLogger("gen_quality_metric")

OUTPUT_FIELDS = [
	"breed",
	"count",
	"avg_width",
	"avg_height",
	"avg_area",
	"avg_aspect_ratio",
	"avg_coverage",
	"area_cv",
	"aspect_cv",
	"pct_small",
	"pct_medium",
	"pct_large",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate per-breed quality metrics CSV from bbox/image analytics CSVs."
	)
	parser.add_argument(
		"--hbox-csv",
		type=Path,
		default=DEFAULT_HBOX_CSV,
		help="Path to bbox analytics CSV (default: csv/bbox_analytics.csv)",
	)
	parser.add_argument(
		"--image-csv",
		type=Path,
		default=None,
		help=(
			"Optional image statistics CSV for fallback image dimensions. "
			"Default: auto-pick csv/image_statistics.csv then csv/image_statistic.csv"
		),
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=DEFAULT_OUTPUT_CSV,
		help="Path to generated quality metrics CSV",
	)
	parser.add_argument(
		"--small-coverage-threshold",
		type=float,
		default=0.10,
		help="Coverage threshold for small boxes when size_category is missing",
	)
	parser.add_argument(
		"--medium-coverage-threshold",
		type=float,
		default=0.20,
		help="Coverage threshold for medium boxes when size_category is missing",
	)
	parser.add_argument(
		"--log-every",
		type=int,
		default=1000,
		help="Log progress every N rows (default: 1000)",
	)
	return parser.parse_args()


def choose_image_csv(arg_value: Optional[Path]) -> Optional[Path]:
	if arg_value is not None:
		return arg_value
	if DEFAULT_IMAGE_CSV.exists():
		return DEFAULT_IMAGE_CSV
	if LEGACY_IMAGE_CSV.exists():
		return LEGACY_IMAGE_CSV
	return None


def to_float(value: object) -> Optional[float]:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	try:
		return float(text)
	except ValueError:
		return None


def load_rows_by_image_id(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
	if path is None:
		return {}
	if not path.exists():
		raise FileNotFoundError(f"Missing image CSV file: {path}")

	data: Dict[str, Dict[str, str]] = {}
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		if not reader.fieldnames or "image_id" not in reader.fieldnames:
			raise ValueError(f"CSV must contain image_id column: {path}")
		for row in reader:
			image_id = (row.get("image_id") or "").strip()
			if image_id:
				data[image_id] = row
	return data


def mean(values: List[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def population_std(values: List[float]) -> float:
	if not values:
		return 0.0
	mu = mean(values)
	return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def safe_cv(values: List[float]) -> float:
	if not values:
		return 0.0
	mu = mean(values)
	if mu <= 0:
		return 0.0
	return population_std(values) / mu


def infer_size_category(coverage: float, small_thr: float, medium_thr: float) -> str:
	if coverage < small_thr:
		return "small"
	if coverage < medium_thr:
		return "medium"
	return "large"


def normalize_split_ratio(count: int, total: int) -> float:
	if total <= 0:
		return 0.0
	return count / float(total)


def build_row_data(
	hbox_row: Dict[str, str],
	image_row: Dict[str, str],
	small_thr: float,
	medium_thr: float,
) -> Optional[Dict[str, object]]:
	breed = (hbox_row.get("breed") or image_row.get("breed") or "").strip()
	if not breed:
		return None

	width = to_float(hbox_row.get("width"))
	height = to_float(hbox_row.get("height"))
	area = to_float(hbox_row.get("area"))

	if width is None or height is None or area is None:
		return None
	if width <= 0 or height <= 0:
		return None

	image_width = to_float(hbox_row.get("image_width"))
	image_height = to_float(hbox_row.get("image_height"))
	if image_width is None:
		image_width = to_float(image_row.get("width"))
	if image_height is None:
		image_height = to_float(image_row.get("height"))

	if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
		return None

	aspect_ratio = width / height
	coverage = area / (image_width * image_height)

	size_category = (hbox_row.get("size_category") or "").strip().lower()
	if size_category not in {"small", "medium", "large"}:
		size_category = infer_size_category(
			coverage=coverage,
			small_thr=small_thr,
			medium_thr=medium_thr,
		)

	return {
		"breed": breed,
		"width": width,
		"height": height,
		"area": area,
		"aspect_ratio": aspect_ratio,
		"coverage": coverage,
		"size_category": size_category,
	}


def main() -> None:
	args = parse_args()

	if args.small_coverage_threshold >= args.medium_coverage_threshold:
		raise ValueError(
			"--small-coverage-threshold must be lower than --medium-coverage-threshold"
		)

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
	)

	image_csv = choose_image_csv(args.image_csv)

	LOGGER.info("hbox csv      : %s", args.hbox_csv)
	LOGGER.info("image csv     : %s", image_csv if image_csv else "(not used)")
	LOGGER.info("output csv    : %s", args.output_csv)

	if not args.hbox_csv.exists():
		raise FileNotFoundError(f"Missing hbox CSV: {args.hbox_csv}")

	image_rows = load_rows_by_image_id(image_csv)

	grouped: Dict[str, Dict[str, object]] = {}
	skipped = 0

	with args.hbox_csv.open("r", encoding="utf-8", newline="") as in_f:
		reader = csv.DictReader(in_f)

		for idx, hbox_row in enumerate(reader, start=1):
			image_id = (hbox_row.get("image_id") or "").strip()
			image_row = image_rows.get(image_id, {})

			row = build_row_data(
				hbox_row=hbox_row,
				image_row=image_row,
				small_thr=args.small_coverage_threshold,
				medium_thr=args.medium_coverage_threshold,
			)

			if row is None:
				skipped += 1
				continue

			breed = str(row["breed"])
			if breed not in grouped:
				grouped[breed] = {
					"widths": [],
					"heights": [],
					"areas": [],
					"aspects": [],
					"coverages": [],
					"small": 0,
					"medium": 0,
					"large": 0,
				}

			bucket = grouped[breed]
			bucket["widths"].append(float(row["width"]))
			bucket["heights"].append(float(row["height"]))
			bucket["areas"].append(float(row["area"]))
			bucket["aspects"].append(float(row["aspect_ratio"]))
			bucket["coverages"].append(float(row["coverage"]))

			size_category = str(row["size_category"])
			if size_category in ("small", "medium", "large"):
				bucket[size_category] += 1

			if args.log_every > 0 and idx % args.log_every == 0:
				LOGGER.info(
					"Progress: %d rows processed | breeds=%d | skipped=%d",
					idx,
					len(grouped),
					skipped,
				)

	out_rows: List[Dict[str, object]] = []
	for breed in sorted(grouped.keys()):
		bucket = grouped[breed]

		widths = bucket["widths"]
		heights = bucket["heights"]
		areas = bucket["areas"]
		aspects = bucket["aspects"]
		coverages = bucket["coverages"]

		n = len(widths)
		if n == 0:
			continue

		out_rows.append(
			{
				"breed": breed,
				"count": n,
				"avg_width": mean(widths),
				"avg_height": mean(heights),
				"avg_area": mean(areas),
				"avg_aspect_ratio": mean(aspects),
				"avg_coverage": mean(coverages),
				"area_cv": safe_cv(areas),
				"aspect_cv": safe_cv(aspects),
				"pct_small": normalize_split_ratio(int(bucket["small"]), n),
				"pct_medium": normalize_split_ratio(int(bucket["medium"]), n),
				"pct_large": normalize_split_ratio(int(bucket["large"]), n),
			}
		)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	with args.output_csv.open("w", encoding="utf-8", newline="") as out_f:
		writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
		writer.writeheader()
		writer.writerows(out_rows)

	LOGGER.info(
		"Done. breeds=%d | skipped=%d | output=%s",
		len(out_rows),
		skipped,
		args.output_csv,
	)


if __name__ == "__main__":
	main()
