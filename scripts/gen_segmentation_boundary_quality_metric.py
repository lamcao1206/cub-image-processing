"""
Generate per-breed segmentation boundary quality metrics.

Default input:
- csv/mask_statistics.csv

Default output:
- csv/segmentation_boundary_quality_metric.csv

Output columns:
breed,count,avg_coverage,coverage_cv,avg_fg_pct,avg_boundary_pct,avg_bg_pct,fg_cv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

DEFAULT_INPUT_CSV = CSV_DIR / "mask_statistics.csv"
DEFAULT_OUTPUT_CSV = CSV_DIR / "segmentation_boundary_quality_metric.csv"

LOGGER = logging.getLogger("gen_segmentation_boundary_quality_metric")

OUTPUT_FIELDS = [
    "breed",
    "count",
    "avg_coverage",
    "coverage_cv",
    "avg_fg_pct",
    "avg_boundary_pct",
    "avg_bg_pct",
    "fg_cv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate segmentation boundary quality metrics per breed."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input mask statistics CSV path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output quality metric CSV path",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    return float(value.strip())


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pop_std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def cv(values: List[float]) -> float:
    m = mean(values)
    if m <= 0:
        return 0.0
    return pop_std(values) / m


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {args.input_csv}. Please run scripts/gen_mask_statistic.py first."
        )

    by_breed: Dict[str, Dict[str, List[float]]] = {}

    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"breed", "mask_coverage", "fg_percentage", "boundary_percentage", "bg_percentage"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns in {args.input_csv}: {sorted(missing)}")

        for row in reader:
            breed = (row.get("breed") or "").strip()
            if not breed:
                continue

            bucket = by_breed.setdefault(
                breed,
                {
                    "coverage": [],
                    "fg_pct": [],
                    "boundary_pct": [],
                    "bg_pct": [],
                },
            )

            bucket["coverage"].append(to_float(row["mask_coverage"]))
            bucket["fg_pct"].append(to_float(row["fg_percentage"]))
            bucket["boundary_pct"].append(to_float(row["boundary_percentage"]))
            bucket["bg_pct"].append(to_float(row["bg_percentage"]))

    out_rows = []
    for breed in sorted(by_breed.keys()):
        stats = by_breed[breed]
        coverage_vals = stats["coverage"]
        fg_vals = stats["fg_pct"]
        boundary_vals = stats["boundary_pct"]
        bg_vals = stats["bg_pct"]

        out_rows.append(
            {
                "breed": breed,
                "count": len(coverage_vals),
                "avg_coverage": mean(coverage_vals),
                "coverage_cv": cv(coverage_vals),
                "avg_fg_pct": mean(fg_vals),
                "avg_boundary_pct": mean(boundary_vals),
                "avg_bg_pct": mean(bg_vals),
                "fg_cv": cv(fg_vals),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    LOGGER.info("Done. breeds=%d | output=%s", len(out_rows), args.output_csv)


if __name__ == "__main__":
    main()
