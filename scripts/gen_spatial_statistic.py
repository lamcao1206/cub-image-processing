"""
Generate spatial-distribution CSV from hbox and image CSV data.

Input defaults:
- csv/bbox_analytics.csv  (hbox data)
- csv/image_statistics.csv (image data; fallback: image_statistic.csv)

Output:
- csv/spatial_distribution_analytics.csv

Output columns are aligned with the sample format:
image_id, breed, species, split, center_x, center_y,
xmin, ymin, xmax, ymax, normalized_area
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / "csv"

DEFAULT_HBOX_CSV = CSV_DIR / "bbox_analytics.csv"
DEFAULT_IMAGE_CSV = CSV_DIR / "image_statistics.csv"
LEGACY_IMAGE_CSV = CSV_DIR / "image_statistic.csv"
DEFAULT_OUTPUT_CSV = CSV_DIR / "spatial_distribution_analytics.csv"

LOGGER = logging.getLogger("gen_spatial_statistic")

OUTPUT_FIELDS = [
    "image_id",
    "breed",
    "species",
    "split",
    "center_x",
    "center_y",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "normalized_area",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spatial CSV by combining hbox and image CSV data."
    )
    parser.add_argument(
        "--hbox-csv",
        type=Path,
        default=DEFAULT_HBOX_CSV,
        help="Path to hbox/bbox analytics CSV (default: csv/bbox_analytics.csv)",
    )
    parser.add_argument(
        "--image-csv",
        type=Path,
        default=None,
        help=(
            "Path to image statistics CSV. "
            "Default: auto-pick csv/image_statistics.csv then csv/image_statistic.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to generated spatial CSV",
    )
    parser.add_argument(
        "--default-species",
        type=str,
        default="bird",
        help="Fallback species value when image CSV does not have a species column",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N rows (default: 500)",
    )
    return parser.parse_args()


def choose_image_csv(arg_value: Optional[Path]) -> Path:
    if arg_value is not None:
        return arg_value
    if DEFAULT_IMAGE_CSV.exists():
        return DEFAULT_IMAGE_CSV
    return LEGACY_IMAGE_CSV


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


def normalize_num(value: float) -> float | int:
    # Keep bbox coordinates compact (int when whole number).
    if float(value).is_integer():
        return int(value)
    return round(value, 6)


def load_rows_by_image_id(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

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


def pick_species(image_row: Dict[str, str], default_species: str) -> str:
    for key in ("species", "Species", "SPECIES"):
        value = (image_row.get(key) or "").strip()
        if value:
            return value
    return default_species


def build_spatial_row(
    hbox_row: Dict[str, str],
    image_row: Dict[str, str],
    default_species: str,
) -> Optional[Dict[str, object]]:
    image_id = (hbox_row.get("image_id") or "").strip()
    if not image_id:
        return None

    breed = (hbox_row.get("breed") or image_row.get("breed") or "unknown").strip()
    split = (hbox_row.get("split") or image_row.get("split") or "unknown").strip()
    species = pick_species(image_row=image_row, default_species=default_species)

    xmin = to_float(hbox_row.get("xmin"))
    ymin = to_float(hbox_row.get("ymin"))

    xmax = to_float(hbox_row.get("xmax"))
    ymax = to_float(hbox_row.get("ymax"))

    width = to_float(hbox_row.get("width"))
    height = to_float(hbox_row.get("height"))

    if xmax is None and xmin is not None and width is not None:
        xmax = xmin + width
    if ymax is None and ymin is not None and height is not None:
        ymax = ymin + height

    if None in (xmin, ymin, xmax, ymax):
        return None

    image_width = to_float(hbox_row.get("image_width"))
    image_height = to_float(hbox_row.get("image_height"))

    if image_width is None:
        image_width = to_float(image_row.get("width"))
    if image_height is None:
        image_height = to_float(image_row.get("height"))

    center_x = to_float(hbox_row.get("center_x"))
    center_y = to_float(hbox_row.get("center_y"))

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    area = to_float(hbox_row.get("area"))
    if area is None:
        area = bbox_width * bbox_height

    if center_x is None:
        if image_width is None or image_width <= 0:
            return None
        center_x = (xmin + bbox_width / 2.0) / image_width

    if center_y is None:
        if image_height is None or image_height <= 0:
            return None
        center_y = (ymin + bbox_height / 2.0) / image_height

    normalized_area = to_float(hbox_row.get("normalized_area"))
    if normalized_area is None:
        if image_width is None or image_height is None:
            return None
        if image_width <= 0 or image_height <= 0:
            return None
        normalized_area = area / (image_width * image_height)

    return {
        "image_id": image_id,
        "breed": breed,
        "species": species,
        "split": split,
        "center_x": round(center_x, 6),
        "center_y": round(center_y, 6),
        "xmin": normalize_num(xmin),
        "ymin": normalize_num(ymin),
        "xmax": normalize_num(xmax),
        "ymax": normalize_num(ymax),
        "normalized_area": normalized_area,
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    image_csv = choose_image_csv(args.image_csv)

    LOGGER.info("hbox csv      : %s", args.hbox_csv)
    LOGGER.info("image csv     : %s", image_csv)
    LOGGER.info("output csv    : %s", args.output_csv)

    image_rows = load_rows_by_image_id(image_csv)

    if not args.hbox_csv.exists():
        raise FileNotFoundError(f"Missing hbox CSV: {args.hbox_csv}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with args.hbox_csv.open("r", encoding="utf-8", newline="") as in_f, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for idx, hbox_row in enumerate(reader, start=1):
            image_id = (hbox_row.get("image_id") or "").strip()
            if not image_id:
                skipped += 1
                continue

            image_row = image_rows.get(image_id, {})

            row = build_spatial_row(
                hbox_row=hbox_row,
                image_row=image_row,
                default_species=args.default_species,
            )

            if row is None:
                skipped += 1
                continue

            writer.writerow(row)
            written += 1

            if args.log_every > 0 and idx % args.log_every == 0:
                LOGGER.info(
                    "Progress: %d rows processed | written=%d | skipped=%d",
                    idx,
                    written,
                    skipped,
                )

    LOGGER.info(
        "Done. rows written=%d | skipped=%d | output=%s",
        written,
        skipped,
        args.output_csv,
    )


if __name__ == "__main__":
    main()
