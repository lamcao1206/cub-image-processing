import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def safe_cv(values: pd.Series) -> float:
    mean = float(values.mean())
    if mean <= 0.0:
        return float("nan")
    return float(values.std(ddof=0) / mean)


def compute_size_percentages(group: pd.DataFrame) -> Dict[str, float]:
    n = len(group)
    if n == 0:
        return {"pct_small": 0.0, "pct_medium": 0.0, "pct_large": 0.0}

    counts = group["size_category"].value_counts(dropna=False)
    return {
        "pct_small": float(counts.get("small", 0) / n),
        "pct_medium": float(counts.get("medium", 0) / n),
        "pct_large": float(counts.get("large", 0) / n),
    }


def iqr_flags(values: pd.Series) -> pd.Series:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=values.index)

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (values < low) | (values > high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-breed annotation quality metrics from bbox analytics CSV "
            "(size consistency, coverage, and outlier stats)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "bbox_analytics.csv",
        help="Input bbox analytics CSV (default: data/bbox_analytics.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "annotation_quality_by_breed.csv",
        help="Output per-breed quality CSV (default: data/annotation_quality_by_breed.csv)",
    )
    parser.add_argument(
        "--outliers-output",
        type=Path,
        default=Path("data") / "annotation_outliers.csv",
        help="Output image-level outlier CSV (default: data/annotation_outliers.csv)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum samples per breed to include in summary (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    required_cols = [
        "image_id",
        "breed",
        "split",
        "width",
        "height",
        "area",
        "size_category",
        "image_width",
        "image_height",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Derived fields for quality analysis.
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["coverage"] = df["area"] / (df["image_width"] * df["image_height"]).replace(0, np.nan)

    df = df.dropna(subset=["breed", "width", "height", "area", "aspect_ratio", "coverage"])

    outlier_parts = []
    summary_rows = []

    for breed, group in df.groupby("breed", sort=True):
        if len(group) < args.min_count:
            continue

        area_out = iqr_flags(group["area"])
        aspect_out = iqr_flags(group["aspect_ratio"])
        coverage_out = iqr_flags(group["coverage"])
        any_out = area_out | aspect_out | coverage_out

        out_block = group.loc[any_out].copy()
        if not out_block.empty:
            out_block["is_area_outlier"] = area_out[any_out].values
            out_block["is_aspect_outlier"] = aspect_out[any_out].values
            out_block["is_coverage_outlier"] = coverage_out[any_out].values
            out_block["is_any_outlier"] = True
            outlier_parts.append(out_block)

        size_pct = compute_size_percentages(group)

        summary_rows.append(
            {
                "breed": breed,
                "count": int(len(group)),
                "avg_width": float(group["width"].mean()),
                "avg_height": float(group["height"].mean()),
                "avg_area": float(group["area"].mean()),
                "avg_aspect_ratio": float(group["aspect_ratio"].mean()),
                "avg_coverage": float(group["coverage"].mean()),
                "area_cv": safe_cv(group["area"]),
                "aspect_cv": safe_cv(group["aspect_ratio"]),
                "coverage_cv": safe_cv(group["coverage"]),
                "pct_small": size_pct["pct_small"],
                "pct_medium": size_pct["pct_medium"],
                "pct_large": size_pct["pct_large"],
                "outlier_count": int(any_out.sum()),
                "pct_outliers": float(any_out.mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No rows available after filtering. Check input data or --min-count.")

    ordered_cols = [
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
        "coverage_cv",
        "outlier_count",
        "pct_outliers",
    ]
    summary_df = summary_df[ordered_cols].sort_values("breed").reset_index(drop=True)

    if outlier_parts:
        outlier_df = pd.concat(outlier_parts, ignore_index=True)
    else:
        outlier_df = pd.DataFrame(
            columns=[
                "image_id",
                "breed",
                "split",
                "width",
                "height",
                "area",
                "aspect_ratio",
                "coverage",
                "size_category",
                "is_area_outlier",
                "is_aspect_outlier",
                "is_coverage_outlier",
                "is_any_outlier",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.outliers_output.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(args.output, index=False)
    outlier_df.to_csv(args.outliers_output, index=False)

    print("Annotation quality analysis complete")
    print(f"- input rows: {len(df):,}")
    print(f"- breeds summarized: {len(summary_df):,}")
    print(f"- summary csv: {args.output}")
    print(f"- outlier csv: {args.outliers_output}")


if __name__ == "__main__":
    main()
