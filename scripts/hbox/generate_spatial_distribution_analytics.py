import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spatial distribution analytics from bbox_analytics.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "bbox_analytics.csv",
        help="Input bbox analytics CSV (default: data/bbox_analytics.csv)",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("data") / "spatial_distribution_analysis.png",
        help="Path to save the 2x2 spatial analysis figure",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data") / "spatial_distribution_analytics.csv",
        help="Detailed per-image spatial metrics CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data") / "spatial_distribution_summary.csv",
        help="Summary CSV with split and quadrant distribution stats",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Max points in scatter plot (default: 5000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")

    spatial_df = pd.read_csv(args.input)

    required_cols = [
        "image_id",
        "breed",
        "split",
        "area",
        "image_width",
        "image_height",
    ]
    missing_cols = [c for c in required_cols if c not in spatial_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {args.input}: {missing_cols}")

    if "center_x" not in spatial_df.columns or "center_y" not in spatial_df.columns:
        need = ["xmin", "ymin", "width", "height"]
        miss_need = [c for c in need if c not in spatial_df.columns]
        if miss_need:
            raise ValueError(
                f"Need either center_x/center_y or {need}. Missing: {miss_need}"
            )
        spatial_df["center_x"] = (
            (spatial_df["xmin"] + spatial_df["width"] / 2)
            / spatial_df["image_width"].replace(0, np.nan)
        )
        spatial_df["center_y"] = (
            (spatial_df["ymin"] + spatial_df["height"] / 2)
            / spatial_df["image_height"].replace(0, np.nan)
        )

    spatial_df["normalized_area"] = (
        spatial_df["area"]
        / (spatial_df["image_width"] * spatial_df["image_height"]).replace(0, np.nan)
    )

    spatial_df = spatial_df.dropna(
        subset=["center_x", "center_y", "normalized_area", "split"]
    )
    spatial_df = spatial_df[
        spatial_df["center_x"].between(0, 1)
        & spatial_df["center_y"].between(0, 1)
    ]

    print("=" * 70)
    print("Spatial Distribution Analysis")
    print("=" * 70)
    print(f"Rows used: {len(spatial_df):,}")

    # Build/normalize bbox edge columns for CSV export.
    if "xmax" not in spatial_df.columns and "xmin" in spatial_df.columns and "width" in spatial_df.columns:
        spatial_df["xmax"] = spatial_df["xmin"] + spatial_df["width"]
    if "ymax" not in spatial_df.columns and "ymin" in spatial_df.columns and "height" in spatial_df.columns:
        spatial_df["ymax"] = spatial_df["ymin"] + spatial_df["height"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Bounding Box Spatial Distribution (Bird Dataset)",
        fontsize=16,
        fontweight="bold",
    )

    hb = axes[0, 0].hexbin(
        spatial_df["center_x"],
        spatial_df["center_y"],
        gridsize=35,
        cmap="viridis",
        mincnt=1,
    )
    axes[0, 0].set_title("Center Density (Hexbin)")
    axes[0, 0].set_xlabel("center_x (normalized)")
    axes[0, 0].set_ylabel("center_y (normalized)")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    cb = fig.colorbar(hb, ax=axes[0, 0])
    cb.set_label("Count")

    axes[0, 1].hist(
        spatial_df["center_x"],
        bins=40,
        alpha=0.7,
        color="#3b82f6",
        label="center_x",
    )
    axes[0, 1].hist(
        spatial_df["center_y"],
        bins=40,
        alpha=0.55,
        color="#10b981",
        label="center_y",
    )
    axes[0, 1].set_title("Center Coordinate Distributions")
    axes[0, 1].set_xlabel("Normalized coordinate value")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    sample_n = min(max(args.sample_size, 1), len(spatial_df))
    sample_df = (
        spatial_df.sample(sample_n, random_state=42)
        if len(spatial_df) > sample_n
        else spatial_df
    )

    sc = axes[1, 0].scatter(
        sample_df["center_x"],
        sample_df["center_y"],
        c=sample_df["normalized_area"],
        cmap="plasma",
        s=12,
        alpha=0.65,
        edgecolors="none",
    )
    axes[1, 0].set_title("Center Scatter Colored by Normalized Area")
    axes[1, 0].set_xlabel("center_x")
    axes[1, 0].set_ylabel("center_y")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    cb2 = fig.colorbar(sc, ax=axes[1, 0])
    cb2.set_label("normalized_area")

    spatial_df["x_side"] = np.where(spatial_df["center_x"] < 0.5, "left", "right")
    spatial_df["y_side"] = np.where(spatial_df["center_y"] < 0.5, "top", "bottom")
    spatial_df["quadrant"] = spatial_df["y_side"] + "-" + spatial_df["x_side"]

    quad_order = ["top-left", "top-right", "bottom-left", "bottom-right"]
    quad_counts = spatial_df["quadrant"].value_counts().reindex(quad_order, fill_value=0)
    quad_ratio = quad_counts / len(spatial_df)

    axes[1, 1].bar(
        quad_ratio.index,
        quad_ratio.values,
        color=["#f59e0b", "#10b981", "#3b82f6", "#ef4444"],
    )
    axes[1, 1].set_title("Quadrant Occupancy Ratio")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].set_ylim(0, max(0.35, quad_ratio.max() * 1.2))
    axes[1, 1].grid(axis="y", alpha=0.25)
    for i, val in enumerate(quad_ratio.values):
        axes[1, 1].text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=150, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("\nSpatial summary")
    print("-" * 70)
    print(
        f"center_x mean/median: {spatial_df['center_x'].mean():.3f} / "
        f"{spatial_df['center_x'].median():.3f}"
    )
    print(
        f"center_y mean/median: {spatial_df['center_y'].mean():.3f} / "
        f"{spatial_df['center_y'].median():.3f}"
    )
    print(
        f"normalized_area mean/median: {spatial_df['normalized_area'].mean():.3f} / "
        f"{spatial_df['normalized_area'].median():.3f}"
    )
    print("Quadrant ratios:")
    for q in quad_order:
        print(f"  {q}: {quad_ratio[q]:.3f}")

    print("\nBy split (center and normalized area):")
    for split_name, block in spatial_df.groupby("split"):
        print(
            f"  {split_name}: n={len(block):,}, center_x_mean={block['center_x'].mean():.3f}, "
            f"center_y_mean={block['center_y'].mean():.3f}, "
            f"norm_area_mean={block['normalized_area'].mean():.3f}"
        )

    # Export detailed CSV (no species column; birds-only dataset).
    export_cols = [
        "image_id",
        "breed",
        "split",
        "center_x",
        "center_y",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "normalized_area",
    ]
    present_export_cols = [c for c in export_cols if c in spatial_df.columns]
    detailed_df = spatial_df[present_export_cols].copy()

    # Summary CSV: global + split + quadrant ratios.
    summary_rows = [
        {
            "group": "all",
            "label": "all",
            "count": int(len(spatial_df)),
            "center_x_mean": float(spatial_df["center_x"].mean()),
            "center_y_mean": float(spatial_df["center_y"].mean()),
            "normalized_area_mean": float(spatial_df["normalized_area"].mean()),
        }
    ]
    for split_name, block in spatial_df.groupby("split"):
        summary_rows.append(
            {
                "group": "split",
                "label": split_name,
                "count": int(len(block)),
                "center_x_mean": float(block["center_x"].mean()),
                "center_y_mean": float(block["center_y"].mean()),
                "normalized_area_mean": float(block["normalized_area"].mean()),
            }
        )
    for q in quad_order:
        summary_rows.append(
            {
                "group": "quadrant",
                "label": q,
                "count": int(quad_counts[q]),
                "center_x_mean": np.nan,
                "center_y_mean": np.nan,
                "normalized_area_mean": float(quad_ratio[q]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"\nSaved figure: {args.output_figure}")
    print(f"Saved detailed CSV: {args.output_csv}")
    print(f"Saved summary CSV: {args.summary_csv}")
    print("=" * 70)
    print("Spatial Distribution Analysis complete")


if __name__ == "__main__":
    main()
