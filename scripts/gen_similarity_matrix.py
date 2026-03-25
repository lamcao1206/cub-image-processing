"""
Generate breed similarity matrix from embedding CSV.

Default input expects columns:
    image_name, breed, tsne_x, tsne_y

Output:
    csv/similarity_matrix.csv
Matrix is square with breed labels on both rows and columns, ordered by CUB classes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_EMBEDDINGS_CSV = PROJECT_ROOT / "csv" / "tsne_embeddings.csv"
DEFAULT_CLASSES_TXT = PROJECT_ROOT / "data" / "CUB_200_2011" / "classes.txt"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "csv" / "similarity_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ordered breed similarity matrix from embeddings."
    )
    parser.add_argument("--embeddings-csv", type=Path, default=DEFAULT_EMBEDDINGS_CSV)
    parser.add_argument("--classes-txt", type=Path, default=DEFAULT_CLASSES_TXT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def normalize_breed_name(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"^\d+[._\-\s]*", "", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.lower()


def load_ordered_breeds(classes_txt: Path) -> list[str]:
    if not classes_txt.exists():
        raise FileNotFoundError(f"Missing file: {classes_txt}")

    ordered: list[str] = []
    with classes_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            breed_name = parts[1] if len(parts) > 1 else parts[0]
            ordered.append(normalize_breed_name(breed_name))

    # Keep insertion order while removing duplicates.
    return list(dict.fromkeys(ordered))


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["tsne_x", "tsne_y"]
    if set(preferred).issubset(df.columns):
        return preferred

    ignore = {"image_name", "image_id", "breed", "class", "species", "split"}
    numeric_cols = [
        c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found to compute similarity.")
    return numeric_cols


def main() -> None:
    args = parse_args()

    if not args.embeddings_csv.exists():
        raise FileNotFoundError(f"Missing file: {args.embeddings_csv}")

    df = pd.read_csv(args.embeddings_csv)
    if "breed" not in df.columns:
        raise ValueError("Input embeddings CSV must contain a 'breed' column.")

    feature_cols = choose_feature_columns(df)

    work_df = df[["breed", *feature_cols]].dropna().copy()
    work_df["breed_norm"] = work_df["breed"].map(normalize_breed_name)

    breed_centroids = work_df.groupby("breed_norm")[feature_cols].mean()
    sim = cosine_similarity(breed_centroids.values)
    sim_df = pd.DataFrame(sim, index=breed_centroids.index, columns=breed_centroids.index)

    ordered_from_classes = load_ordered_breeds(args.classes_txt)
    existing = set(sim_df.index)

    ordered_present = [b for b in ordered_from_classes if b in existing]
    leftovers = sorted(existing - set(ordered_present))
    final_order = ordered_present + leftovers

    sim_df = sim_df.loc[final_order, final_order]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sim_df.index.name = "breed"
    sim_df.to_csv(args.output_csv, index=True)

    print(f"Saved: {args.output_csv}")
    print(f"Breeds: {sim_df.shape[0]}")
    print(f"Features used: {feature_cols}")


if __name__ == "__main__":
    main()
