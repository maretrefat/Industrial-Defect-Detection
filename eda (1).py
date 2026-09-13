"""
eda.py
======
Exploratory Data Analysis for the DAGM 2007 dataset.

Generates and saves three figures to the `outputs/eda/` directory:
    1. class_distribution.png  - bar plot of Normal vs Defect image counts
    2. samples_grid.png        - grid of sample Normal and Defective images
    3. pixel_distribution.png  - grayscale pixel intensity histogram, split
                                  by class, to reveal texture/illumination
                                  differences between Normal and Defective surfaces.

Usage
-----
    python eda.py --data_root /path/to/DAGM2007 --output_dir outputs/eda
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from dataset import build_dataframe

sns.set_theme(style="whitegrid")
LABEL_NAMES = {0: "Normal", 1: "Defect"}


def plot_class_distribution(df: pd.DataFrame, output_dir: str) -> None:
    counts = df["label"].map(LABEL_NAMES).value_counts().reindex(["Normal", "Defect"])

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                      palette={"Normal": "#4C72B0", "Defect": "#C44E52"}, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center", fontweight="bold")
    ax.set_title("Class Distribution: Normal vs. Defect", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_samples_grid(df: pd.DataFrame, output_dir: str, n_per_class: int = 4) -> None:
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6.5))

    for row_idx, label in enumerate([0, 1]):
        subset = df[df["label"] == label].sample(
            n=min(n_per_class, (df["label"] == label).sum()), random_state=42
        )
        for col_idx, (_, sample_row) in enumerate(subset.iterrows()):
            img = Image.open(sample_row["filepath"]).convert("RGB")
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(LABEL_NAMES[label], fontsize=12)
        axes[row_idx, 0].text(
            -0.15, 0.5, LABEL_NAMES[label], transform=axes[row_idx, 0].transAxes,
            fontsize=13, fontweight="bold", va="center", ha="right", rotation=90,
        )

    fig.suptitle("Sample Images: Normal (top) vs. Defective (bottom)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "samples_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pixel_distribution(df: pd.DataFrame, output_dir: str, sample_size: int = 200) -> None:
    rng = np.random.default_rng(42)
    intensities = {0: [], 1: []}

    for label in (0, 1):
        subset = df[df["label"] == label]
        n = min(sample_size, len(subset))
        if n == 0:
            continue
        sampled_paths = subset["filepath"].sample(n=n, random_state=42).tolist()
        for path in sampled_paths:
            img = Image.open(path).convert("L")
            arr = np.asarray(img, dtype=np.float32).flatten()
            # subsample pixels per image to keep the histogram computation light
            if arr.size > 2000:
                idx = rng.choice(arr.size, size=2000, replace=False)
                arr = arr[idx]
            intensities[label].append(arr)

    plt.figure(figsize=(7, 5))
    colors = {0: "#4C72B0", 1: "#C44E52"}
    for label in (0, 1):
        if not intensities[label]:
            continue
        all_pixels = np.concatenate(intensities[label])
        sns.histplot(all_pixels, bins=50, stat="density", color=colors[label],
                     label=LABEL_NAMES[label], alpha=0.5, kde=True)

    plt.title("Grayscale Pixel Intensity Distribution by Class", fontsize=14, fontweight="bold")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "pixel_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def run_eda(data_root: str, output_dir: str = "outputs/eda") -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = build_dataframe(data_root)

    print(f"Loaded {len(df)} images.")
    print(df["label"].map(LABEL_NAMES).value_counts())

    plot_class_distribution(df, output_dir)
    plot_samples_grid(df, output_dir)
    plot_pixel_distribution(df, output_dir)

    # Save a small summary CSV alongside the figures for reference in the report
    summary = df["label"].map(LABEL_NAMES).value_counts().rename_axis("class").reset_index(name="count")
    summary.to_csv(os.path.join(output_dir, "class_distribution_summary.csv"), index=False)
    print(f"EDA complete. Figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on the DAGM 2007 dataset.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to DAGM 2007 dataset root.")
    parser.add_argument("--output_dir", type=str, default="outputs/eda", help="Directory to save EDA plots.")
    args = parser.parse_args()

    run_eda(args.data_root, args.output_dir)
