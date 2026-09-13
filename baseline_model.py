"""
baseline_model.py
==================
Non-deep baseline for DAGM 2007 binary defect classification.

Rationale
---------
Before reaching for a deep CNN (ResNet18, see model.py), academic and
engineering rigor calls for a simple, interpretable baseline to quantify
how much the deep model actually improves over classical hand-crafted
features + a linear classifier. This script:

    1. Extracts hand-crafted features from each image:
         - Downsampled grayscale pixel intensities (flattened, 32x32)
         - Global statistics: mean, std, skewness-proxy, min, max
         - A coarse 16-bin grayscale intensity histogram
    2. Trains a scikit-learn LogisticRegression classifier on those features.
    3. Reports Accuracy, Precision, Recall, and F1-Score on the held-out test set.

Usage
-----
    python baseline_model.py --data_root /path/to/DAGM2007
"""

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from dataset import build_dataframe, split_dataframe

FEATURE_IMG_SIZE = 32  # small downsample size for hand-crafted pixel features
HIST_BINS = 16


def extract_features(filepath: str) -> np.ndarray:
    """Extract a hand-crafted feature vector from a single grayscale image."""
    img = Image.open(filepath).convert("L").resize((FEATURE_IMG_SIZE, FEATURE_IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)

    flat = arr.flatten() / 255.0  # normalized downsampled pixels

    stats = np.array(
        [
            arr.mean() / 255.0,
            arr.std() / 255.0,
            arr.min() / 255.0,
            arr.max() / 255.0,
            np.median(arr) / 255.0,
        ]
    )

    hist, _ = np.histogram(arr, bins=HIST_BINS, range=(0, 255), density=True)

    return np.concatenate([flat, stats, hist]).astype(np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    features = [extract_features(fp) for fp in df["filepath"]]
    return np.vstack(features)


def train_and_evaluate_baseline(data_root: str, seed: int = 42) -> dict:
    df = build_dataframe(data_root)
    train_df, val_df, test_df = split_dataframe(df, seed=seed)

    # For the classical baseline we combine train+val for fitting since there
    # is no learning-rate/early-stopping loop that needs a validation split.
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Extracting features for {len(fit_df)} training images...")
    X_train = build_feature_matrix(fit_df)
    y_train = fit_df["label"].values

    print(f"Extracting features for {len(test_df)} test images...")
    X_test = build_feature_matrix(test_df)
    y_test = test_df["label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    print("\n=== Baseline (Logistic Regression on hand-crafted features) ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():>10}: {v:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Defect"], zero_division=0))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the non-deep baseline model.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to DAGM 2007 dataset root.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate_baseline(args.data_root, seed=args.seed)
