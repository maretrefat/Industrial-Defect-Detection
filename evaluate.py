"""
evaluate.py
===========
Held-out test-set evaluation of the trained ResNet18 defect classifier.

Generates:
    1. Standard classification metrics (Accuracy, Precision, Recall, F1)
    2. confusion_matrix.png       - heatmap of the confusion matrix
    3. misclassified_samples.png  - grid of the top-K most-confidently-wrong
                                     predictions (Worst-performing samples),
                                     annotated with predicted vs. true label
                                     and the model's confidence score.

Usage
-----
    python evaluate.py --data_root /path/to/DAGM2007 --checkpoint_path best_model.pth
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from PIL import Image

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

LABEL_NAMES = {0: "Normal", 1: "Defect"}


def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor back to an HWC uint8 image for display."""
    img = tensor_img.clone().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img


def evaluate_model(
    data_root: str,
    checkpoint_path: str = "best_model.pth",
    batch_size: int = 32,
    output_dir: str = "outputs/evaluate",
    top_k_errors: int = 8,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, _, _, test_df = get_dataloaders(data_root, batch_size=batch_size, seed=seed)

    model = build_model(pretrained=False, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_labels, all_preds, all_confidences = [], [], []
    all_images, all_probs = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_images.extend([img.cpu() for img in images])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)

    # ---------------- Metrics ----------------
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
    }

    print("\n=== Test Set Evaluation (ResNet18) ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():>10}: {v:.4f}")
    print("\nDetailed classification report:")
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Defect"], zero_division=0)
    print(report)

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n" + report)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Defect"], yticklabels=["Normal", "Defect"],
        cbar=True,
    )
    plt.title("Confusion Matrix - Test Set (ResNet18)", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # ---------------- Error Analysis: Top-K Misclassified Images ----------------
    misclassified_idx = np.where(all_preds != all_labels)[0]

    if len(misclassified_idx) > 0:
        # Rank misclassified samples by confidence (most confidently WRONG first)
        misclassified_confidences = all_confidences[misclassified_idx]
        ranked = misclassified_idx[np.argsort(-misclassified_confidences)]
        top_errors = ranked[:top_k_errors]

        n_cols = min(4, len(top_errors))
        n_rows = int(np.ceil(len(top_errors) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)

        for i, idx in enumerate(top_errors):
            r, c = divmod(i, n_cols)
            ax = axes[r, c]
            img = denormalize_image(all_images[idx])
            ax.imshow(img)
            true_label = LABEL_NAMES[all_labels[idx]]
            pred_label = LABEL_NAMES[all_preds[idx]]
            conf = all_confidences[idx]
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.2f})",
                         fontsize=10, color="red", fontweight="bold")
            ax.axis("off")

        # Hide any unused subplots
        for j in range(len(top_errors), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r, c].axis("off")

        fig.suptitle("Top Misclassified Samples (Ranked by Model Confidence)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        err_path = os.path.join(output_dir, "misclassified_samples.png")
        plt.savefig(err_path, dpi=150)
        plt.close()
        print(f"Saved error analysis grid to: {err_path}")
    else:
        print("No misclassified samples found on the test set!")

    return metrics, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained ResNet18 defect classifier.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to DAGM 2007 dataset root.")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluate")
    parser.add_argument("--top_k_errors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_model(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        top_k_errors=args.top_k_errors,
        seed=args.seed,
    )
