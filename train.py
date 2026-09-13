"""
train.py
========
Full training loop for the ResNet18 transfer-learning defect classifier.

Features
--------
- Adam optimizer + CrossEntropyLoss
- ReduceLROnPlateau LR scheduler (monitors validation loss)
- Early stopping (configurable patience) to prevent overfitting
- Saves the best model checkpoint (by validation loss) to `best_model.pth`
- Saves training/validation loss & accuracy curves to `outputs/train/`

Usage
-----
    python train.py --data_root /path/to/DAGM2007 --epochs 30 --batch_size 32
"""

import os
import argparse
import copy
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import build_model


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train() if train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def plot_curves(history: dict, output_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    axes[1].set_title("Accuracy Curves", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {out_path}")


def train_model(
    data_root: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 7,
    output_dir: str = "outputs/train",
    checkpoint_path: str = "best_model.pth",
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        data_root, batch_size=batch_size, seed=seed
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model = build_model(pretrained=True, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # Early stopping / checkpointing on validation loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_model_state, checkpoint_path)
            print(f"  -> New best model saved (val_loss={val_loss:.4f}) to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    plot_curves(history, output_dir)

    # Restore best weights before returning
    model.load_state_dict(best_model_state)

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model weights saved to: {checkpoint_path}")

    return model, history, (train_loader, val_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 defect classifier on DAGM 2007.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to DAGM 2007 dataset root.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs).")
    parser.add_argument("--output_dir", type=str, default="outputs/train")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
