import os
import glob
import random
import zipfile
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

# ----------------------------------------------------------------------------
# Mount Google Drive and Extract Dataset
# ----------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    pass

zip_path = '/content/drive/MyDrive/Project/Deep Learning/deep/archive.zip'
extract_path = '/content/dataset'

if os.path.exists(zip_path):
    if not os.path.exists(extract_path) or len(os.listdir(extract_path)) == 0:
        print("Extracting archive.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extraction completed successfully.")
    else:
        print("Dataset is already extracted at the destination folder.")
else:
    print("Warning: archive.zip was not found at the specified path.")


# ----------------------------------------------------------------------------
# Configurations & Constants
# ----------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG")


# ----------------------------------------------------------------------------
# Dataset Directory Parsing
# ----------------------------------------------------------------------------
def _find_labels_txt(label_dir: str) -> Optional[str]:
    """Find the Labels.txt file inside a given folder."""
    candidates = glob.glob(os.path.join(label_dir, "*abels*.txt"))
    return candidates[0] if candidates else None


def _parse_folder(split_dir: str, split_name: str) -> list:
    """Read image paths and corresponding labels from a dataset folder."""
    rows = []
    if not os.path.isdir(split_dir):
        return rows

    label_dir = os.path.join(split_dir, "Label")
    labels_txt = _find_labels_txt(label_dir) if os.path.isdir(label_dir) else None

    label_map = {}
    if labels_txt is not None:
        with open(labels_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_id, label = parts[0], parts[1]
                try:
                    label_map[img_id] = int(label)
                except ValueError:
                    continue

    for img_path in glob.glob(os.path.join(split_dir, "*")):
        if not img_path.lower().endswith(VALID_EXTS):
            continue
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        label = label_map.get(img_id, label_map.get(img_id.lstrip("0") or "0", 0))
        rows.append({"filepath": img_path, "label": int(label), "split": split_name.lower()})

    return rows


def build_full_dataframe(extract_root: str) -> pd.DataFrame:
    """Collect all images from Train and Test subdirectories into one unified DataFrame."""
    target_dir = None
    for root, dirs, _ in os.walk(extract_root):
        if any(d.lower() == "train" for d in dirs) and any(d.lower() == "test" for d in dirs):
            target_dir = root
            break

    if target_dir is None:
        target_dir = extract_root

    train_dir = os.path.join(target_dir, "Train")
    test_dir = os.path.join(target_dir, "Test")

    if not os.path.exists(train_dir):
        for sub in os.listdir(target_dir):
            if sub.lower() == "train":
                train_dir = os.path.join(target_dir, sub)
            elif sub.lower() == "test":
                test_dir = os.path.join(target_dir, sub)

    all_rows = _parse_folder(train_dir, "Train") + _parse_folder(test_dir, "Test")

    if not all_rows:
        raise FileNotFoundError(f"Could not load images from extracted directory '{extract_root}'.")

    full_df = pd.DataFrame(all_rows).drop_duplicates(subset="filepath").reset_index(drop=True)
    return full_df


# ----------------------------------------------------------------------------
# Custom Stratified Split (70% Train / 15% Val / 15% Test)
# ----------------------------------------------------------------------------
def split_all_data(
    full_df: pd.DataFrame, 
    train_frac: float = 0.70, 
    val_frac: float = 0.15, 
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a proper 3-way stratified split on the full dataset."""
    test_frac = 1.0 - train_frac - val_frac

    train_df, temp_df = train_test_split(
        full_df, 
        test_size=(val_frac + test_frac), 
        stratify=full_df["label"], 
        random_state=seed
    )

    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1.0 - relative_val_frac), 
        stratify=temp_df["label"], 
        random_state=seed
    )

    return (
        train_df.reset_index(drop=True), 
        val_df.reset_index(drop=True), 
        test_df.reset_index(drop=True)
    )


def get_transforms(train: bool = True) -> T.Compose:
    """Define standard image preprocessing and augmentations."""
    if train:
        return T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ----------------------------------------------------------------------------
# Dataset Definition
# ----------------------------------------------------------------------------
class DAGMDataset(Dataset):
    """Custom PyTorch Dataset for loading images and binary labels."""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[T.Compose] = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


# ----------------------------------------------------------------------------
# DataLoaders Constructor
# ----------------------------------------------------------------------------
def get_dataloaders(
    extract_path: str = extract_path,
    batch_size: int = 32,
    num_workers: int = 2,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
):
    """Build and return properly proportioned train, validation, and test PyTorch DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_df = build_full_dataframe(extract_path)
    train_df, val_df, test_df = split_all_data(full_df, train_frac=train_frac, val_frac=val_frac, seed=seed)

    train_ds = DAGMDataset(train_df, transform=get_transforms(train=True))
    val_ds = DAGMDataset(val_df, transform=get_transforms(train=False))
    test_ds = DAGMDataset(test_df, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders()

    print("\nDataset ready for training:")
    print(f"- Training set   : {len(train_df)} images (Defects found: {train_df['label'].sum()})")
    print(f"- Validation set : {len(val_df)} images (Defects found: {val_df['label'].sum()})")
    print(f"- Testing set    : {len(test_df)} images (Defects found: {test_df['label'].sum()})")
