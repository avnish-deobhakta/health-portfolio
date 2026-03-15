"""
Dataset classes for diabetic retinopathy screening experiments.

Supports both binary (referable vs non-referable) and 5-class (grade 0-4)
formulations, as well as adversarial training with primary model predictions.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def get_transforms(split="val", image_size=224):
    """Return transforms for training or validation/test."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class DRDatasetBinary(Dataset):
    """Binary classification: referable (grade >= 2) vs non-referable."""

    def __init__(self, csv_path, image_dir, transform, id_col="image_id"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.id_col = id_col
        if "binary_label" not in self.df.columns:
            self.df["binary_label"] = (self.df["level"] >= 2).astype(int)
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir) if f.endswith(".png")
        }
        self.df = self.df[self.df[id_col].isin(existing)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, f"{row[self.id_col]}.png")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return img, label, str(row[self.id_col])


class DRDataset5Class(Dataset):
    """5-class severity grading: grades 0 through 4."""

    def __init__(self, csv_path, image_dir, transform, id_col="image_id"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.id_col = id_col
        if "binary_label" not in self.df.columns:
            self.df["binary_label"] = (self.df["level"] >= 2).astype(int)
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir) if f.endswith(".png")
        }
        self.df = self.df[self.df[id_col].isin(existing)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, f"{row[self.id_col]}.png")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        grade = torch.tensor(row["level"], dtype=torch.long)
        binary = torch.tensor(row["binary_label"], dtype=torch.float32)
        return img, grade, str(row[self.id_col]), binary


class DRDatasetAdversarial(Dataset):
    """Binary classification with cached primary model predictions for
    adversarial decorrelation training."""

    def __init__(self, csv_path, image_dir, transform, primary_probs, id_col="image_id"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.primary_probs = primary_probs
        self.id_col = id_col
        if "binary_label" not in self.df.columns:
            self.df["binary_label"] = (self.df["level"] >= 2).astype(int)
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir) if f.endswith(".png")
        }
        self.df = self.df[self.df[id_col].isin(existing)].reset_index(drop=True)
        self.df = self.df[self.df[id_col].isin(primary_probs.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, f"{row[self.id_col]}.png")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        primary_p = torch.tensor(self.primary_probs[row[self.id_col]], dtype=torch.float32)
        return img, label, str(row[self.id_col]), primary_p
