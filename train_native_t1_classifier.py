#!/usr/bin/env python3
"""
train_native_t1_classifier.py
==============================
Trains 3D DenseNet-121 on ALL available native T1 whole-brain scans:
- Glioma (MSD Task01 T1w, n=484)
- Meningioma (BraTS-MEN t1n, n=200)
- Metastasis (Brain-Mets t1n, n=200)
Total = 884 subjects, partitioned 70/15/15 by patient ID.
Uses cached preprocessed volumes for maximum training throughput.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from monai.networks.nets import DenseNet121
from monai.losses import FocalLoss
import monai.transforms as mt

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
CACHE_DIR = BASE_DIR / "cache_preprocessed"


class CachedMRIDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], is_train: bool = True):
        self.items = items
        self.is_train = is_train

        if is_train:
            self.aug = mt.Compose([
                mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0), # LR flip only
                mt.RandAffined(
                    keys=["image"],
                    prob=0.3,
                    rotate_range=(0.05, 0.05, 0.05),
                    scale_range=(0.03, 0.03, 0.03),
                    mode="bilinear",
                    padding_mode="zeros"
                )
            ])
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.items[idx]
        npy_path = rec["cache_file"]
        
        # Load cached float16 and convert to float32 tensor
        vol = np.load(npy_path).astype(np.float32)
        # Add channel dimension: (1, 128, 128, 128)
        data = {"image": torch.from_numpy(vol).unsqueeze(0)}
        
        if self.is_train and self.aug is not None:
            data = self.aug(data)

        data["label"] = torch.tensor(rec["class_id"], dtype=torch.long)
        data["subject_id"] = rec["subject_id"]
        return data


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, present_classes: List[int]) -> float:
    recalls = []
    for c in present_classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
    return float(np.mean(recalls)) if recalls else 0.0


def train_classifier(epochs: int = 10, batch_size: int = 4, lr: float = 1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE]: Using {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_path = CACHE_DIR / "splits.json"
    assert splits_path.exists(), f"Missing splits.json at {splits_path}"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ds = CachedMRIDataset(splits["train"], is_train=True)
    val_ds = CachedMRIDataset(splits["val"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nCohort Sizes:")
    print(f"  - Train: {len(train_ds)} volumes")
    print(f"  - Val  : {len(val_ds)} volumes")
    print(f"  - Test : {len(splits['test'])} volumes")

    # 10-class model conforming to competition specification
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=10).to(device)

    # Class-weighted Focal Loss to balance 339 Glioma vs 140 Meningioma vs 140 Metastasis
    # Active classes: 1 (Glioma), 2 (Meningioma), 3 (Metastasis)
    present_classes = [1, 2, 3]
    loss_fn = FocalLoss(to_onehot_y=True, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_acc = -1.0
    best_ckpt_path = BASE_DIR / "best_model.pt"
    log_csv = BASE_DIR / "training_log.csv"

    log_history = []
    print(f"\n================ STARTING TRAINING (Epochs={epochs}, Batch={batch_size}) ================")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = loss_fn(logits, labels.unsqueeze(1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_bal_acc = compute_balanced_accuracy(val_targets, val_preds, present_classes)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {train_loss:.4f} | Val Bal Acc: {val_bal_acc*100:.2f}% | Time: {elapsed:.1f}s", flush=True)

        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_balanced_acc": val_bal_acc,
                "present_classes": present_classes,
                "timestamp": time.time()
            }, str(best_ckpt_path))
            print(f"  >> Checkpoint saved: best val bal acc = {val_bal_acc*100:.2f}%", flush=True)

        log_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_balanced_accuracy": val_bal_acc
        })

    # Save log
    import csv
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_balanced_accuracy"])
        writer.writeheader()
        writer.writerows(log_history)

    print(f"\n[OK] Training completed. Best Val Balanced Accuracy: {best_val_acc*100:.2f}%")
    return best_ckpt_path


if __name__ == "__main__":
    train_classifier(epochs=10, batch_size=4, lr=1e-4)
