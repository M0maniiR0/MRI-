#!/usr/bin/env python3
"""
train_brain_metastasis_transfer.py
==================================
NeuroScope AI — Brain Metastasis (الانتقالات والأورام الدبقية المنتشرة) 3D Segmentation Engine.
Dataset: 40 3D Patient Cohorts (Pakawat-Phasook / Hugging Face).
Hardware: NVIDIA GeForce RTX 2050 with Automatic Mixed Precision (AMP FP16).
Outputs:
  - Checkpoint: models/segresnet_brain_metastasis_best.pt
  - Metrics Report: reports/brain_metastasis_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    Spacingd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_BrainMetastasis"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_brain_metastasis_best.pt"
DEFAULT_REPORT = SCRIPT_DIR / "reports" / "brain_metastasis_evaluation.json"


class BrainMetastasisDataset(Dataset):
    """Loads 3D T1-CE MRI and binary metastatic lesion ground-truth mask."""
    def __init__(self, manifest_items: List[Dict[str, str]], transform=None):
        self.items = manifest_items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mri_nii = nib.load(item["mri_path"])
        mask_nii = nib.load(item["lesion_mask_path"])

        mri_data = mri_nii.get_fdata(dtype=np.float32)
        # Any label > 0 is metastatic lesion
        mask_data = (mask_nii.get_fdata(dtype=np.float32) > 0).astype(np.float32)

        # [1, H, W, D]
        img_tensor = torch.from_numpy(mri_data).unsqueeze(0)
        lbl_tensor = torch.from_numpy(mask_data).unsqueeze(0)

        data_dict = {
            "image": img_tensor,
            "label": lbl_tensor,
            "patient_id": item["patient_id"],
            "spacing": list(mri_nii.header.get_zooms()[:3]),
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def get_metastasis_transforms(crop_size=(96, 96, 96)):
    train_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=3,
            neg=1,
            num_samples=1,
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device) -> nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,  # Binary sigmoid for Metastatic Lesion
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)
    return model


def compute_dice_score(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-5) -> float:
    intersection = (pred * gt).sum().item()
    cardinality = pred.sum().item() + gt.sum().item()
    if cardinality == 0:
        return 1.0
    return (2.0 * intersection) / (cardinality + eps)


def run_metastasis_training(
    data_dir: Path,
    output_checkpoint: Path,
    report_file: Path,
    epochs: int = 15,
    lr: float = 2e-4,
    batch_size: int = 2,
    crop_size: Tuple[int, int, int] = (96, 96, 96),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print("=" * 80)
    print("  NEUROSCOPE AI: BRAIN METASTASIS 3D SEGMENTATION & DEEP TRANSFER")
    print("=" * 80)
    print(f"  Device              : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Target Checkpoint   : {output_checkpoint.name}")
    print(f"  Training Epochs     : {epochs}")
    print(f"  Crop Size           : {crop_size}")
    print(f"  Initial LR          : {lr}")
    print("=" * 80, flush=True)

    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found in {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 80/20 train/validation split
    np.random.seed(42)
    indices = np.random.permutation(len(manifest))
    val_count = max(4, int(len(manifest) * 0.2))
    train_entries = [manifest[i] for i in indices[:-val_count]]
    val_entries = [manifest[i] for i in indices[-val_count:]]

    print(f"[DATA] Total Metastasis Cohort : {len(manifest)} patients")
    print(f"  -> Training Split           : {len(train_entries)} patients")
    print(f"  -> Validation Cohort        : {len(val_entries)} patients")

    train_pipe, val_pipe = get_metastasis_transforms(crop_size)
    train_ds = BrainMetastasisDataset(train_entries, transform=train_pipe)
    val_ds = BrainMetastasisDataset(val_entries, transform=val_pipe)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=list_data_collate)

    # 2. Model & Optimization
    model = create_model(device)
    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[ACCELERATION] AMP FP16 Enabled: {use_amp}", flush=True)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    history = []

    # 3. Training Loop
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        step = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            step += 1

        scheduler.step()
        avg_loss = ep_loss / max(1, step)
        dur = time.time() - t0
        print(f"--- Epoch [{ep:>2}/{epochs}] ({dur:.1f}s) | Train Loss: {avg_loss:.4f} ---", flush=True)

        # Fast Validation on Val Cohort
        if ep % 2 == 0 or ep == epochs:
            model.eval()
            dices = []
            with torch.no_grad():
                for v_idx in range(len(val_ds)):
                    v_item = val_ds[v_idx]
                    v_x = v_item["image"].unsqueeze(0).to(device)
                    v_y = v_item["label"].to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        v_out = sliding_window_inference(
                            inputs=v_x,
                            roi_size=crop_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.25,
                            mode="gaussian",
                        )
                    pred_bin = (torch.sigmoid(v_out).squeeze(0) > 0.5).float()
                    dice = compute_dice_score(pred_bin, v_y)
                    dices.append(dice)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            val_dice = float(np.mean(dices))
            print(f"  >>> Val Brain Metastasis Dice: {val_dice:.4f}", flush=True)

            history.append({
                "epoch": ep,
                "train_loss": avg_loss,
                "val_dice": val_dice,
                "epoch_duration_sec": dur,
            })

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "model_version": "1.0.0-brain-metastasis",
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_dice,
                    "dataset": "Pakawat-Phasook / Brain Metastasis",
                }, str(output_checkpoint))
                print(f"  [SAVED BEST CHECKPOINT] -> {output_checkpoint.name} (Val Dice: {best_dice:.4f})", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  BRAIN METASTASIS TRAINING COMPLETE in {total_time:.1f}s!")
    print(f"  Best Validation Dice: {best_dice:.4f}")
    print(f"  Saved Model Checkpoint: {output_checkpoint.resolve()}")
    print("=" * 80, flush=True)

    report_data = {
        "project": "NeuroScope AI - BrainAgenticthon 2026",
        "task": "Brain Metastasis 3D Segmentation (الأورام الدماغية المنتشرة)",
        "dataset": "Pakawat-Phasook / Brain Metastases Dataset",
        "cohort_size": len(manifest),
        "train_patients": len(train_entries),
        "val_patients": len(val_entries),
        "epochs_trained": epochs,
        "best_val_dice": round(best_dice, 4),
        "total_training_time_seconds": round(total_time, 1),
        "history": history,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved Report: {report_file.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Brain Metastasis 3D SegResNet Training")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_metastasis_training(
        data_dir=args.data.resolve(),
        output_checkpoint=args.output.resolve(),
        report_file=args.report.resolve(),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
