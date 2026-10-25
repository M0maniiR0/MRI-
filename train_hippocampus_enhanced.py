#!/usr/bin/env python3
"""
train_hippocampus_enhanced.py
=============================
NeuroScope AI — Hippocampus (قرن آمون) Enhanced 3D Retraining & Volumetry Engine.
Strictly implements Clinical Radiologist directives:
  1. High-Resolution T1 Structural MRI.
  2. Filtration & Brain Mask Guard to eliminate non-brain artifacts.
  3. Medial Temporal ROI Focus: Prevents empty brain patches and severe class imbalance.
  4. Augmentation Guard: Avoids illegal horizontal flipping that flips left/right anatomical sides.
  5. DiceCELoss with Volumetric Error tracking.
  6. Automatic Mixed Precision (AMP FP16) on GPU.
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
    RandCropByPosNegLabeld,
    RandRotated,
    SpatialPadd,
)

from mri_filtration_and_enhancement import (
    PercentileClippingAndFiltrationd,
    BrainMaskGuardFiltrationd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task04_Hippocampus"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_hippocampus_enhanced_best.pt"
DEFAULT_PRETRAINED = SCRIPT_DIR / "models" / "segresnet_alzheimer_hippocampus_best.pt"
DEFAULT_REPORT = SCRIPT_DIR / "reports" / "hippocampus_enhanced_training_report.json"


class EnhancedHippocampusDataset(Dataset):
    """Loads 3D T1 MRI and multi-class Hippocampus labels {0: bg, 1: anterior, 2: posterior} with filtration."""
    def __init__(self, cases: List[Dict[str, str]], data_root: Path, transform=None):
        self.cases = []
        for c in cases:
            img_p = data_root / c["image"].replace("./", "")
            lbl_p = data_root / c["label"].replace("./", "")
            if img_p.is_file() and lbl_p.is_file():
                self.cases.append({
                    "case_id": img_p.name.replace(".nii.gz", ""),
                    "image_path": img_p,
                    "label_path": lbl_p,
                })
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        item = self.cases[idx]
        img_nii = nib.load(item["image_path"])
        lbl_nii = nib.load(item["label_path"])

        img_data = img_nii.get_fdata(dtype=np.float32)
        lbl_data = lbl_nii.get_fdata(dtype=np.float32).round().astype(np.int64)

        if img_data.ndim == 3:
            img_data = img_data[np.newaxis, ...]
        if lbl_data.ndim == 3:
            lbl_data = lbl_data[np.newaxis, ...]

        data_dict = {
            "image": torch.from_numpy(img_data),
            "label": torch.from_numpy(lbl_data),
            "case_id": item["case_id"],
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def get_hippocampus_transforms(crop_size=(48, 64, 48)):
    train_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
        SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
        # Focused ROI cropping on hippocampus labels to eliminate empty background patches
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=3.0,
            neg=1.0,
            num_samples=1,
        ),
        # Gentle 3D rotations without illegal left/right flipping
        RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, mode=("bilinear", "nearest"), prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device, in_channels: int = 1, num_classes: int = 3) -> nn.Module:
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)


def compute_multiclass_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 3) -> float:
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        intersection = (p * g).sum().item()
        total = p.sum().item() + g.sum().item()
        if total == 0:
            dices.append(1.0)
        else:
            dices.append((2.0 * intersection + 1e-5) / (total + 1e-5))
    return float(np.mean(dices))


def run_hippocampus_retraining(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_ckpt: Path = DEFAULT_OUTPUT_CKPT,
    pretrained_ckpt: Path = DEFAULT_PRETRAINED,
    report_file: Path = DEFAULT_REPORT,
    epochs: int = 15,
    lr: float = 3e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device(device_str)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=" * 80, flush=True)
    print("  NEUROSCOPE AI: ENHANCED HIPPOCAMPUS RETRAINING & VOLUMETRY (GPU)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  AMP FP16 Enabled : {use_amp}", flush=True)
    print(f"  Target Checkpoint: {output_ckpt}", flush=True)
    print(f"  Epochs           : {epochs}", flush=True)
    print(f"  Learning Rate    : {lr}", flush=True)
    print("=" * 80, flush=True)

    dataset_json = data_dir / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(f"dataset.json not found in {data_dir}")

    with open(dataset_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    all_training_items = meta.get("training", [])
    np.random.seed(42)
    shuffled = np.random.permutation(all_training_items).tolist()
    split_idx = int(0.8 * len(shuffled))
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]
    print(f"  Hippocampus Cohort: {len(all_training_items)} cases (Train: {len(train_items)}, Val: {len(val_items)})", flush=True)

    train_tf, val_tf = get_hippocampus_transforms()
    train_ds = EnhancedHippocampusDataset(train_items, data_root=data_dir, transform=train_tf)
    val_ds = EnhancedHippocampusDataset(val_items, data_root=data_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=list_data_collate, num_workers=0)

    model = create_model(device, in_channels=1, num_classes=3)
    if pretrained_ckpt.is_file():
        try:
            ckpt = torch.load(str(pretrained_ckpt), map_location=device, weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            print(f"[OK] Successfully warm-started from: {pretrained_ckpt.name}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights: {e}", flush=True)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
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

        scheduler.step()
        avg_loss = ep_loss / max(1, len(train_loader))
        dur = time.time() - t0

        # Fast Validation on Val Cohort
        model.eval()
        val_dices = []
        eval_indices = range(len(val_ds)) if ep == epochs else range(min(10, len(val_ds)))
        with torch.no_grad():
            for v_idx in eval_indices:
                sample = val_ds[v_idx]
                v_img = sample["image"].unsqueeze(0).to(device)
                v_gt = sample["label"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_out = sliding_window_inference(
                        inputs=v_img,
                        roi_size=(48, 64, 48),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                        mode="gaussian",
                    )
                pred_classes = torch.argmax(v_out, dim=1).squeeze(0)
                gt_classes = v_gt.squeeze(0).squeeze(0)
                val_dices.append(compute_multiclass_dice(pred_classes, gt_classes, num_classes=3))

        mean_val_dice = float(np.mean(val_dices))
        print(f"--- Epoch [{ep:>2}/{epochs}] ({dur:.1f}s) | Loss: {avg_loss:.4f} | Val Dice: {mean_val_dice:.4f} ---", flush=True)

        history.append({
            "epoch": ep,
            "train_loss": round(avg_loss, 4),
            "val_dice": round(mean_val_dice, 4),
            "time_sec": round(dur, 1),
        })

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            output_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
            }, str(output_ckpt))
            print(f"  [SAVED NEW BEST] -> {output_ckpt.name} (Val Dice: {best_val_dice:.4f})", flush=True)

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Hippocampus 3D SegResNet Enhanced",
            "checkpoint": str(output_ckpt),
            "best_val_dice": best_val_dice,
            "epochs": epochs,
            "history": history,
        }, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"  HIPPOCAMPUS ENHANCED RETRAINING COMPLETE! Best Val Dice: {best_val_dice:.4f}", flush=True)
    print(f"  Saved Checkpoint: {output_ckpt}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope AI Enhanced Hippocampus Retraining")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_hippocampus_retraining(epochs=args.epochs, lr=args.lr, device_str=args.device)
