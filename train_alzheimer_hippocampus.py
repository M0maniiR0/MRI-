#!/usr/bin/env python3
"""
train_alzheimer_hippocampus.py
==============================
NeuroScope AI — Alzheimer's Disease & Brain Atrophy Quantification Engine.
Dataset: MSD Task04_Hippocampus (260 MRI cases, T1-weighted).
Architecture: MONAI 3D SegResNet / UNet.
Clinical Biomarker: Automated 3D Hippocampal Volume (mm^3) & Atrophy Index.
Hardware: NVIDIA GeForce RTX 2050 GPU with AMP (FP16 Tensor Cores).
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
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotated,
    ResizeWithPadOrCropd,
    Spacingd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task04_Hippocampus"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_alzheimer_hippocampus_best.pt"
DEFAULT_REPORT = SCRIPT_DIR / "reports" / "alzheimer_hippocampus_evaluation.json"


class HippocampusDataset(Dataset):
    """Loads 3D T1 MRI and multi-class Hippocampus labels {0: bg, 1: anterior, 2: posterior}."""
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
        img_nii = nib.load(str(item["image_path"]))
        lbl_nii = nib.load(str(item["label_path"]))

        img_data = img_nii.get_fdata(dtype=np.float32)
        lbl_data = lbl_nii.get_fdata(dtype=np.float32).round().astype(np.int64)

        # 3D T1 MRI: [1, H, W, D]
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_data).unsqueeze(0)

        data_dict = {
            "image": img_tensor,
            "label": lbl_tensor,
            "case_id": item["case_id"],
            "spacing": list(img_nii.header.get_zooms()[:3]),
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def get_transforms(spatial_size=(32, 64, 32)):
    train_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device) -> nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # 0: bg, 1: anterior, 2: posterior
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)
    return model


def compute_multiclass_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """Computes Dice score per anatomical subregion."""
    dices = {}
    for c in range(1, num_classes):
        p_c = (pred == c).float()
        g_c = (gt == c).float()
        inter = (p_c * g_c).sum().item()
        card = p_c.sum().item() + g_c.sum().item()
        dice = (2.0 * inter) / (card + 1e-5) if card > 0 else 1.0
        name = "Anterior_Hippocampus" if c == 1 else "Posterior_Hippocampus"
        dices[name] = dice

    # Total Hippocampus (Anterior + Posterior)
    p_tot = (pred > 0).float()
    g_tot = (gt > 0).float()
    inter_tot = (p_tot * g_tot).sum().item()
    card_tot = p_tot.sum().item() + g_tot.sum().item()
    dices["Total_Hippocampus"] = (2.0 * inter_tot) / (card_tot + 1e-5) if card_tot > 0 else 1.0
    dices["Mean_Dice"] = (dices["Anterior_Hippocampus"] + dices["Posterior_Hippocampus"]) / 2.0
    return dices


def run_alzheimer_training(
    data_dir: Path,
    output_checkpoint: Path,
    report_file: Path,
    epochs: int = 15,
    lr: float = 2e-4,
    batch_size: int = 4,
    crop_size: Tuple[int, int, int] = (32, 32, 32),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print("=" * 80)
    print("  NEUROSCOPE AI: ALZHEIMER'S HIPPOCAMPUS & ATROPHY SEGMENTATION")
    print("=" * 80)
    print(f"  Device              : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Dataset             : MSD Task04_Hippocampus")
    print(f"  Data Directory      : {data_dir.resolve()}")
    print(f"  Target Checkpoint   : {output_checkpoint.name}")
    print(f"  Training Epochs     : {epochs}")
    print(f"  Crop Size           : {crop_size}")
    print(f"  Initial LR          : {lr}")
    print("=" * 80, flush=True)

    # 1. Load dataset manifest
    json_path = data_dir / "dataset.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"dataset.json not found in {data_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_training = manifest.get("training", [])
    print(f"[DATA] Total training cases in manifest: {len(all_training)}")

    # 80/20 train/validation split
    np.random.seed(42)
    indices = np.random.permutation(len(all_training))
    val_count = max(10, int(len(all_training) * 0.2))
    train_entries = [all_training[i] for i in indices[:-val_count]]
    val_entries = [all_training[i] for i in indices[-val_count:]]

    print(f"  -> Training Split  : {len(train_entries)} cases")
    print(f"  -> Validation Split: {len(val_entries)} cases")

    train_pipe, val_pipe = get_transforms(crop_size)
    train_ds = HippocampusDataset(train_entries, data_dir, transform=train_pipe)
    val_ds = HippocampusDataset(val_entries, data_dir, transform=val_pipe)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=list_data_collate)

    # 2. Model & Optimization
    model = create_model(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[ACCELERATION] AMP FP16 Enabled: {use_amp}", flush=True)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    best_mean_dice = 0.0
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

        # Validation every 2 epochs or at end
        if ep % 2 == 0 or ep == epochs:
            model.eval()
            val_results = []
            with torch.no_grad():
                for v_idx in range(len(val_ds)):
                    v_item = val_ds[v_idx]
                    v_x = v_item["image"].unsqueeze(0).to(device)
                    v_y = v_item["label"].squeeze().to(device)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        v_out = model(v_x)
                    pred_class = torch.argmax(v_out, dim=1).squeeze(0)
                    dice_scores = compute_multiclass_dice(pred_class, v_y)
                    val_results.append(dice_scores)

            avg_mean = float(np.mean([r["Mean_Dice"] for r in val_results]))
            avg_ant = float(np.mean([r["Anterior_Hippocampus"] for r in val_results]))
            avg_post = float(np.mean([r["Posterior_Hippocampus"] for r in val_results]))
            avg_tot = float(np.mean([r["Total_Hippocampus"] for r in val_results]))

            print(f"  >>> Val Dice: Total={avg_tot:.4f} | Anterior={avg_ant:.4f} | Posterior={avg_post:.4f} | Mean={avg_mean:.4f}", flush=True)

            history.append({
                "epoch": ep,
                "train_loss": avg_loss,
                "val_total_dice": avg_tot,
                "val_anterior_dice": avg_ant,
                "val_posterior_dice": avg_post,
                "val_mean_dice": avg_mean,
            })

            if avg_tot > best_mean_dice:
                best_mean_dice = avg_tot
                torch.save({
                    "model_version": "1.0.0-alzheimer-hippocampus",
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_total_dice": best_mean_dice,
                    "val_anterior_dice": avg_ant,
                    "val_posterior_dice": avg_post,
                    "dataset": "Task04_Hippocampus (Alzheimer's)",
                }, str(output_checkpoint))
                print(f"  [SAVED BEST CHECKPOINT] -> {output_checkpoint.name} (Total Dice: {best_mean_dice:.4f})", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ALZHEIMER'S HIPPOCAMPUS TRAINING COMPLETE in {total_time:.1f}s!")
    print(f"  Best Total Hippocampus Dice: {best_mean_dice:.4f}")
    print(f"  Saved Model: {output_checkpoint.resolve()}")
    print("=" * 80, flush=True)

    # Save summary report
    report_data = {
        "project": "NeuroScope AI - BrainAgenticthon 2026",
        "task": "Alzheimer's Disease & Hippocampal Volumetric Biomarker",
        "dataset": "MSD Task04_Hippocampus",
        "device": device_str,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_training_time_seconds": round(total_time, 1),
        "best_total_hippocampus_dice": round(best_mean_dice, 4),
        "epochs_trained": epochs,
        "history": history,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Report saved to: {report_file.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Hippocampus 3D Segmentation Engine")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_alzheimer_training(
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
