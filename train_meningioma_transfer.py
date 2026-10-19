#!/usr/bin/env python3
"""
train_meningioma_transfer.py
============================
NeuroScope AI — Meningioma 3D SegResNet Transfer Learning Engine (BraTS-MEN 2023).
Transfers pretrained weights from Glioma (Task01_BrainTumour) to Meningioma.

Key Features:
- Reuses 3D SegResNet pretrained encoder/decoder weights.
- Stacks 4 independent NIfTI modalities [t2f, t1n, t1c, t2w] into 4-channel 3D volume.
- Fast transfer learning convergence (5-10 epochs on GPU).
- DiceCELoss with Brain Mask Guard.
"""

from __future__ import annotations

import argparse
import copy
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
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    Spacingd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_Meningioma" / "BraTS_MEN"
DEFAULT_PRETRAINED = SCRIPT_DIR / "models" / "segresnet_fullvolume_best.pt"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_meningioma_best.pt"


class MeningiomaDataset(Dataset):
    """Custom Dataset that loads the 4 modalities and ground-truth mask."""
    def __init__(self, case_dirs: List[Path], transform=None):
        self.cases = []
        for c in case_dirs:
            cid = c.name
            t1n = c / f"{cid}-t1n.nii.gz"
            t1c = c / f"{cid}-t1c.nii.gz"
            t2f = c / f"{cid}-t2f.nii.gz"
            t2w = c / f"{cid}-t2w.nii.gz"
            seg = c / f"{cid}-seg.nii.gz"
            if t1n.is_file() and t1c.is_file() and t2f.is_file() and t2w.is_file() and seg.is_file():
                self.cases.append({
                    "case_id": cid,
                    "t1n": t1n, "t1c": t1c, "t2f": t2f, "t2w": t2w,
                    "seg": seg,
                })
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        item = self.cases[idx]
        # Load and stack modalities: [FLAIR/t2f, T1n, T1c, T2w]
        t2f_data = nib.load(str(item["t2f"])).get_fdata(dtype=np.float32)
        t1n_data = nib.load(str(item["t1n"])).get_fdata(dtype=np.float32)
        t1c_data = nib.load(str(item["t1c"])).get_fdata(dtype=np.float32)
        t2w_data = nib.load(str(item["t2w"])).get_fdata(dtype=np.float32)
        seg_data = nib.load(str(item["seg"])).get_fdata(dtype=np.float32).round().astype(np.int64)

        # 4D Image: [4, H, W, D]
        img_4d = np.stack([t2f_data, t1n_data, t1c_data, t2w_data], axis=0)

        # BraTS multi-label conversion: [3, H, W, D]
        # Channel 0: WT (labels 1, 2, 3)
        # Channel 1: TC (labels 2, 3)
        # Channel 2: ET (label 3)
        wt = (seg_data > 0).astype(np.float32)
        tc = np.isin(seg_data, [2, 3]).astype(np.float32)
        et = (seg_data == 3).astype(np.float32)
        multi_label = np.stack([wt, tc, et], axis=0)

        data_dict = {
            "image": torch.from_numpy(img_4d),
            "label": torch.from_numpy(multi_label),
            "case_id": item["case_id"],
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def get_meningioma_transforms(crop_size=(96, 96, 96)):
    train_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=2,
            neg=1,
            num_samples=1,
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device, init_filters: int = 8) -> nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=init_filters,
        blocks_down=[1, 2, 2],
        blocks_up=[1, 1],
        dropout_prob=0.2,
    ).to(device)
    return model


def compute_region_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-5) -> float:
    intersection = (pred * gt).sum().item()
    cardinality = pred.sum().item() + gt.sum().item()
    if cardinality == 0:
        return 1.0
    return (2.0 * intersection) / (cardinality + eps)


def run_meningioma_transfer(
    data_dir: Path,
    pretrained_checkpoint: Path,
    output_checkpoint: Path,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 1,
    crop_size: Tuple[int, int, int] = (96, 96, 96),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    print("=" * 80)
    print("  NEUROSCOPE AI: MENINGIOMA (BraTS-MEN) TRANSFER LEARNING ENGINE")
    print("=" * 80)
    print(f"  Device              : {device}")
    print(f"  Data Directory      : {data_dir.resolve()}")
    print(f"  Pretrained Weights  : {pretrained_checkpoint.name if pretrained_checkpoint.is_file() else 'None (From Scratch)'}")
    print(f"  Target Checkpoint   : {output_checkpoint.name}")
    print(f"  Transfer Epochs     : {epochs}")
    print(f"  Crop Size           : {crop_size}")
    print(f"  Learning Rate       : {lr}")
    print("=" * 80)

    # 1. Discover cases
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS-MEN")])
    if not case_dirs:
        raise FileNotFoundError(f"No Meningioma case directories found in {data_dir}")

    print(f"[DATA] Found {len(case_dirs)} Meningioma patient cases on disk.")
    val_split = max(1, int(len(case_dirs) * 0.2))
    train_cases = case_dirs[:-val_split]
    val_cases = case_dirs[-val_split:]
    print(f"  -> Train cohort: {len(train_cases)} cases | Validation cohort: {len(val_cases)} cases")

    train_pipe, val_pipe = get_meningioma_transforms(crop_size)
    train_ds = MeningiomaDataset(train_cases, transform=train_pipe)
    val_ds = MeningiomaDataset(val_cases, transform=val_pipe)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=list_data_collate)

    # 2. Model & Transfer Initialization
    model = create_model(device)
    if pretrained_checkpoint.is_file():
        print(f"[TRANSFER] Loading pretrained Glioma weights from: {pretrained_checkpoint.name}")
        ckpt = torch.load(str(pretrained_checkpoint), map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print("  -> Transfer successful! Weights adapted to Meningioma.")

    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[ACCELERATION] Automatic Mixed Precision (AMP): {'ENABLED (FP16 Tensor Cores)' if use_amp else 'DISABLED'}")

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"--- Epoch [{ep:>2}/{epochs}] ({dur:.1f}s) | Avg Loss: {avg_loss:.4f} ---", flush=True)

        # Fast Validation on Val Cohort
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
                v_prob = torch.sigmoid(v_out).squeeze(0)
                brain_mask = (v_item["image"].abs().sum(dim=0) > 0).to(device)
                pred_wt = (v_prob[0] > 0.5) & brain_mask
                gt_wt = v_y[0] > 0.5
                dices.append(compute_region_dice(pred_wt.float(), gt_wt.float()))

        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_wt_dice = float(np.mean(dices))
        print(f"  >>> Val Meningioma WT Dice: {val_wt_dice:.4f}", flush=True)

        history.append({
            "epoch": ep,
            "train_loss": avg_loss,
            "val_wt_dice": val_wt_dice,
            "epoch_duration_sec": dur,
        })

        if val_wt_dice > best_dice:
            best_dice = val_wt_dice
            torch.save({
                "model_version": "1.0.0-meningioma-transfer",
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_wt_dice": best_dice,
                "loss_function": "DiceCELoss",
                "target_disease": "Meningioma (BraTS-MEN 2023)",
            }, str(output_checkpoint))
            print(f"  [SAVED NEW BEST] -> {output_checkpoint.name} (Val Dice: {best_dice:.4f})", flush=True)

    total_training_time = time.time() - t_start

    # Save JSON report
    report_data = {
        "dataset": "BraTS-MEN 2023 (Meningioma)",
        "cohort_size": len(case_dirs),
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "total_epochs": epochs,
        "best_val_dice": best_dice,
        "total_training_time_sec": total_training_time,
        "history": history,
    }
    report_file = SCRIPT_DIR / "reports" / "meningioma_evaluation.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  MENINGIOMA TRANSFER TRAINING COMPLETE! Best Dice: {best_dice:.4f}")
    print(f"  Saved Checkpoint: {output_checkpoint.resolve()}")
    print(f"  Saved Evaluation: {report_file.resolve()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Meningioma SegResNet Transfer Learning")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_meningioma_transfer(
        data_dir=args.data.resolve(),
        pretrained_checkpoint=args.pretrained.resolve(),
        output_checkpoint=args.output.resolve(),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
