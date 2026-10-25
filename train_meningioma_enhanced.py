#!/usr/bin/env python3
"""
train_meningioma_enhanced.py
============================
NeuroScope AI — Meningioma 3D SegResNet Enhanced Retraining Engine (BraTS-MEN 2023).
Applies Clinical Filtration & Enhancement directives:
  1. Percentile Clipping (0.5% - 99.5%) to eliminate scanner noise.
  2. Brain Mask Guard to eliminate skull bone / dural sinus false positives.
  3. Focused T1ce contrast enhancement.
  4. Automatic Mixed Precision (AMP FP16) for GPU memory safety.
  5. Evaluates on full validation cohort without data leakage.
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
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    SpatialPadd,
)

from mri_filtration_and_enhancement import (
    PercentileClippingAndFiltrationd,
    BrainMaskGuardFiltrationd,
    ContrastEnhancementT1ced,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_Meningioma" / "BraTS_MEN"
DEFAULT_PRETRAINED = SCRIPT_DIR / "models" / "segresnet_meningioma_best.pt"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_meningioma_enhanced_best.pt"
DEFAULT_REPORT_FILE = SCRIPT_DIR / "reports" / "meningioma_enhanced_training_report.json"


class EnhancedMeningiomaDataset(Dataset):
    """Loads the 4 MRI modalities [t2f, t1n, t1c, t2w] and ground-truth mask with filtration."""
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
        t2f_data = nib.load(str(item["t2f"])).get_fdata(dtype=np.float32)
        t1n_data = nib.load(str(item["t1n"])).get_fdata(dtype=np.float32)
        t1c_data = nib.load(str(item["t1c"])).get_fdata(dtype=np.float32)
        t2w_data = nib.load(str(item["t2w"])).get_fdata(dtype=np.float32)
        seg_data = nib.load(str(item["seg"])).get_fdata(dtype=np.float32).round().astype(np.int64)

        # 4D Image: [4, H, W, D] in strictly verified order: [t2f, t1n, t1c, t2w]
        img_4d = np.stack([t2f_data, t1n_data, t1c_data, t2w_data], axis=0)

        # Multi-Label Target: [3, H, W, D]
        # Ch0: WT (seg > 0)
        # Ch1: Solid Core / TC (seg in [2, 3])
        # Ch2: ET (seg == 3)
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


def get_enhanced_transforms(crop_size=(96, 96, 96)):
    train_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=2, gamma=1.2),
        SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=3,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=2, gamma=1.2),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device, init_filters: int = 8) -> nn.Module:
    return SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=init_filters,
        blocks_down=[1, 2, 2],
        blocks_up=[1, 1],
        dropout_prob=0.2,
    ).to(device)


def compute_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-5) -> float:
    intersection = (pred * gt).sum().item()
    total = pred.sum().item() + gt.sum().item()
    if total == 0:
        return 1.0
    return (2.0 * intersection + eps) / (total + eps)


def run_meningioma_retraining(
    data_dir: Path = DEFAULT_DATA_DIR,
    pretrained_ckpt: Path = DEFAULT_PRETRAINED,
    output_ckpt: Path = DEFAULT_OUTPUT_CKPT,
    report_path: Path = DEFAULT_REPORT_FILE,
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
    print("  NEUROSCOPE AI: ENHANCED MENINGIOMA RETRAINING ENGINE (GPU)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  AMP FP16 Enabled : {use_amp}", flush=True)
    print(f"  Pretrained Checkpoint: {pretrained_ckpt}", flush=True)
    print(f"  Target Checkpoint: {output_ckpt}", flush=True)
    print(f"  Epochs           : {epochs}", flush=True)
    print(f"  Learning Rate    : {lr}", flush=True)
    print("=" * 80, flush=True)

    all_cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS-MEN")])
    if not all_cases:
        raise FileNotFoundError(f"No BraTS-MEN cases found in {data_dir}")

    # Patient-level split: 80% train, 20% val (zero leakage)
    np.random.seed(42)
    shuffled = np.random.permutation(all_cases).tolist()
    split_idx = int(0.8 * len(shuffled))
    train_cases = shuffled[:split_idx]
    val_cases = shuffled[split_idx:]
    print(f"  Total Cases: {len(all_cases)} (Train: {len(train_cases)}, Val: {len(val_cases)})", flush=True)

    train_tf, val_tf = get_enhanced_transforms()
    train_ds = EnhancedMeningiomaDataset(train_cases, transform=train_tf)
    val_ds = EnhancedMeningiomaDataset(val_cases, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=list_data_collate, num_workers=0)

    model = create_model(device, init_filters=8)
    if pretrained_ckpt.is_file():
        try:
            ckpt = torch.load(str(pretrained_ckpt), map_location=device, weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            print(f"[OK] Successfully warm-started from: {pretrained_ckpt.name}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights: {e}", flush=True)

    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
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

        # Fast Validation on Val Cohort (Ch1 Solid Core @ 0.50)
        model.eval()
        val_dices = []
        eval_cases = val_cases if ep == epochs else val_cases[:10]
        with torch.no_grad():
            for v_dir in eval_cases:
                item_ds = EnhancedMeningiomaDataset([v_dir], transform=val_tf)
                sample = item_ds[0]
                v_img = sample["image"].unsqueeze(0).to(device)
                v_gt = sample["label"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_out = sliding_window_inference(
                        inputs=v_img,
                        roi_size=(96, 96, 96),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                        mode="gaussian",
                    )
                v_probs = torch.sigmoid(v_out).squeeze(0)
                # Channel 1: Solid Tumor Core
                brain_mask = v_img.squeeze(0).abs().sum(dim=0) > 1e-4
                pred_tc = (v_probs[1] > 0.5) & brain_mask
                gt_tc = v_gt[1] > 0.5
                val_dices.append(compute_dice(pred_tc.float(), gt_tc.float()))

        mean_val_dice = float(np.mean(val_dices))
        print(f"--- Epoch [{ep:>2}/{epochs}] ({dur:.1f}s) | Loss: {avg_loss:.4f} | Val TC Dice: {mean_val_dice:.4f} ---", flush=True)

        history.append({
            "epoch": ep,
            "train_loss": round(avg_loss, 4),
            "val_tc_dice": round(mean_val_dice, 4),
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
            print(f"  [SAVED NEW BEST] -> {output_ckpt.name} (TC Dice: {best_val_dice:.4f})", flush=True)

    # Save final report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Meningioma 3D SegResNet Enhanced",
            "checkpoint": str(output_ckpt),
            "best_val_tc_dice": best_val_dice,
            "epochs": epochs,
            "history": history,
        }, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"  MENINGIOMA ENHANCED RETRAINING COMPLETE! Best Val TC Dice: {best_val_dice:.4f}", flush=True)
    print(f"  Saved Checkpoint: {output_ckpt}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope AI Enhanced Meningioma Retraining")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_meningioma_retraining(epochs=args.epochs, lr=args.lr, device_str=args.device)
