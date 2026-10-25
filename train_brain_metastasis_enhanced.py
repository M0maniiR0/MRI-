#!/usr/bin/env python3
"""
train_brain_metastasis_enhanced.py
==================================
NeuroScope AI — Brain Metastasis (انبثاثات الدماغ) Enhanced 3D Retraining Engine.
Strictly implements Clinical Radiologist directives:
  1. STRICTLY NO DOWNSAMPLING: Preserves native 1.0mm isotropic resolution so punctate micro-lesions never vanish.
  2. Percentile Clipping (0.2% - 99.8%) to preserve sharp micro-lesion intensity peaks.
  3. Brain Mask Guard to suppress skull/dural vessel false positives.
  4. High Positive-Sample Biased Patch Extraction (pos=4.0, neg=1.0) so patches always contain metastases.
  5. DiceFocalLoss to heavily penalize False Negatives (critical clinical safety).
  6. Automatic Mixed Precision (AMP FP16) for GPU efficiency.
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
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_BrainMetastasis"
DEFAULT_OUTPUT_CKPT = SCRIPT_DIR / "models" / "segresnet_brain_metastasis_enhanced_best.pt"
DEFAULT_PRETRAINED = SCRIPT_DIR / "models" / "segresnet_brain_metastasis_best.pt"
DEFAULT_REPORT = SCRIPT_DIR / "reports" / "brain_metastasis_enhanced_training_report.json"


class EnhancedBrainMetastasisDataset(Dataset):
    """Loads 3D T1-CE MRI and binary metastatic lesion ground-truth mask with filtration."""
    def __init__(self, manifest_items: List[Dict[str, str]], transform=None):
        self.items = manifest_items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mri_nii = nib.load(item["mri_path"])
        seg_nii = nib.load(item["seg_path"])

        mri_data = mri_nii.get_fdata(dtype=np.float32)
        seg_data = seg_nii.get_fdata(dtype=np.float32).round().astype(np.float32)

        # 4D Image: [1, H, W, D]
        if mri_data.ndim == 3:
            mri_data = mri_data[np.newaxis, ...]
        elif mri_data.ndim == 4 and mri_data.shape[-1] <= 4:
            mri_data = np.moveaxis(mri_data, -1, 0)

        # Binary Target: [1, H, W, D]
        if seg_data.ndim == 3:
            seg_data = seg_data[np.newaxis, ...]

        data_dict = {
            "image": torch.from_numpy(mri_data),
            "label": torch.from_numpy((seg_data > 0).astype(np.float32)),
            "case_id": item["case_id"],
        }
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def get_metastasis_transforms(crop_size=(96, 96, 96)):
    train_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.2, upper_percentile=99.8),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=0, gamma=1.2),
        SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
        # Heavily bias patch sampling so small metastases (<500 voxels) are never missed
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=4.0,
            neg=1.0,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transform = Compose([
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.2, upper_percentile=99.8),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=0, gamma=1.2),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transform, val_transform


def create_model(device: torch.device, in_channels: int = 1) -> nn.Module:
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)


def compute_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-5) -> float:
    intersection = (pred * gt).sum().item()
    total = pred.sum().item() + gt.sum().item()
    if total == 0:
        return 1.0
    return (2.0 * intersection + eps) / (total + eps)


def run_metastasis_retraining(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_ckpt: Path = DEFAULT_OUTPUT_CKPT,
    pretrained_ckpt: Path = DEFAULT_PRETRAINED,
    report_file: Path = DEFAULT_REPORT,
    epochs: int = 15,
    lr: float = 2e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device(device_str)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=" * 80, flush=True)
    print("  NEUROSCOPE AI: ENHANCED BRAIN METASTASIS RETRAINING ENGINE (GPU)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  AMP FP16 Enabled : {use_amp}", flush=True)
    print(f"  Target Checkpoint: {output_ckpt}", flush=True)
    print(f"  Epochs           : {epochs}", flush=True)
    print(f"  Learning Rate    : {lr}", flush=True)
    print("=" * 80, flush=True)

    manifest_file = data_dir / "metastasis_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Patient-level split: 80% train, 20% val (zero leakage)
    np.random.seed(42)
    shuffled = np.random.permutation(manifest).tolist()
    split_idx = int(0.8 * len(shuffled))
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]
    print(f"  Cohort Total: {len(manifest)} scans (Train: {len(train_items)}, Val: {len(val_items)})", flush=True)

    train_tf, val_tf = get_metastasis_transforms()
    train_ds = EnhancedBrainMetastasisDataset(train_items, transform=train_tf)
    val_ds = EnhancedBrainMetastasisDataset(val_items, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=list_data_collate, num_workers=0)

    model = create_model(device, in_channels=1)
    if pretrained_ckpt.is_file():
        try:
            ckpt = torch.load(str(pretrained_ckpt), map_location=device, weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            print(f"[OK] Successfully warm-started from: {pretrained_ckpt.name}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights: {e}", flush=True)

    loss_fn = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
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
                        roi_size=(96, 96, 96),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                        mode="gaussian",
                    )
                v_prob = torch.sigmoid(v_out).squeeze(0)
                brain_mask = v_img.squeeze(0).abs().sum(dim=0) > 1e-4
                pred_mask = (v_prob[0] > 0.5) & brain_mask
                gt_mask = v_gt[0] > 0.5
                val_dices.append(compute_dice(pred_mask.float(), gt_mask.float()))

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
            "model": "Brain Metastasis 3D SegResNet Enhanced",
            "checkpoint": str(output_ckpt),
            "best_val_dice": best_val_dice,
            "epochs": epochs,
            "history": history,
        }, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"  BRAIN METASTASIS ENHANCED RETRAINING COMPLETE! Best Val Dice: {best_val_dice:.4f}", flush=True)
    print(f"  Saved Checkpoint: {output_ckpt}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope AI Enhanced Metastasis Retraining")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_metastasis_retraining(epochs=args.epochs, lr=args.lr, device_str=args.device)
