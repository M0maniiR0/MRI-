#!/usr/bin/env python3
"""
segmentation_baseline.py
========================
3D Glioma Tumor Segmentation Baseline (SegResNet / MONAI) for MSD Task01.
BrainAgenticthon 2026 - Medical AI Workstream (Abdulrahman & Rou'a).

Features:
---------
1. Model: SegResNet (3D, 4 input channels, 3 sigmoid output channels for WT, TC, ET).
2. Training: Zero-leakage using splits/train_ids.csv and splits/val_ids.csv.
3. Loss: DiceLoss with Sigmoid activation across [WT, TC, ET].
4. Evaluation: Calculates per-case and mean Dice scores for WT, TC, and ET.
5. Inversion & Clinical Export:
   - Reconstructs multiclass mask {0:bg, 1:edema, 2:non-enhancing, 3:enhancing}.
   - Inverts mask back to patient original space.
   - Saves 'mask.nii.gz' with original affine and header.
   - Computes clinical tumor volumes (mL and mm^3).
   - Generates 'segmentation_validation_metrics.json'.
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SegResNet

# Import verified preprocessing pipeline
from monai_preprocessing_pipeline import (
    MODALITIES,
    ConvertToBraTSMultiLabeld,
    convert_multilabel_to_multiclass,
    get_spatial_pipeline,
    get_training_pipeline,
    get_validation_pipeline,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = _SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"
DEFAULT_SPLITS_DIR = _SCRIPT_DIR / "splits"
DEFAULT_CHECKPOINT = _SCRIPT_DIR / "models" / "segresnet_baseline.pt"


def load_split_case_ids(csv_path: Path) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row[0].strip() for row in reader if row and row[0].strip() != "case_id"]


def build_case_dicts(data_dir: Path, case_ids: List[str]) -> List[Dict[str, str]]:
    data_list = []
    for cid in case_ids:
        img = data_dir / f"imagesTr/{cid}.nii.gz"
        lbl = data_dir / f"labelsTr/{cid}.nii.gz"
        if img.is_file() and lbl.is_file():
            data_list.append({"image": str(img), "label": str(lbl), "case_id": cid})
    return data_list


def create_model(device: torch.device) -> nn.Module:
    """
    Build SegResNet 3D model for 4-channel MRI input and 3 multi-label outputs (WT, TC, ET).
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    return model.to(device)


def compute_volumes(mask_data: np.ndarray, voxel_vol_mm3: float) -> Dict[str, float]:
    """
    Calculates WT, TC, ET volumes in mL and mm^3 from a discrete label mask.
    Labels: 1 = edema, 2 = non-enhancing, 3 = enhancing.
    """
    wt_vox = int(np.sum(mask_data > 0))
    tc_vox = int(np.sum((mask_data == 2) | (mask_data == 3)))
    et_vox = int(np.sum(mask_data == 3))

    return {
        "WT_voxels": wt_vox,
        "WT_volume_ml": round(wt_vox * voxel_vol_mm3 / 1000.0, 3),
        "TC_voxels": tc_vox,
        "TC_volume_ml": round(tc_vox * voxel_vol_mm3 / 1000.0, 3),
        "ET_voxels": et_vox,
        "ET_volume_ml": round(et_vox * voxel_vol_mm3 / 1000.0, 3),
    }


def predict_and_export_case(
    model: nn.Module,
    case_dict: Dict[str, str],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Runs full inference on a single case, inverts mask back to original space,
    saves mask.nii.gz, and computes clinical metrics.
    """
    model.eval()
    cid = case_dict["case_id"]
    t0 = time.perf_counter()

    # Load and transform using invertible spatial pipeline
    spatial_pipe = get_spatial_pipeline(allow_missing_keys=True)
    val_pipe = get_validation_pipeline(to_multilabel=True)

    data_item = {"image": case_dict["image"], "label": case_dict["label"]}
    spatial_res = spatial_pipe(data_item)
    processed = val_pipe(data_item)

    image_tensor = processed["image"].unsqueeze(0).to(device)  # [1, 4, H, W, D]
    target_tensor = processed["label"].unsqueeze(0).to(device) # [1, 3, H, W, D]

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float()

    # Calculate Dice on transformed space
    dice_metric = DiceMetric(include_background=True, reduction="none")
    dice_metric(y_pred=pred_binary, y=target_tensor)
    dices = dice_metric.aggregate().squeeze(0).cpu().numpy()  # [WT, TC, ET]
    dice_wt = float(dices[0]) if not np.isnan(dices[0]) else 1.0
    dice_tc = float(dices[1]) if not np.isnan(dices[1]) else 1.0
    dice_et = float(dices[2]) if not np.isnan(dices[2]) else 1.0
    mean_dice = float(np.mean([dice_wt, dice_tc, dice_et]))

    # Convert binary [3, H, W, D] -> multiclass [1, H, W, D] {0,1,2,3}
    pred_multiclass = convert_multilabel_to_multiclass(pred_binary[0].cpu())
    pred_tensor = torch.as_tensor(pred_multiclass).unsqueeze(0)  # [1, H, W, D]

    # Invert back to patient space
    from monai.data import MetaTensor
    lbl_ref = spatial_res["label"]
    meta = lbl_ref.meta if hasattr(lbl_ref, "meta") else {}
    applied_ops = getattr(lbl_ref, "applied_operations", None)
    pred_meta = MetaTensor(pred_tensor, meta=meta, applied_operations=applied_ops)

    inverted = spatial_pipe.inverse({"label": pred_meta})
    inv_mask = inverted["label"][0].cpu().numpy().astype(np.uint8)

    # Load original header & affine
    orig_nii = nib.load(case_dict["image"])
    voxel_vol = float(np.prod(orig_nii.header.get_zooms()[:3]))
    volumes = compute_volumes(inv_mask, voxel_vol)

    # Save mask.nii.gz
    case_out_dir = output_dir / cid
    case_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_path = case_out_dir / "mask.nii.gz"
    out_nii = nib.Nifti1Image(inv_mask, orig_nii.affine, orig_nii.header)
    nib.save(out_nii, str(mask_out_path))

    inference_time = time.perf_counter() - t0

    result = {
        "case_id": cid,
        "dice": {
            "Whole_Tumor_WT": round(dice_wt, 4),
            "Tumor_Core_TC": round(dice_tc, 4),
            "Enhancing_Tumor_ET": round(dice_et, 4),
            "Mean_Dice": round(mean_dice, 4),
        },
        "volumes": volumes,
        "inference_time_seconds": round(inference_time, 2),
        "exported_mask": str(mask_out_path.resolve()),
    }
    return result


def train_baseline(
    data_dir: Path,
    splits_dir: Path,
    output_checkpoint: Path,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Runs baseline training on the official train split, validating on val split.
    """
    train_ids = load_split_case_ids(splits_dir / "train_ids.csv")
    val_ids = load_split_case_ids(splits_dir / "val_ids.csv")

    train_files = build_case_dicts(data_dir, train_ids)
    val_files = build_case_dicts(data_dir, val_ids)

    print("=" * 65)
    print("  TRAINING SEGRESNET BASELINE")
    print("=" * 65)
    print(f"  Device        : {device}")
    print(f"  Train cases   : {len(train_files)}")
    print(f"  Val cases     : {len(val_files)}")
    print(f"  Epochs        : {epochs}")
    print(f"  Learning Rate : {learning_rate}")
    print("=" * 65)

    # Pipeline with patch crop for memory safety (128x128x128)
    train_transforms = get_training_pipeline(crop_size=(128, 128, 128), to_multilabel=True)
    val_transforms = get_validation_pipeline(to_multilabel=True)

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = create_model(device)
    loss_function = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_dice = -1.0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.perf_counter()

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0 or step == len(train_loader):
                print(f"  [Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(step, 1)
        epoch_time = time.perf_counter() - t0
        print(f"  >>> Epoch {epoch} Done in {epoch_time:.1f}s | Avg Train Loss: {avg_loss:.4f}")

        # Quick validation on first 5 val cases per epoch
        model.eval()
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        with torch.no_grad():
            for val_case in val_files[:5]:
                v_data = val_transforms(val_case)
                v_img = v_data["image"].unsqueeze(0).to(device)
                v_logits = sliding_window_inference(v_img, (128, 128, 128), 1, model, overlap=0.5)
                v_out = (torch.sigmoid(v_logits) > 0.5).float()
                dice_metric(y_pred=v_out, y=v_lbl)
        val_dice = float(dice_metric.aggregate().item())
        print(f"  >>> Val Mean Dice (sample): {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "model_name": "SegResNet_3D_4Ch",
                },
                str(output_checkpoint),
            )
            print(f"  [SAVED] Best model saved -> {output_checkpoint.name}")

    print(f"\n[SUCCESS] Training complete. Best Val Dice: {best_val_dice:.4f}")


def evaluate_split(
    data_dir: Path,
    splits_dir: Path,
    checkpoint_path: Path,
    split_name: str = "val",
    output_dir: Path = _SCRIPT_DIR / "outputs" / "segmentation",
    device: torch.device = torch.device("cpu"),
    num_cases: int | None = 5,
) -> None:
    """
    Evaluates model on cases from specified split, exports masks and logs metrics.
    """
    split_csv = splits_dir / f"{split_name}_ids.csv"
    case_ids = load_split_case_ids(split_csv)
    if num_cases is not None:
        case_ids = case_ids[:num_cases]

    case_dicts = build_case_dicts(data_dir, case_ids)

    model = create_model(device)
    if checkpoint_path.is_file():
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK] Loaded checkpoint: {checkpoint_path.name} (Val Dice: {ckpt.get('val_dice', 'N/A')})")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}; running with initialized weights.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print(f"  EVALUATING ON {len(case_dicts)} CASES FROM {split_name.upper()} SPLIT")
    print("=" * 65)

    all_results = []
    wt_dices, tc_dices, et_dices = [], [], []

    for i, cdict in enumerate(case_dicts, start=1):
        res = predict_and_export_case(model, cdict, device, output_dir)
        all_results.append(res)
        d = res["dice"]
        wt_dices.append(d["Whole_Tumor_WT"])
        tc_dices.append(d["Tumor_Core_TC"])
        et_dices.append(d["Enhancing_Tumor_ET"])
        v = res["volumes"]
        print(
            f"  [{i:>2}/{len(case_dicts)}] {res['case_id']}: "
            f"Dice [WT={d['Whole_Tumor_WT']:.3f}, TC={d['Tumor_Core_TC']:.3f}, ET={d['Enhancing_Tumor_ET']:.3f}] | "
            f"WT_vol={v['WT_volume_ml']} mL | Time={res['inference_time_seconds']}s"
        )

    summary = {
        "split": split_name,
        "cases_evaluated": len(case_dicts),
        "mean_dice_WT": round(float(np.mean(wt_dices)), 4),
        "mean_dice_TC": round(float(np.mean(tc_dices)), 4),
        "mean_dice_ET": round(float(np.mean(et_dices)), 4),
        "mean_overall_dice": round(float(np.mean([np.mean(wt_dices), np.mean(tc_dices), np.mean(et_dices)])), 4),
        "average_inference_time_sec": round(float(np.mean([r["inference_time_seconds"] for r in all_results])), 2),
    }

    full_report = {
        "summary": summary,
        "cases": all_results,
    }

    report_path = _SCRIPT_DIR / "segmentation_validation_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print()
    print("=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Cases evaluated : {summary['cases_evaluated']}")
    print(f"  Mean WT Dice    : {summary['mean_dice_WT']:.4f}")
    print(f"  Mean TC Dice    : {summary['mean_dice_TC']:.4f}")
    print(f"  Mean ET Dice    : {summary['mean_dice_ET']:.4f}")
    print(f"  Overall Dice    : {summary['mean_overall_dice']:.4f}")
    print(f"  Avg Time / Case : {summary['average_inference_time_sec']:.2f}s")
    print(f"  Saved Report    : {report_path.name}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="3D Glioma Segmentation Baseline")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Task01_BrainTumour root")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_DIR, help="Path to splits directory")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train", action="store_true", help="Run baseline training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on validation split")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of cases to evaluate (default: 5)")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.train:
        train_baseline(
            data_dir=args.data.resolve(),
            splits_dir=args.splits.resolve(),
            output_checkpoint=args.checkpoint.resolve(),
            epochs=args.epochs,
            device=device,
        )

    # Run evaluation if requested or if neither train nor evaluate explicitly provided
    if args.evaluate or not args.train:
        evaluate_split(
            data_dir=args.data.resolve(),
            splits_dir=args.splits.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            split_name="val",
            device=device,
            num_cases=args.num_cases,
        )


if __name__ == "__main__":
    main()
