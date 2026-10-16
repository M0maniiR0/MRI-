#!/usr/bin/env python3
"""
segmentation_baseline.py
========================
Upgraded 3D Glioma Tumor Segmentation & Full-Volume Training Engine (SegResNet / MONAI).
BrainAgenticthon 2026 - Medical AI Workstream (Abdulrahman & Rou'a).

Final Training-Readiness Scope:
-------------------------------
1. Checkpoint Selection:
   - Evaluates full validation cohort (all 73 validation cases by default) using sliding-window inference.
2. Spatial Inversion:
   - Restores discrete masks {0, 1, 2, 3} to native patient space using strictly nearest-neighbor interpolation.
   - Validates shape, affine, and discrete label set upon saving and reloading.
3. Strict Dataset Guards:
   - Fails loudly with FileNotFoundError if any requested train or val case is missing on disk.
   - Zero-leakage guard strictly verifies zero overlap with the 72 holdout test cases.
4. Transparent HD95 Metric:
   - Proper surface-distance 95th percentile Hausdorff Distance in physical millimeters.
   - Documented empty-mask policy (0.0mm if both empty, diagonal penalty if one empty).
   - Technical failures on non-empty masks fail explicitly without disguise.
5. Deterministic Reproducibility:
   - Centralized seeds for Python random, NumPy, PyTorch, and MONAI.
6. Comprehensive Checkpoint Metadata:
   - Saves architecture, preprocessing configuration, hyperparameters, epoch, loss, and full validation metrics.
7. Safe Resume-From-Checkpoint:
   - Safely resumes model state, optimizer state, epoch counter, and best validation score from disk.
8. Dynamic Stochastic Augmentation:
   - Generates fresh tumor-positive / background crops with stochastic augmentations on every epoch.
9. Artifact Preservation:
   - Preserves baseline checkpoint (models/baseline_5ep_best_model.pt) and baseline reports.
10. Input Isolation:
   - Predictions use MRI inputs only. Ground-truth masks are strictly excluded from inference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Suppress benign MONAI deprecation messages for clean clinical logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance
from monai.networks.nets import SegResNet
from monai.utils import set_determinism

# Import verified preprocessing pipeline
from monai_preprocessing_pipeline import (
    MODALITIES,
    ConvertToBraTSMultiLabeld,
    convert_multilabel_to_multiclass,
    get_spatial_pipeline,
    get_training_pipeline,
    get_validation_pipeline,
)

# ──────────────────────────────────────────────────────────────
# Directory Configurations
# ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = _SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"
DEFAULT_SPLITS_DIR = _SCRIPT_DIR / "splits"
DEFAULT_CHECKPOINT = _SCRIPT_DIR / "models" / "best_model.pt"
BASELINE_PRESERVED_CHECKPOINT = _SCRIPT_DIR / "models" / "baseline_5ep_best_model.pt"
DEFAULT_OUTPUT_CHECKPOINT = _SCRIPT_DIR / "models" / "segresnet_fullvolume_best.pt"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "segmentation_full_volume"
DEFAULT_REPORT_PATH = _SCRIPT_DIR / "reports" / "full_volume_evaluation_metrics.json"

# Empty Mask Policy Documentation
EMPTY_MASK_POLICY_DOC = {
    "case_both_empty": "0.0 mm (both prediction and ground truth contain 0 voxels; true negative boundary alignment)",
    "case_one_empty": "Volume Euclidean diagonal penalty in mm: sqrt(sum((dim_i * spacing_i)^2)) (complete false positive or false negative)",
    "case_both_nonempty": "95th percentile Euclidean surface distance in physical millimeters computed via MONAI compute_hausdorff_distance",
    "error_handling": "Technical calculation failures on non-empty masks raise explicit RuntimeError and are never disguised as penalties",
}


# ──────────────────────────────────────────────────────────────
# Deterministic Seeding
# ──────────────────────────────────────────────────────────────
def set_deterministic_seed(seed: int = 42) -> None:
    """
    Sets seeds across Python random, NumPy, PyTorch, and MONAI for deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)


# ──────────────────────────────────────────────────────────────
# Split Loading & Zero-Leakage Dataset Guards
# ──────────────────────────────────────────────────────────────
def load_split_case_ids(csv_path: Path) -> List[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required split file missing: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row[0].strip() for row in reader if row and row[0].strip() != "case_id"]


def enforce_zero_leakage_guard(splits_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Validates split counts and strictly ensures train/val sets never intersect with test set.
    """
    train_ids = load_split_case_ids(splits_dir / "train_ids.csv")
    val_ids = load_split_case_ids(splits_dir / "val_ids.csv")
    test_ids = load_split_case_ids(splits_dir / "test_ids.csv")

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    # Hard assertions
    assert len(train_set & test_set) == 0, f"DATA LEAKAGE: {len(train_set & test_set)} train cases overlap with test set!"
    assert len(val_set & test_set) == 0, f"DATA LEAKAGE: {len(val_set & test_set)} val cases overlap with test set!"
    assert len(train_set & val_set) == 0, f"DATA OVERLAP: {len(train_set & val_set)} train cases overlap with val set!"
    assert len(test_set) == 72, f"Expected 72 holdout test cases, found {len(test_set)}"
    assert len(train_set) == 339, f"Expected 339 train cases, found {len(train_set)}"
    assert len(val_set) == 73, f"Expected 73 val cases, found {len(val_set)}"

    return train_ids, val_ids, test_ids


def build_case_dicts(data_dir: Path, case_ids: List[str], require_label: bool = True) -> List[Dict[str, str]]:
    """
    Builds data dictionaries for requested case IDs.
    Fails loudly if any requested file is missing from disk.
    """
    data_list = []
    missing_files = []

    for cid in case_ids:
        img = data_dir / f"imagesTr/{cid}.nii.gz"
        lbl = data_dir / f"labelsTr/{cid}.nii.gz"

        if not img.is_file():
            missing_files.append(f"Image missing for {cid}: {img}")
        if require_label and not lbl.is_file():
            missing_files.append(f"Label missing for {cid}: {lbl}")

        if img.is_file() and (not require_label or lbl.is_file()):
            item = {"image": str(img), "case_id": cid}
            if lbl.is_file():
                item["label"] = str(lbl)
            data_list.append(item)

    if missing_files:
        error_msg = f"CRITICAL: {len(missing_files)} requested files are missing on disk:\n" + "\n".join(missing_files[:10])
        if len(missing_files) > 10:
            error_msg += f"\n... and {len(missing_files) - 10} more."
        raise FileNotFoundError(error_msg)

    return data_list


# ──────────────────────────────────────────────────────────────
# Model Creation & Dynamic Checkpoint Loading
# ──────────────────────────────────────────────────────────────
def create_model(
    device: torch.device,
    init_filters: int = 8,
    blocks_down: Tuple[int, ...] = (1, 2, 2),
    blocks_up: Tuple[int, ...] = (1, 1),
    dropout_prob: float = 0.2,
) -> SegResNet:
    """
    Creates a 3D SegResNet for 4-channel MRI input and 3 multi-label sigmoid outputs (WT, TC, ET).
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=init_filters,
        blocks_down=list(blocks_down),
        blocks_up=list(blocks_up),
        dropout_prob=dropout_prob,
    )
    return model.to(device)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[SegResNet, Dict[str, Any]]:
    """
    Reloads model weights strictly from disk, auto-detecting SegResNet layer architecture.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found on disk at: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Detect init_filters from first conv layer
    first_conv_weight = state_dict.get("convInit.conv.weight")
    if first_conv_weight is not None:
        init_filters = first_conv_weight.shape[0]
    else:
        init_filters = 8

    # Detect blocks_up count
    up_layer_keys = [k for k in state_dict if "up_layers" in k and "conv1.conv.weight" in k]
    num_up = len(up_layer_keys) if up_layer_keys else 2

    if num_up == 3:
        blocks_down = (1, 2, 2, 4)
        blocks_up = (1, 1, 1)
    else:
        blocks_down = (1, 2, 2)
        blocks_up = (1, 1)

    model = create_model(
        device=device,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=0.0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    meta_info = {
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "epoch": ckpt.get("epoch", "N/A"),
        "training_loss": ckpt.get("training_loss", "N/A"),
        "val_mean_dice": ckpt.get("val_mean_dice", "N/A"),
        "optimizer": ckpt.get("optimizer", "N/A"),
        "learning_rate": ckpt.get("learning_rate", "N/A"),
        "init_filters": init_filters,
        "blocks_down": list(blocks_down),
        "blocks_up": list(blocks_up),
        "timestamp": ckpt.get("timestamp", "N/A"),
        "raw_checkpoint": ckpt,
    }
    return model, meta_info


# ──────────────────────────────────────────────────────────────
# Metric Computation: Full-Volume Dice & Explicit HD95
# ──────────────────────────────────────────────────────────────
def compute_region_dice(pred_binary: torch.Tensor, gt_binary: torch.Tensor) -> float:
    """
    Calculates Dice score with robust True Negative handling:
    - Both empty -> 1.0 (perfect True Negative)
    - One empty, one non-empty -> 0.0 (missed region or false alarm)
    - Both non-empty -> standard 2 * |P & G| / (|P| + |G|)
    """
    p_sum = float(pred_binary.sum().item())
    g_sum = float(gt_binary.sum().item())

    if p_sum == 0.0 and g_sum == 0.0:
        return 1.0
    if p_sum == 0.0 or g_sum == 0.0:
        return 0.0

    intersection = float((pred_binary * gt_binary).sum().item())
    dice = (2.0 * intersection) / (p_sum + g_sum)
    return round(float(dice), 4)


def compute_region_hd95(
    pred_binary: torch.Tensor,
    gt_binary: torch.Tensor,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_shape: Tuple[int, int, int] = (240, 240, 155),
) -> Tuple[float, bool]:
    """
    Computes 95th percentile surface-distance Hausdorff Distance in physical mm:
    - Both empty: 0.0 mm (empty_penalty = False, true negative)
    - One empty, one non-empty: volume diagonal penalty in mm (empty_penalty = True)
    - Both non-empty: MONAI surface-distance HD95 in physical mm.
      Technical calculation errors on non-empty masks are NOT disguised as empty-mask penalties;
      they raise a descriptive RuntimeError.
    """
    p_sum = int(pred_binary.sum().item())
    g_sum = int(gt_binary.sum().item())

    diag_penalty_mm = float(np.sqrt(np.sum((np.array(spatial_shape) * np.array(voxel_spacing)) ** 2)))

    if p_sum == 0 and g_sum == 0:
        return 0.0, False
    if p_sum == 0 or g_sum == 0:
        return round(diag_penalty_mm, 2), True

    # Both masks are non-empty: calculate real surface distance
    p_5d = pred_binary.unsqueeze(0).unsqueeze(0).float()
    g_5d = gt_binary.unsqueeze(0).unsqueeze(0).float()

    try:
        val = compute_hausdorff_distance(
            y_pred=p_5d,
            y=g_5d,
            include_background=True,
            percentile=95.0,
            spacing=voxel_spacing,
        )
        res = float(val.item())
        if np.isnan(res) or np.isinf(res):
            raise RuntimeError(
                f"HD95 calculation returned non-finite value ({res}) for non-empty masks "
                f"(pred_voxels={p_sum}, gt_voxels={g_sum})"
            )
        return round(res, 2), False
    except Exception as e:
        if isinstance(e, RuntimeError) and "non-finite" in str(e):
            raise
        raise RuntimeError(
            f"HD95 technical computation error on non-empty masks (pred_voxels={p_sum}, gt_voxels={g_sum}): {e}"
        ) from e


def compute_clinical_volumes(mask_array: np.ndarray, voxel_vol_mm3: float) -> Dict[str, Any]:
    """
    Calculates WT, TC, ET volumes in mL and voxel counts from discrete mask {0:bg, 1:edema, 2:net, 3:et}.
    """
    wt_vox = int(np.sum(mask_array > 0))
    tc_vox = int(np.sum((mask_array == 2) | (mask_array == 3)))
    et_vox = int(np.sum(mask_array == 3))

    return {
        "WT_voxels": wt_vox,
        "WT_volume_ml": round(wt_vox * voxel_vol_mm3 / 1000.0, 3),
        "TC_voxels": tc_vox,
        "TC_volume_ml": round(tc_vox * voxel_vol_mm3 / 1000.0, 3),
        "ET_voxels": et_vox,
        "ET_volume_ml": round(et_vox * voxel_vol_mm3 / 1000.0, 3),
    }


# ──────────────────────────────────────────────────────────────
# Spatial Inversion with Nearest-Neighbor Interpolation
# ──────────────────────────────────────────────────────────────
def invert_mask_to_native_space(
    pred_multiclass: np.ndarray,
    img_meta: Dict[str, Any],
    applied_operations: List[Dict[str, Any]],
    spatial_pipe: Any,
) -> np.ndarray:
    """
    Inverts predicted discrete multi-class mask {0, 1, 2, 3} back to original patient voxel grid
    using strictly nearest-neighbor interpolation.
    """
    pred_tensor = torch.as_tensor(pred_multiclass, dtype=torch.float32).unsqueeze(0)  # [1, H, W, D]

    # Clone applied operations and force nearest-neighbor interpolation for spatial resampling
    ops = copy.deepcopy(applied_operations) if applied_operations is not None else []
    for op in ops:
        if op.get("class") == "SpatialResample":
            if "extra_info" in op:
                op["extra_info"]["mode"] = "nearest"

    pred_meta = MetaTensor(pred_tensor, meta=img_meta, applied_operations=ops)
    inverted = spatial_pipe.inverse({"image": pred_meta})
    inv_arr = inverted["image"][0].cpu().numpy().round().astype(np.uint8)

    # Safety check on discrete label values
    unique_vals = set(np.unique(inv_arr))
    if not unique_vals.issubset({0, 1, 2, 3}):
        raise ValueError(f"Inverted mask contains invalid discrete class labels: {unique_vals}")

    return inv_arr


# ──────────────────────────────────────────────────────────────
# Full-Volume Evaluation on Single Case
# ──────────────────────────────────────────────────────────────
def evaluate_single_case_full_volume(
    model: SegResNet,
    case_dict: Dict[str, str],
    device: torch.device,
    output_dir: Optional[Path] = None,
    sw_roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_overlap: float = 0.25,
    sw_batch_size: int = 4,
    save_nifti: bool = True,
) -> Dict[str, Any]:
    """
    Runs full-volume 3D sliding window inference with strict input isolation:
    1. Loads 4-channel MRI input only.
    2. Runs model inference (zero access to ground truth).
    3. Reconstructs multi-class prediction and inverts back to native space via nearest-neighbor interpolation.
    4. Evaluates WT, TC, ET Dice and HD95 against ground truth.
    5. Validates exported NIfTI shape, affine, and discrete labels upon reload.
    6. Measures granular wall-clock timing.
    """
    cid = case_dict["case_id"]
    t_start = time.perf_counter()

    # Pipelines
    spatial_pipe = get_spatial_pipeline(allow_missing_keys=True)
    val_pipe = get_validation_pipeline(to_multilabel=True)

    # 1. Preprocessing (Input MRI only for forward inference)
    t0_pre = time.perf_counter()
    has_label = "label" in case_dict and os.path.isfile(case_dict["label"])
    data_item = {"image": case_dict["image"]}
    if has_label:
        data_item["label"] = case_dict["label"]

    spatial_res = spatial_pipe(data_item)
    processed = val_pipe(data_item)
    t_pre = time.perf_counter() - t0_pre

    # Strict isolation: model sees only image_tensor
    image_tensor = processed["image"].unsqueeze(0).to(device)  # [1, 4, H, W, D]
    target_tensor = processed.get("label")                     # [3, H, W, D] or None

    # 2. Sliding Window Inference
    t0_inf = time.perf_counter()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=sw_roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )
        probs = torch.sigmoid(logits).squeeze(0)  # [3, H, W, D]
        pred_binary = (probs > 0.5).float()       # [3, H, W, D]
    t_inf = time.perf_counter() - t0_inf

    # 3. Dice & HD95 Calculation on Full Volume (if label exists)
    dice_wt, dice_tc, dice_et, mean_dice = 0.0, 0.0, 0.0, 0.0
    hd95_wt, hd95_tc, hd95_et, mean_hd95 = 0.0, 0.0, 0.0, 0.0
    pen_wt, pen_tc, pen_et = False, False, False
    gt_wt_vox, gt_tc_vox, gt_et_vox = 0, 0, 0

    if target_tensor is not None:
        spatial_shape = tuple(target_tensor.shape[1:])
        voxel_spacing = (1.0, 1.0, 1.0)

        dice_wt = compute_region_dice(pred_binary[0], target_tensor[0])
        dice_tc = compute_region_dice(pred_binary[1], target_tensor[1])
        dice_et = compute_region_dice(pred_binary[2], target_tensor[2])
        mean_dice = round(float(np.mean([dice_wt, dice_tc, dice_et])), 4)

        hd95_wt, pen_wt = compute_region_hd95(pred_binary[0], target_tensor[0], voxel_spacing, spatial_shape)
        hd95_tc, pen_tc = compute_region_hd95(pred_binary[1], target_tensor[1], voxel_spacing, spatial_shape)
        hd95_et, pen_et = compute_region_hd95(pred_binary[2], target_tensor[2], voxel_spacing, spatial_shape)
        mean_hd95 = round(float(np.mean([hd95_wt, hd95_tc, hd95_et])), 2)

        gt_wt_vox = int(target_tensor[0].sum().item())
        gt_tc_vox = int(target_tensor[1].sum().item())
        gt_et_vox = int(target_tensor[2].sum().item())

    pred_wt_vox = int(pred_binary[0].sum().item())
    pred_tc_vox = int(pred_binary[1].sum().item())
    pred_et_vox = int(pred_binary[2].sum().item())

    empty_status = {
        "WT_gt_empty": gt_wt_vox == 0,
        "WT_pred_empty": pred_wt_vox == 0,
        "TC_gt_empty": gt_tc_vox == 0,
        "TC_pred_empty": pred_tc_vox == 0,
        "ET_gt_empty": gt_et_vox == 0,
        "ET_pred_empty": pred_et_vox == 0,
        "penalties_applied": {"WT": pen_wt, "TC": pen_tc, "ET": pen_et},
    }

    # 4. Nearest-Neighbor Inversion to Native Patient Space & NIfTI Export
    t0_post = time.perf_counter()
    pred_multiclass = convert_multilabel_to_multiclass(pred_binary.cpu())

    img_ref = spatial_res["image"]
    meta = getattr(img_ref, "meta", {})
    applied_ops = getattr(img_ref, "applied_operations", None)

    inv_mask = invert_mask_to_native_space(
        pred_multiclass=pred_multiclass,
        img_meta=meta,
        applied_operations=applied_ops,
        spatial_pipe=spatial_pipe,
    )

    orig_nii = nib.load(case_dict["image"])
    voxel_vol = float(np.prod(orig_nii.header.get_zooms()[:3]))
    pred_volumes = compute_clinical_volumes(inv_mask, voxel_vol)

    exported_mask_path = None
    if save_nifti and output_dir is not None:
        case_dir = output_dir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        mask_out = case_dir / "mask.nii.gz"
        out_nii = nib.Nifti1Image(inv_mask, orig_nii.affine, orig_nii.header)
        nib.save(out_nii, str(mask_out))
        exported_mask_path = str(mask_out.resolve())

        # Verify saved NIfTI by reloading from disk
        reloaded = nib.load(str(mask_out))
        assert reloaded.shape == orig_nii.shape[:3], f"Shape mismatch: {reloaded.shape} vs {orig_nii.shape[:3]}"
        assert np.allclose(reloaded.affine, orig_nii.affine), "Affine matrix mismatch after reloading mask!"
        reloaded_vals = set(np.unique(reloaded.get_fdata()))
        assert reloaded_vals.issubset({0, 1, 2, 3}), f"Invalid values in reloaded NIfTI mask: {reloaded_vals}"

    t_post = time.perf_counter() - t0_post
    t_total = time.perf_counter() - t_start

    return {
        "case_id": cid,
        "status": "SUCCESS",
        "dice": {
            "Whole_Tumor_WT": dice_wt,
            "Tumor_Core_TC": dice_tc,
            "Enhancing_Tumor_ET": dice_et,
            "Mean_Dice": mean_dice,
        },
        "hd95_mm": {
            "Whole_Tumor_WT": hd95_wt,
            "Tumor_Core_TC": hd95_tc,
            "Enhancing_Tumor_ET": hd95_et,
            "Mean_HD95": mean_hd95,
        },
        "empty_status": empty_status,
        "predicted_volumes": pred_volumes,
        "ground_truth_voxels": {
            "WT": gt_wt_vox,
            "TC": gt_tc_vox,
            "ET": gt_et_vox,
        },
        "timing_seconds": {
            "preprocessing": round(t_pre, 3),
            "sliding_window_inference": round(t_inf, 3),
            "inversion_and_nifti_save": round(t_post, 3),
            "end_to_end_total": round(t_total, 3),
        },
        "exported_mask": exported_mask_path,
    }


# ──────────────────────────────────────────────────────────────
# Cohort Evaluation Engine
# ──────────────────────────────────────────────────────────────
def evaluate_cohort(
    data_dir: Path,
    splits_dir: Path,
    checkpoint_path: Path,
    num_cases: int = 73,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    sw_roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_overlap: float = 0.25,
    sw_batch_size: int = 4,
    save_nifti: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full-volume evaluation across validation cases with metric aggregation.
    """
    set_deterministic_seed(seed)
    train_ids, val_ids, test_ids = enforce_zero_leakage_guard(splits_dir)

    target_val_ids = val_ids if num_cases <= 0 else val_ids[:num_cases]
    case_dicts = build_case_dicts(data_dir, target_val_ids, require_label=True)

    print("=" * 75)
    print("  MONAI SEGRESNET 3D: FULL-VOLUME SLIDING-WINDOW EVALUATION")
    print("=" * 75)
    print(f"  Checkpoint source : {checkpoint_path}")
    print(f"  Device            : {device}")
    print(f"  Validation cases  : {len(case_dicts)} (out of {len(val_ids)} available)")
    print(f"  Holdout test set  : {len(test_ids)} cases (STRICTLY EXCLUDED)")
    print(f"  Sliding ROI size  : {sw_roi_size}")
    print(f"  Sliding Overlap   : {sw_overlap}")
    print(f"  Batch size        : {sw_batch_size}")
    print(f"  Deterministic Seed: {seed}")
    print("=" * 75)

    # Reload model weights strictly from disk
    model, ckpt_meta = load_model_from_checkpoint(checkpoint_path, device)
    print(f"[OK] Checkpoint verified & loaded from disk: {ckpt_meta['checkpoint_file']}")
    print(f"     Epoch: {ckpt_meta['epoch']} | Loss: {ckpt_meta['training_loss']} | Filters: {ckpt_meta['init_filters']}")
    print("-" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    wt_dices, tc_dices, et_dices, mean_dices = [], [], [], []
    wt_hd95s, tc_hd95s, et_hd95s = [], [], []
    wt_hd95s_valid, tc_hd95s_valid, et_hd95s_valid = [], [], []
    e2e_times = []

    failures = []

    for i, cdict in enumerate(case_dicts, start=1):
        cid = cdict["case_id"]
        try:
            res = evaluate_single_case_full_volume(
                model=model,
                case_dict=cdict,
                device=device,
                output_dir=output_dir,
                sw_roi_size=sw_roi_size,
                sw_overlap=sw_overlap,
                sw_batch_size=sw_batch_size,
                save_nifti=save_nifti,
            )
            results.append(res)

            d = res["dice"]
            h = res["hd95_mm"]
            t = res["timing_seconds"]
            p = res["empty_status"]["penalties_applied"]

            wt_dices.append(d["Whole_Tumor_WT"])
            tc_dices.append(d["Tumor_Core_TC"])
            et_dices.append(d["Enhancing_Tumor_ET"])
            mean_dices.append(d["Mean_Dice"])

            wt_hd95s.append(h["Whole_Tumor_WT"])
            tc_hd95s.append(h["Tumor_Core_TC"])
            et_hd95s.append(h["Enhancing_Tumor_ET"])

            if not p["WT"]:
                wt_hd95s_valid.append(h["Whole_Tumor_WT"])
            if not p["TC"]:
                tc_hd95s_valid.append(h["Tumor_Core_TC"])
            if not p["ET"]:
                et_hd95s_valid.append(h["Enhancing_Tumor_ET"])

            e2e_times.append(t["end_to_end_total"])

            print(
                f"  [{i:>2}/{len(case_dicts)}] {cid} -> "
                f"Dice: [WT={d['Whole_Tumor_WT']:.4f}, TC={d['Tumor_Core_TC']:.4f}, ET={d['Enhancing_Tumor_ET']:.4f}] | "
                f"HD95: [WT={h['Whole_Tumor_WT']:.1f}mm, TC={h['Tumor_Core_TC']:.1f}mm, ET={h['Enhancing_Tumor_ET']:.1f}mm] | "
                f"E2E={t['end_to_end_total']}s"
            )

        except Exception as e:
            fail_record = {"case_id": cid, "error": str(e)}
            failures.append(fail_record)
            print(f"  [{i:>2}/{len(case_dicts)}] {cid} -> FAILED: {e}")

    # Aggregate Statistics
    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_evaluated": {
            "checkpoint_file": ckpt_meta["checkpoint_file"],
            "epoch": ckpt_meta["epoch"],
            "training_loss": ckpt_meta["training_loss"],
            "init_filters": ckpt_meta["init_filters"],
            "blocks_down": ckpt_meta["blocks_down"],
            "blocks_up": ckpt_meta["blocks_up"],
        },
        "cases_requested": len(case_dicts),
        "cases_evaluated_successfully": len(results),
        "cases_failed": len(failures),
        "dice_metrics": {
            "Whole_Tumor_WT": {
                "mean": round(float(np.mean(wt_dices)), 4) if wt_dices else 0.0,
                "median": round(float(np.median(wt_dices)), 4) if wt_dices else 0.0,
                "std": round(float(np.std(wt_dices)), 4) if wt_dices else 0.0,
            },
            "Tumor_Core_TC": {
                "mean": round(float(np.mean(tc_dices)), 4) if tc_dices else 0.0,
                "median": round(float(np.median(tc_dices)), 4) if tc_dices else 0.0,
                "std": round(float(np.std(tc_dices)), 4) if tc_dices else 0.0,
            },
            "Enhancing_Tumor_ET": {
                "mean": round(float(np.mean(et_dices)), 4) if et_dices else 0.0,
                "median": round(float(np.median(et_dices)), 4) if et_dices else 0.0,
                "std": round(float(np.std(et_dices)), 4) if et_dices else 0.0,
            },
            "Overall_Mean_Dice": round(float(np.mean(mean_dices)), 4) if mean_dices else 0.0,
        },
        "hd95_metrics_mm": {
            "with_penalty": {
                "Whole_Tumor_WT_mean": round(float(np.mean(wt_hd95s)), 2) if wt_hd95s else 0.0,
                "Tumor_Core_TC_mean": round(float(np.mean(tc_hd95s)), 2) if tc_hd95s else 0.0,
                "Enhancing_Tumor_ET_mean": round(float(np.mean(et_hd95s)), 2) if et_hd95s else 0.0,
            },
            "valid_nonpenalty_only": {
                "Whole_Tumor_WT_mean": round(float(np.mean(wt_hd95s_valid)), 2) if wt_hd95s_valid else 0.0,
                "Tumor_Core_TC_mean": round(float(np.mean(tc_hd95s_valid)), 2) if tc_hd95s_valid else 0.0,
                "Enhancing_Tumor_ET_mean": round(float(np.mean(et_hd95s_valid)), 2) if et_hd95s_valid else 0.0,
                "valid_sample_counts": {
                    "WT": len(wt_hd95s_valid),
                    "TC": len(tc_hd95s_valid),
                    "ET": len(et_hd95s_valid),
                },
            },
        },
        "timing_breakdown": {
            "avg_preprocessing_sec": round(float(np.mean([r["timing_seconds"]["preprocessing"] for r in results])), 2) if results else 0.0,
            "avg_sliding_window_sec": round(float(np.mean([r["timing_seconds"]["sliding_window_inference"] for r in results])), 2) if results else 0.0,
            "avg_inversion_save_sec": round(float(np.mean([r["timing_seconds"]["inversion_and_nifti_save"] for r in results])), 2) if results else 0.0,
            "avg_end_to_end_sec": round(float(np.mean(e2e_times)), 2) if e2e_times else 0.0,
        },
        "empty_mask_policy": EMPTY_MASK_POLICY_DOC,
    }

    full_report = {
        "summary": summary,
        "per_case_results": results,
        "failures": failures,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print()
    print("=" * 75)
    print("  EVALUATION SUMMARY & EMPIRICAL RESULTS")
    print("=" * 75)
    print(f"  Cases Evaluated : {summary['cases_evaluated_successfully']} / {summary['cases_requested']}")
    print(f"  Mean WT Dice    : {summary['dice_metrics']['Whole_Tumor_WT']['mean']:.4f}")
    print(f"  Mean TC Dice    : {summary['dice_metrics']['Tumor_Core_TC']['mean']:.4f}")
    print(f"  Mean ET Dice    : {summary['dice_metrics']['Enhancing_Tumor_ET']['mean']:.4f}")
    print(f"  Overall Dice    : {summary['dice_metrics']['Overall_Mean_Dice']:.4f}")
    print(f"  Mean HD95 (WT)  : {summary['hd95_metrics_mm']['with_penalty']['Whole_Tumor_WT_mean']:.2f} mm")
    print(f"  Mean HD95 (TC)  : {summary['hd95_metrics_mm']['with_penalty']['Tumor_Core_TC_mean']:.2f} mm")
    print(f"  Mean HD95 (ET)  : {summary['hd95_metrics_mm']['with_penalty']['Enhancing_Tumor_ET_mean']:.2f} mm")
    print(f"  Avg E2E Time    : {summary['timing_breakdown']['avg_end_to_end_sec']:.2f}s per case")
    print(f"  Report Saved    : {report_path.name}")
    print("=" * 75)

    return full_report


def get_seg_training_pipeline(
    crop_size: Tuple[int, int, int] = (64, 64, 64),
    to_multilabel: bool = True,
    pos_ratio: float = 2.0,
    neg_ratio: float = 1.0,
    num_samples: int = 2,
) -> Any:
    """
    Dynamic stochastic training pipeline:
    - Balanced tumor-positive (pos_ratio) and background (neg_ratio) spatial sampling
    - Fresh stochastic augmentations: random flips along 3 axes, intensity normalization
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        NormalizeIntensityd,
        RandCropByPosNegLabeld,
        RandFlipd,
        EnsureTyped,
    )
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=crop_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "label"]),
    ]
    if to_multilabel:
        transforms.append(ConvertToBraTSMultiLabeld(keys=["label"]))
    return Compose(transforms)


# ──────────────────────────────────────────────────────────────
# Training Engine with Full-Validation Best Checkpoint Selection
# ──────────────────────────────────────────────────────────────
def train_full_volume(
    data_dir: Path,
    splits_dir: Path,
    output_checkpoint: Path,
    resume_checkpoint: Optional[Path] = None,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    val_frequency: int = 1,
    val_num_cases: int = 73,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> None:
    """
    Reproducible SegResNet training across all 339 train cases with representative sliding-window
    validation across all 73 validation cases. Strictly excludes 72 holdout cases and performs
    best-checkpoint selection with complete metadata.
    """
    set_deterministic_seed(seed)
    train_ids, val_ids, test_ids = enforce_zero_leakage_guard(splits_dir)

    print("=" * 75)
    print("  TRAINING SEGRESNET 3D WITH ZERO-LEAKAGE SEPARATION")
    print("=" * 75)
    print(f"  Train cohort      : {len(train_ids)} cases (all 339 cases)")
    print(f"  Validation cohort : {val_num_cases} cases (out of {len(val_ids)} available)")
    print(f"  Holdout test set  : {len(test_ids)} cases (strictly excluded)")
    print(f"  Epochs            : {epochs}")
    print(f"  Learning rate     : {learning_rate}")
    print(f"  Training Patch    : {patch_size}")
    print(f"  Deterministic Seed: {seed}")
    print(f"  Target Checkpoint : {output_checkpoint}")
    print("=" * 75)

    # Ensure baseline is preserved before writing anything
    if output_checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve() and not BASELINE_PRESERVED_CHECKPOINT.is_file():
        import shutil
        shutil.copy(DEFAULT_CHECKPOINT, BASELINE_PRESERVED_CHECKPOINT)
        print(f"[PRESERVED] Baseline checkpoint copied to {BASELINE_PRESERVED_CHECKPOINT.name}")

    train_files = build_case_dicts(data_dir, train_ids, require_label=True)
    val_files = build_case_dicts(data_dir, val_ids[:val_num_cases], require_label=True)

    # Stochastic training pipeline: generates fresh crops & augmentations on every pass
    from monai.data import list_data_collate
    train_pipe = get_seg_training_pipeline(
        crop_size=patch_size,
        to_multilabel=True,
        pos_ratio=2.0,
        neg_ratio=1.0,
        num_samples=2,
    )

    train_ds = Dataset(data=train_files, transform=train_pipe)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=list_data_collate)

    init_filters = 8
    blocks_down = (1, 2, 2)
    blocks_up = (1, 1)
    dropout_prob = 0.2

    model = create_model(
        device=device,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
    )
    loss_fn = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    start_epoch = 1
    best_val_dice = -1.0

    # Safe resume from checkpoint
    if resume_checkpoint is not None and resume_checkpoint.is_file():
        print(f"[RESUME] Loading weights and state from {resume_checkpoint.name}...")
        res_ckpt = torch.load(str(resume_checkpoint), map_location=device)
        model.load_state_dict(res_ckpt["model_state_dict"])
        if "optimizer_state_dict" in res_ckpt:
            optimizer.load_state_dict(res_ckpt["optimizer_state_dict"])
        start_epoch = res_ckpt.get("epoch", 0) + 1
        best_val_dice = res_ckpt.get("val_mean_dice", -1.0)
        print(f"[RESUME] Resuming at Epoch {start_epoch} (Prior Best Val Dice: {best_val_dice:.4f})")
    elif resume_checkpoint is not None:
        raise FileNotFoundError(f"Resume checkpoint specified but not found on disk: {resume_checkpoint}")

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.perf_counter()

        # Dynamic stochastic DataLoader pass
        for batch in train_loader:
            step += 1
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                print(f"  [Epoch {epoch}/{start_epoch + epochs - 1} | Step {step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(step, 1)
        epoch_dur = time.perf_counter() - t0
        print(f"  >>> Epoch {epoch} finished in {epoch_dur:.1f}s | Avg Train Loss: {avg_loss:.4f}")

        # Representative full-volume validation across validation cohort
        if epoch % val_frequency == 0:
            print(f"  Evaluating full-volume sliding window across {len(val_files)} validation cases...")
            model.eval()
            val_dices = []
            for v_case in val_files:
                v_res = evaluate_single_case_full_volume(
                    model=model,
                    case_dict=v_case,
                    device=device,
                    save_nifti=False,
                )
                val_dices.append(v_res["dice"]["Mean_Dice"])

            mean_val_dice = float(np.mean(val_dices))
            print(f"  >>> Val Mean Dice across {len(val_files)} cases: {mean_val_dice:.4f}")

            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice

                # Complete Checkpoint Metadata
                checkpoint_dict = {
                    "model_version": "1.0.0-fullvolume",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_loss": avg_loss,
                    "val_mean_dice": best_val_dice,
                    "val_cohort_size": len(val_files),
                    "architecture": {
                        "model_name": "SegResNet_3D",
                        "spatial_dims": 3,
                        "in_channels": 4,
                        "out_channels": 3,
                        "init_filters": init_filters,
                        "blocks_down": list(blocks_down),
                        "blocks_up": list(blocks_up),
                        "dropout_prob": dropout_prob,
                    },
                    "preprocessing_config": {
                        "orientation": "RAS",
                        "spacing_mm": [1.0, 1.0, 1.0],
                        "intensity_normalization": "zscore_nonzero_brain",
                        "train_crop_size": list(patch_size),
                        "sliding_window_roi": [64, 64, 64],
                        "sliding_window_overlap": 0.25,
                        "target_modalities": ["FLAIR", "T1w", "t1gd", "T2w"],
                        "labels": {"0": "background", "1": "edema", "2": "non_enhancing", "3": "enhancing"},
                        "multilabel_channels": {"0": "WT", "1": "TC", "2": "ET"},
                    },
                    "training_hyperparameters": {
                        "learning_rate": learning_rate,
                        "optimizer": "AdamW",
                        "weight_decay": 1e-5,
                        "batch_size": batch_size,
                        "loss_function": "DiceLoss_sigmoid",
                        "seed": seed,
                    },
                    "split_manifest": {
                        "train_count": len(train_ids),
                        "val_count": len(val_ids),
                        "test_count": len(test_ids),
                        "test_excluded": True,
                    },
                }

                torch.save(checkpoint_dict, str(output_checkpoint))
                print(f"  [SAVED] New best checkpoint saved -> {output_checkpoint.name} (Val Dice: {best_val_dice:.4f})")

    print(f"\n[DONE] Training complete. Best validation Dice achieved: {best_val_dice:.4f}")


# ──────────────────────────────────────────────────────────────
# Main CLI Interface
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3D Glioma Full-Volume Segmentation & Evaluation Engine")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Task01_BrainTumour root")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_DIR, help="Path to splits directory")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate", action="store_true", help="Run full-volume evaluation on validation split")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of validation cases to evaluate (default: 5, 0 or 73 for all)")
    parser.add_argument("--train", action="store_true", help="Run model training across all 339 train cases")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-num-cases", type=int, default=73, help="Number of val cases for checkpoint selection (default: 73)")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume training from existing checkpoint")
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_OUTPUT_CHECKPOINT, help="Output checkpoint for new training")
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_PATH, help="Path to save evaluation JSON report")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to save exported NIfTI masks")
    parser.add_argument("--case-id", type=str, default=None, help="Run single case offline evaluation (e.g. BRATS_002)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_deterministic_seed(args.seed)

    # 1. Single Case Test Mode
    if args.case_id:
        model, _ = load_model_from_checkpoint(args.checkpoint.resolve(), device)
        case_dict = {
            "image": str(args.data / f"imagesTr/{args.case_id}.nii.gz"),
            "label": str(args.data / f"labelsTr/{args.case_id}.nii.gz"),
            "case_id": args.case_id,
        }
        res = evaluate_single_case_full_volume(
            model=model,
            case_dict=case_dict,
            device=device,
            output_dir=args.output_dir.resolve(),
            save_nifti=True,
        )
        print(json.dumps(res, indent=2))
        return

    # 2. Training Mode
    if args.train:
        train_full_volume(
            data_dir=args.data.resolve(),
            splits_dir=args.splits.resolve(),
            output_checkpoint=args.output_checkpoint.resolve(),
            resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
            epochs=args.epochs,
            learning_rate=args.lr,
            val_num_cases=args.val_num_cases,
            device=device,
            seed=args.seed,
        )
        return

    # 3. Full-Volume Evaluation Mode (Default)
    evaluate_cohort(
        data_dir=args.data.resolve(),
        splits_dir=args.splits.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        num_cases=args.num_cases,
        device=device,
        output_dir=args.output_dir.resolve(),
        report_path=args.report_file.resolve(),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
