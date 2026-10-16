#!/usr/bin/env python3
"""
monai_preprocessing_pipeline.py
===============================
Standard MONAI Preprocessing Pipeline for MSD Task01_BrainTumour (BraTS Glioma).
Designed for BrainAgenticthon 2026 - Medical AI Workstream.

Key Design Principles & Standards:
----------------------------------
1. Channel Order (Strictly preserved, never reversed):
   - Channel 0: FLAIR  (Fluid Attenuated Inversion Recovery - edema & peritumoral)
   - Channel 1: T1w    (T1-weighted native - anatomical baseline)
   - Channel 2: T1gd   (T1 with Gadolinium contrast - enhancing tumor core)
   - Channel 3: T2w    (T2-weighted - hyperintense edema & fluid)

2. Segmentation Target Labels (MSD Task01 mapping):
   - Label 0: Background
   - Label 1: Edema (ED)
   - Label 2: Non-enhancing tumor (NET / necrosis)
   - Label 3: Enhancing tumor (ET)

3. Clinical Derived Evaluation Regions (BraTS standard):
   - Whole Tumor (WT)  : Labels 1, 2, 3 -> (mask >= 1)
   - Tumor Core  (TC)  : Labels 2, 3    -> (mask == 2) | (mask == 3)
   - Enhancing   (ET)  : Label 3        -> (mask == 3)

4. Spatial Standard:
   - Target Orientation : RAS (Right-Anterior-Superior)
   - Target Spacing     : (1.0, 1.0, 1.0) mm isotropic
   - Interpolation      : Bilinear/Spline for MRI images, Nearest-Neighbor for masks

5. Intensity Normalization:
   - Channel-wise Z-score normalization computed strictly within non-zero brain voxels.
   - Guarded against zero-variance channels (std < 1e-8).

6. Inversion Support:
   - Spatial transforms are invertible (Orientationd, Spacingd) to map model predictions
     back to the exact input coordinates without spatial drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
import torch

import monai.transforms as mt
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandSpatialCropd,
    Spacingd,
)

# ──────────────────────────────────────────────────────────────
# Metadata Constants
# ──────────────────────────────────────────────────────────────
MODALITIES: Dict[int, str] = {
    0: "FLAIR",
    1: "T1w",
    2: "t1gd",
    3: "T2w",
}

LABELS_MSD: Dict[int, str] = {
    0: "background",
    1: "edema",
    2: "non-enhancing tumor",
    3: "enhancing tumour",
}

DERIVED_REGIONS: Dict[str, str] = {
    "WT": "Whole Tumor (edema + non-enhancing + enhancing: labels 1,2,3)",
    "TC": "Tumor Core (non-enhancing + enhancing: labels 2,3)",
    "ET": "Enhancing Tumor (enhancing: label 3)",
}


# ──────────────────────────────────────────────────────────────
# Custom Transform: Multi-Class to Multi-Label (WT, TC, ET)
# ──────────────────────────────────────────────────────────────
class ConvertToBraTSMultiLabeld(MapTransform):
    """
    Convert single-channel multi-class segmentation mask [1, H, W, D] with
    labels {0, 1, 2, 3} into 3-channel binary multi-label tensor [3, H, W, D]:
      - Channel 0: Whole Tumor (WT)  = labels 1, 2, 3
      - Channel 1: Tumor Core (TC)   = labels 2, 3
      - Channel 2: Enhancing Tumor (ET) = label 3
    """

    def __init__(self, keys: List[str] = ["label"], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # Ensure tensor
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)

            # Squeeze channel dim if needed for indexing
            if val.ndim == 4 and val.shape[0] == 1:
                seg = val[0]
            else:
                seg = val

            wt = (seg == 1) | (seg == 2) | (seg == 3)
            tc = (seg == 2) | (seg == 3)
            et = (seg == 3)

            # Stack into [3, H, W, D]
            multilabel = torch.stack([wt, tc, et], dim=0).to(dtype=torch.float32)

            # Preserve metadata if available (MONAI MetaTensor)
            if hasattr(val, "meta"):
                from monai.data import MetaTensor
                d[key] = MetaTensor(multilabel, meta=val.meta)
            else:
                d[key] = multilabel
        return d


def convert_multilabel_to_multiclass(multilabel: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Reverse conversion: reconstruct discrete labels {0, 1, 2, 3} from
    binary multi-label channels [3, H, W, D] where:
      - WT = ch 0
      - TC = ch 1
      - ET = ch 2
    """
    if isinstance(multilabel, torch.Tensor):
        ml = multilabel.cpu().numpy()
    else:
        ml = np.asarray(multilabel)

    wt = ml[0] > 0.5
    tc = ml[1] > 0.5
    et = ml[2] > 0.5

    out = np.zeros_like(wt, dtype=np.int32)
    out[wt] = 1        # Edema
    out[tc] = 2        # Non-enhancing
    out[et] = 3        # Enhancing
    return out


# ──────────────────────────────────────────────────────────────
# Pipeline Factories
# ──────────────────────────────────────────────────────────────
def get_spatial_pipeline(allow_missing_keys: bool = True) -> Compose:
    """
    Invertible spatial pipeline (Orientation + Spacing) shared across train/val/test.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], allow_missing_keys=allow_missing_keys),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=allow_missing_keys),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel", allow_missing_keys=allow_missing_keys),
        Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=allow_missing_keys),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
            allow_missing_keys=allow_missing_keys,
        ),
    ])


def get_validation_pipeline(to_multilabel: bool = False) -> Compose:
    """
    Deterministic validation/evaluation preprocessing pipeline.
    Preserves spatial coordinates, performs non-zero z-score normalization.
    """
    transforms = [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
    if to_multilabel:
        transforms.append(ConvertToBraTSMultiLabeld(keys=["label"]))
    return Compose(transforms)


def get_training_pipeline(
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    to_multilabel: bool = True,
) -> Compose:
    """
    Training pipeline with data augmentation:
    - Deterministic spatial alignment (RAS, 1.0mm)
    - Foreground z-score normalization
    - Random spatial crops
    - Random flips along X, Y, Z axes
    - Multi-label conversion (WT, TC, ET)
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=crop_size,
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "label"]),
    ]
    if to_multilabel:
        transforms.append(ConvertToBraTSMultiLabeld(keys=["label"]))
    return Compose(transforms)


def get_inference_pipeline() -> Compose:
    """
    Preprocessing pipeline for unlabelled inference volumes (imagesTs).
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])


# ──────────────────────────────────────────────────────────────
# Visual QC & Verification Utility
# ──────────────────────────────────────────────────────────────
def run_visual_qc(
    dataset_root: Path,
    case_id: str = "BRATS_457",
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Executes an end-to-end QC test on a sample case:
    1. Loads raw NIfTI files.
    2. Runs full validation preprocessing pipeline.
    3. Verifies channel order, normalization statistics, label distribution.
    4. Inverts the spatial transform and checks for ZERO spatial drift.
    5. Generates a multi-panel visual QC PNG slice image.
    6. Produces a machine-readable QC JSON report.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)

    img_file = dataset_root / f"imagesTr/{case_id}.nii.gz"
    lbl_file = dataset_root / f"labelsTr/{case_id}.nii.gz"

    if not img_file.is_file() or not lbl_file.is_file():
        raise FileNotFoundError(f"Case files missing: {img_file} or {lbl_file}")

    print("=" * 65)
    print(f"  RUNNING PREPROCESSING QC TEST ON: {case_id}")
    print("=" * 65)

    data_item = {"image": str(img_file), "label": str(lbl_file)}

    # Step 1: Invertible spatial transform
    spatial_pipe = get_spatial_pipeline(allow_missing_keys=True)
    spatial_res = spatial_pipe(data_item)

    # Step 2: Full validation pipeline
    val_pipe = get_validation_pipeline(to_multilabel=False)
    processed = val_pipe(data_item)

    img_tensor: torch.Tensor = processed["image"]
    lbl_tensor: torch.Tensor = processed["label"]

    # Statistics check per channel
    stats = {}
    for c, name in MODALITIES.items():
        ch = img_tensor[c]
        nonzeros = ch[ch != 0]
        mean_val = float(nonzeros.mean().item()) if len(nonzeros) > 0 else 0.0
        std_val = float(nonzeros.std().item()) if len(nonzeros) > 0 else 0.0
        stats[f"channel_{c}_{name}"] = {
            "nonzero_voxels": int(len(nonzeros)),
            "nonzero_mean": round(mean_val, 4),
            "nonzero_std": round(std_val, 4),
            "min": round(float(ch.min().item()), 4),
            "max": round(float(ch.max().item()), 4),
        }
        print(f"  Channel {c} ({name:>5}): mean={mean_val:7.4f}, std={std_val:7.4f}, nonzero={len(nonzeros):>7}")

    # Step 3: Exact Inversion Test
    inverted = spatial_pipe.inverse({"label": spatial_res["label"]})
    orig_lbl = nib.load(str(lbl_file)).get_fdata()
    inv_lbl = inverted["label"][0].cpu().numpy()

    spatial_diff = float(np.sum(np.abs(inv_lbl - orig_lbl)))
    exact_match = spatial_diff == 0.0
    print(f"  Spatial Inversion Delta : {spatial_diff:.1f} (Exact match: {exact_match})")

    # Step 4: Multi-label mapping verification
    multilabel_converter = ConvertToBraTSMultiLabeld(keys=["label"])
    ml_res = multilabel_converter({"label": lbl_tensor.clone()})
    ml_tensor = ml_res["label"]
    wt_count = int((ml_tensor[0] > 0).sum().item())
    tc_count = int((ml_tensor[1] > 0).sum().item())
    et_count = int((ml_tensor[2] > 0).sum().item())

    # Consistency check: ET <= TC <= WT
    assert et_count <= tc_count <= wt_count, "Inconsistent tumor hierarchy (ET <= TC <= WT violated)"
    print(f"  Derived Volumes (voxels): WT={wt_count}, TC={tc_count}, ET={et_count}")

    # Step 5: Generate Visual QC Slice Image
    # Find the axial slice with highest tumor volume
    lbl_np = lbl_tensor[0].cpu().numpy()
    tumor_per_slice = (lbl_np > 0).sum(axis=(0, 1))
    best_slice_idx = int(np.argmax(tumor_per_slice))

    flair_slice = img_tensor[0, :, :, best_slice_idx].cpu().numpy()
    t1gd_slice = img_tensor[2, :, :, best_slice_idx].cpu().numpy()
    t2_slice = img_tensor[3, :, :, best_slice_idx].cpu().numpy()
    mask_slice = lbl_np[:, :, best_slice_idx]

    # Helper to convert float array to 8-bit grayscale
    def to_uint8(arr: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(arr[arr > 0], (1, 99)) if np.any(arr > 0) else (0, 1)
        clipped = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0, 1)
        return (clipped * 255).astype(np.uint8)

    flair_u8 = to_uint8(flair_slice)
    t1gd_u8 = to_uint8(t1gd_slice)
    t2_u8 = to_uint8(t2_slice)

    # Color overlay on FLAIR for Ground Truth:
    # 1: Edema -> Yellow/Green [0, 255, 0]
    # 2: Non-enhancing -> Blue [0, 120, 255]
    # 3: Enhancing -> Red [255, 40, 40]
    rgb_overlay = np.stack([flair_u8, flair_u8, flair_u8], axis=-1)
    rgb_overlay[mask_slice == 1] = [50, 220, 50]    # Green
    rgb_overlay[mask_slice == 2] = [40, 120, 255]   # Blue
    rgb_overlay[mask_slice == 3] = [255, 50, 50]    # Red

    # Multi-label overlay on T1gd:
    # WT contour/fill
    ml_slice_wt = ml_tensor[0, :, :, best_slice_idx].cpu().numpy() > 0.5
    ml_slice_tc = ml_tensor[1, :, :, best_slice_idx].cpu().numpy() > 0.5
    ml_slice_et = ml_tensor[2, :, :, best_slice_idx].cpu().numpy() > 0.5

    rgb_ml = np.stack([t1gd_u8, t1gd_u8, t1gd_u8], axis=-1)
    rgb_ml[ml_slice_wt] = [60, 200, 60]
    rgb_ml[ml_slice_tc] = [50, 130, 240]
    rgb_ml[ml_slice_et] = [255, 40, 40]

    # Stitch panels horizontally:
    # [FLAIR] | [T1gd] | [T2] | [GT Mask Overlay] | [Multi-label WT/TC/ET]
    def rot(arr):
        return np.rot90(arr)

    p1 = rot(np.stack([flair_u8] * 3, axis=-1))
    p2 = rot(np.stack([t1gd_u8] * 3, axis=-1))
    p3 = rot(np.stack([t2_u8] * 3, axis=-1))
    p4 = rot(rgb_overlay)
    p5 = rot(rgb_ml)

    separator = np.zeros((p1.shape[0], 4, 3), dtype=np.uint8) + 120
    stitched = np.concatenate([p1, separator, p2, separator, p3, separator, p4, separator, p5], axis=1)

    qc_image_path = output_dir / f"qc_slice_{case_id}.png"
    Image.fromarray(stitched).save(str(qc_image_path))
    print(f"  Visual QC Image Saved   : {qc_image_path.name}")

    report = {
        "case_id": case_id,
        "input_shape": list(orig_lbl.shape),
        "processed_image_shape": list(img_tensor.shape),
        "processed_label_shape": list(lbl_tensor.shape),
        "best_slice_index": best_slice_idx,
        "channel_statistics": stats,
        "spatial_inversion_exact": exact_match,
        "spatial_inversion_diff_sum": spatial_diff,
        "derived_tumor_voxels": {
            "Whole_Tumor_WT": wt_count,
            "Tumor_Core_TC": tc_count,
            "Enhancing_Tumor_ET": et_count,
        },
        "qc_image": str(qc_image_path.name),
        "status": "PASS" if exact_match else "FAIL",
    }

    report_path = output_dir / "preprocessing_qc_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  QC Report Saved         : {report_path.name}")
    print("=" * 65)

    return report


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MONAI Preprocessing Pipeline for MSD Task01_BrainTumour")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "Task01_BrainTumour" / "Task01_BrainTumour",
        help="Path to Task01_BrainTumour dataset root containing imagesTr and labelsTr",
    )
    parser.add_argument(
        "--qc-case",
        type=str,
        default="BRATS_457",
        help="Case ID to use for visual QC test (default: BRATS_457)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for reports and QC images",
    )
    args = parser.parse_args()

    report = run_visual_qc(
        dataset_root=args.data.resolve(),
        case_id=args.qc_case,
        output_dir=args.out.resolve(),
    )

    if report["status"] == "PASS":
        print("[SUCCESS] MONAI Preprocessing Pipeline validated and ready!")
    else:
        print("[FAIL] Visual QC failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
