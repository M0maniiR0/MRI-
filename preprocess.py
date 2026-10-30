#!/usr/bin/env python3
"""
preprocess.py
=============
Unified Preprocessing & Normalization Pipeline for Track 2 (ACIT26 BrainAgenticthon).
Shared contract between Task C (Roa'a) and Tasks A/B (Abdulrahman).
Used identically during training and offline inference.

Specifications:
1. DICOM -> NIfTI conversion support via offline libraries.
2. Reorientation to standard RAS.
3. Isotropic resampling to 1.0 mm (bilinear/spline for images).
4. Uniform skull-stripping via documented intensity-based 3D brain extraction.
5. Intensity percentile clipping (0.5% / 99.5%) and brain-voxel Z-score normalization.
6. Fixed volumetric shape crop/pad (default [128, 128, 128]) with full inverse transform metadata.
7. Fixed channel convention: Task C uses single T1-weighted channel.
8. Automated Quality Control (QC) detecting NaN/inf, empty masks, bad dimensions, extreme values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch
import json
try:
    import yaml
except ImportError:
    yaml = None
from PIL import Image

import monai.transforms as mt


class QCValidationError(ValueError):
    """Raised when an MRI scan fails automated quality control checks."""
    pass


def load_config(config_path: Union[str, Path] = "preprocessing_config.yaml") -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / config_path

    if p.suffix in [".yaml", ".yml"] and yaml is not None and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # JSON fallback
    json_path = p.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if p.exists() and yaml is not None:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Standard built-in defaults
    return {
        "spatial": {
            "target_orientation": "RAS",
            "target_spacing": [1.0, 1.0, 1.0],
            "interpolation_image": "bilinear",
            "interpolation_mask": "nearest",
            "fixed_shape": [128, 128, 128]
        },
        "intensity": {
            "clip_lower_percentile": 0.5,
            "clip_upper_percentile": 99.5,
            "eps": 1e-8
        },
        "quality_control": {
            "allow_nan_or_inf": False,
            "min_brain_voxels": 500,
            "extreme_value_bounds": [-15.0, 15.0]
        }
    }


def convert_dicom_to_nifti_if_needed(file_or_dir: Union[str, Path], temp_output_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Checks if input is a DICOM file or directory. If DICOM, converts to NIfTI offline.
    If already NIfTI (.nii or .nii.gz), returns path directly.
    """
    path = Path(file_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # Already NIfTI
    if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return path

    # Directory or DICOM file
    if path.is_dir():
        dcm_files = sorted(list(path.glob("*.dcm")) + list(path.glob("*.[0-9]*")))
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM or NIfTI files found in directory: {path}")
        # Use nibabel / pydicom offline logic
        try:
            import pydicom
            slices = [pydicom.dcmread(str(f)) for f in dcm_files]
            slices.sort(key=lambda s: float(getattr(s, "ImagePositionPatient", [0, 0, float(getattr(s, "InstanceNumber", 0))])[2]))
            image_3d = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)
            nii_img = nib.Nifti1Image(image_3d, np.eye(4))
            out_dir = Path(temp_output_dir or path)
            out_file = out_dir / f"{path.name}_converted.nii.gz"
            nib.save(nii_img, str(out_file))
            return out_file
        except Exception as e:
            raise RuntimeError(f"Offline DICOM conversion failed: {e}")
    else:
        # Single DICOM file
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            nii_img = nib.Nifti1Image(arr, np.eye(4))
            out_file = path.with_suffix(".nii.gz")
            nib.save(nii_img, str(out_file))
            return out_file
        except Exception as e:
            raise RuntimeError(f"Offline DICOM file conversion failed: {e}")


def intensity_based_brain_mask(image_data: np.ndarray) -> np.ndarray:
    """
    Documented offline 3D intensity-based morphological skull-stripping.
    1. Computes adaptive threshold on positive voxel intensities.
    2. Identifies largest 3D connected component (the brain tissue).
    3. Fills internal ventricles/holes and applies smooth morphological closing.
    """
    pos_voxels = image_data[image_data > 0]
    if len(pos_voxels) == 0:
        return np.zeros_like(image_data, dtype=bool)

    # Threshold at the 15th percentile of positive voxels, largest 3D connected component, binary fill holes, one dilation
    thresh = np.percentile(pos_voxels, 15.0)
    binary = image_data > thresh

    # 3D Largest connected component
    labeled, num_features = ndi.label(binary)
    if num_features == 0:
        return binary

    sizes = ndi.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    brain_mask = labeled == largest_label

    # Morphological hole filling & slight dilation to protect cortical edges
    brain_mask = ndi.binary_fill_holes(brain_mask)
    structure = ndi.generate_binary_structure(3, 1)
    brain_mask = ndi.binary_dilation(brain_mask, structure=structure, iterations=1)
    return brain_mask


def run_automated_qc(image_data: np.ndarray, brain_mask: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automated QC audit per scan:
    - Check NaN/Inf
    - Brain volume bounds (>1000 voxels)
    - Dimensionality
    - Extreme voxel bounds
    """
    qc_conf = config.get("quality_control", {})
    issues = []

    # 1. NaN / Inf
    if not np.isfinite(image_data).all():
        issues.append("Scan contains NaN or Inf values")

    # 2. Empty or tiny brain mask
    brain_voxels = int(np.sum(brain_mask))
    min_vox = qc_conf.get("min_brain_voxels", 1000)
    if brain_voxels < min_vox:
        issues.append(f"Brain mask volume ({brain_voxels} voxels) is below safety threshold ({min_vox})")

    # 3. Dimensionality
    if image_data.ndim != 3:
        issues.append(f"Invalid dimensionality: expected 3D, got {image_data.ndim}D")

    # 4. Range check
    min_val, max_val = float(np.min(image_data)), float(np.max(image_data))
    bounds = qc_conf.get("extreme_value_bounds", [-15.0, 15.0])
    if min_val < bounds[0] or max_val > bounds[1]:
        issues.append(f"Voxel values [{min_val:.2f}, {max_val:.2f}] exceed allowable bounds {bounds}")

    is_passed = len(issues) == 0
    qc_result = {
        "passed": is_passed,
        "brain_voxels": brain_voxels,
        "min_val": min_val,
        "max_val": max_val,
        "mean_brain_val": float(np.mean(image_data[brain_mask])) if brain_voxels > 0 else 0.0,
        "std_brain_val": float(np.std(image_data[brain_mask])) if brain_voxels > 0 else 0.0,
        "issues": issues
    }

    if not is_passed and not qc_conf.get("allow_nan_or_inf", False) and any("NaN" in s for s in issues):
        raise QCValidationError(f"Scan failed critical QC: {'; '.join(issues)}")

    return qc_result


def crop_or_pad_3d(image: np.ndarray, target_shape: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Crops or pads 3D array to target_shape centered on the volume.
    Returns:
      processed_image, inverse_metadata (containing pad_widths and crop_slices)
    """
    orig_shape = image.shape
    curr_d, curr_h, curr_w = orig_shape
    t_d, t_h, t_w = target_shape

    # Compute crop or pad for each dimension
    pads = []
    crops = []

    for curr_len, target_len in zip([curr_d, curr_h, curr_w], [t_d, t_h, t_w]):
        if curr_len < target_len:
            pad_before = (target_len - curr_len) // 2
            pad_after = target_len - curr_len - pad_before
            pads.append((pad_before, pad_after))
            crops.append(slice(0, target_len))
        else:
            crop_start = (curr_len - target_len) // 2
            crop_end = crop_start + target_len
            pads.append((0, 0))
            crops.append(slice(crop_start, crop_end))

    # Apply padding if any
    padded = np.pad(image, pads, mode="constant", constant_values=0.0)
    # Apply cropping
    cropped = padded[crops[0], crops[1], crops[2]]

    inv_metadata = {
        "original_shape": orig_shape,
        "target_shape": target_shape,
        "pads": pads,
        "crops": [(c.start, c.stop) for c in crops]
    }
    return cropped, inv_metadata


def preprocess_mri(
    input_path: Union[str, Path],
    config_path: Union[str, Path] = "preprocessing_config.yaml",
    channel_index: int = 0
) -> Dict[str, Any]:
    """
    Full unified preprocessing pipeline:
    1. Loads NIfTI (or converts DICOM).
    2. Reorients to RAS.
    3. Resamples to 1.0 mm isotropic.
    4. Skull-strips via intensity-based mask.
    5. Clips percentiles (0.5% - 99.5%) and Z-scores inside the brain.
    6. Crops/pads to fixed target shape (128, 128, 128).
    7. Runs automated QC.
    """
    cfg = load_config(config_path)

    # 1. Resolve path
    nii_path = convert_dicom_to_nifti_if_needed(input_path)

    # 2. Check if already RAS and 1.0mm isotropic to bypass heavy resampling
    nii = nib.load(str(nii_path))
    zooms = nii.header.get_zooms()[:3]
    is_isotropic = np.allclose(zooms, cfg["spatial"]["target_spacing"], atol=0.05)

    data_dict = {"image": str(nii_path)}
    transforms_list = [
        mt.LoadImaged(keys=["image"], ensure_channel_first=True),
        mt.Orientationd(keys=["image"], axcodes=cfg["spatial"]["target_orientation"])
    ]
    if not is_isotropic:
        transforms_list.append(
            mt.Spacingd(
                keys=["image"],
                pixdim=cfg["spatial"]["target_spacing"],
                mode=cfg["spatial"]["interpolation_image"]
            )
        )
    spatial_transforms = mt.Compose(transforms_list)

    transformed = spatial_transforms(data_dict)
    img_tensor = transformed["image"]

    # Support 4D inputs (e.g. BraTS multi-channel) by selecting channel
    if img_tensor.shape[0] > 1:
        # Channel selection: channel_index (e.g. 1 for T1 in BraTS 4-channel, or 0 for single-channel)
        selected_channel = img_tensor[channel_index].numpy()
    else:
        selected_channel = img_tensor[0].numpy()

    raw_before = selected_channel.copy()

    # 4. Skull-stripping
    brain_mask = intensity_based_brain_mask(selected_channel)
    stripped = np.where(brain_mask, selected_channel, 0.0)

    # 5. Intensity Clipping & Non-zero Z-score
    brain_voxels = stripped[brain_mask]
    if len(brain_voxels) > 0:
        p_low = np.percentile(brain_voxels, cfg["intensity"]["clip_lower_percentile"])
        p_high = np.percentile(brain_voxels, cfg["intensity"]["clip_upper_percentile"])
        clipped = np.clip(stripped, p_low, p_high)
        # Recompute mean and std on clipped brain voxels
        brain_clipped = clipped[brain_mask]
        mean_v = np.mean(brain_clipped)
        std_v = np.std(brain_clipped) + cfg["intensity"]["eps"]
        normalized = np.where(brain_mask, (clipped - mean_v) / std_v, 0.0)
    else:
        normalized = stripped

    # 6. Fixed shape crop/pad
    target_shape = tuple(cfg["spatial"]["fixed_shape"])
    final_volume, inv_meta = crop_or_pad_3d(normalized, target_shape=target_shape)

    # Brain mask cropped/padded identically for QC verification
    final_mask, _ = crop_or_pad_3d(brain_mask.astype(np.float32), target_shape=target_shape)
    final_mask = final_mask > 0.5

    # 7. Automated QC
    qc_report = run_automated_qc(final_volume, final_mask, cfg)

    return {
        "processed_volume": final_volume,
        "brain_mask": final_mask,
        "raw_volume_before": raw_before,
        "inverse_transform_info": inv_meta,
        "qc_report": qc_report,
        "original_path": str(input_path),
        "target_shape": target_shape,
        "spacing": cfg["spatial"]["target_spacing"],
        "orientation": cfg["spatial"]["target_orientation"]
    }


def save_orthogonal_slices_png(volume: np.ndarray, out_path: Union[str, Path], title: str = "MRI Slice"):
    """
    Saves before/after orthogonal middle slice visualization (Axial, Coronal, Sagittal).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    d, h, w = volume.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

    # Extract 3 middle slices
    ax_slice = volume[mid_d, :, :]
    cor_slice = volume[:, mid_h, :]
    sag_slice = volume[:, :, mid_w]

    def to_uint8(s):
        s_min, s_max = s.min(), s.max()
        if s_max - s_min > 1e-6:
            norm = (s - s_min) / (s_max - s_min) * 255.0
        else:
            norm = np.zeros_like(s)
        return norm.astype(np.uint8)

    img_ax = Image.fromarray(to_uint8(ax_slice)).resize((256, 256))
    img_cor = Image.fromarray(to_uint8(cor_slice)).resize((256, 256))
    img_sag = Image.fromarray(to_uint8(sag_slice)).resize((256, 256))

    # Composite into horizontal montage
    montage = Image.new("L", (256 * 3, 256))
    montage.paste(img_ax, (0, 0))
    montage.paste(img_cor, (256, 0))
    montage.paste(img_sag, (512, 0))
    montage.save(str(out_path))


if __name__ == "__main__":
    print("[INFO] preprocess.py module loaded successfully.")
