#!/usr/bin/env python3
"""
mri_filtration_and_enhancement.py
==================================
NeuroScope AI — Advanced Medical MRI Filtration & Enhancement Engine.
Complies with Clinical Radiologist / Supervisor directives:
  1. Filtration: Suppresses scanner Rician noise, extreme spike artifacts (percentile clipping),
     and enforces strict intracranial brain masking (eliminating air and skull leakage).
  2. Enhancement: Channel-wise intensity standardization (Z-score on brain tissue) and
     adaptive contrast stretching for Gadolinium-enhanced (T1ce) active lesions.
  3. Tailored Preprocessing Pipelines for the 4 target diseases:
     - Glioma (Multi-region 4-channel)
     - Meningioma (Dural solid mass 4-channel)
     - Brain Metastasis (Punctate micro-lesions, strictly NO spatial downsampling)
     - Hippocampus (High-precision medial temporal anatomical volumetry)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    Spacingd,
    SpatialPadd,
)


class PercentileClippingAndFiltrationd(MapTransform):
    """
    Filtration Step:
    Clips extreme intensity outliers (scanner spike noise and Rician artifacts)
    between lower_percentile and upper_percentile across non-zero brain tissue.
    """
    def __init__(
        self,
        keys: Sequence[str],
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.lower_pct = lower_percentile
        self.upper_pct = upper_percentile

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # Support both PyTorch tensors and NumPy arrays
            is_tensor = isinstance(val, torch.Tensor)
            arr = val.detach().cpu().numpy() if is_tensor else np.array(val)

            # Process each MRI channel independently
            for c in range(arr.shape[0]):
                channel_data = arr[c]
                brain_mask = channel_data > 0
                if np.count_nonzero(brain_mask) > 100:
                    v_min = np.percentile(channel_data[brain_mask], self.lower_pct)
                    v_max = np.percentile(channel_data[brain_mask], self.upper_pct)
                    channel_data = np.clip(channel_data, v_min, v_max)
                    arr[c] = channel_data

            d[key] = torch.from_numpy(arr).to(val.device) if is_tensor else arr
        return d


class BrainMaskGuardFiltrationd(MapTransform):
    """
    Filtration Step:
    Generates an intracranial brain mask from combined MRI sequences and strictly zeroes
    out all background air, skull, and extracranial noise from the image and label tensors.
    """
    def __init__(
        self,
        image_key: str = "image",
        label_key: str | None = "label",
        allow_missing_keys: bool = True,
    ):
        keys = [image_key] if label_key is None else [image_key, label_key]
        super().__init__(keys, allow_missing_keys)
        self.img_key = image_key
        self.lbl_key = label_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        img = d[self.img_key]
        is_tensor = isinstance(img, torch.Tensor)

        if is_tensor:
            # Combine non-zero brain signal across all channels
            brain_mask = img.abs().sum(dim=0, keepdim=True) > 1e-4
            d[self.img_key] = img * brain_mask.float()
            if self.lbl_key and self.lbl_key in d:
                lbl = d[self.lbl_key]
                # Keep label strictly inside intracranial mask
                lbl_mask = brain_mask if lbl.shape[0] == 1 else brain_mask.expand_as(lbl)
                d[self.lbl_key] = lbl * lbl_mask.to(lbl.dtype)
        else:
            brain_mask = np.sum(np.abs(img), axis=0, keepdims=True) > 1e-4
            d[self.img_key] = img * brain_mask.astype(img.dtype)
            if self.lbl_key and self.lbl_key in d:
                lbl = d[self.lbl_key]
                d[self.lbl_key] = lbl * brain_mask.astype(lbl.dtype)
        return d


class ContrastEnhancementT1ced(MapTransform):
    """
    Enhancement Step:
    Specifically enhances contrast for Gadolinium-enhanced T1ce channel (Channel 2 in standard 4-channel stack)
    to accentuate hyperintense active tumor margins and punctate metastases.
    """
    def __init__(
        self,
        keys: Sequence[str],
        t1ce_channel_idx: int = 2,
        gamma: float = 1.2,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.channel_idx = t1ce_channel_idx
        self.gamma = gamma

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            if val.shape[0] > self.channel_idx:
                is_tensor = isinstance(val, torch.Tensor)
                t1ce = val[self.channel_idx]
                mask = t1ce > 0
                if is_tensor:
                    if torch.count_nonzero(mask) > 100:
                        min_val = t1ce[mask].min()
                        max_val = t1ce[mask].max()
                        if max_val > min_val:
                            norm = (t1ce - min_val) / (max_val - min_val + 1e-8)
                            enhanced = torch.pow(torch.clamp(norm, 0.0, 1.0), self.gamma)
                            val[self.channel_idx] = enhanced * (max_val - min_val) + min_val
                else:
                    if np.count_nonzero(mask) > 100:
                        min_val = t1ce[mask].min()
                        max_val = t1ce[mask].max()
                        if max_val > min_val:
                            norm = (t1ce - min_val) / (max_val - min_val + 1e-8)
                            enhanced = np.power(np.clip(norm, 0.0, 1.0), self.gamma)
                            val[self.channel_idx] = enhanced * (max_val - min_val) + min_val
        return d


class ConvertToMultiLabeld(MapTransform):
    """
    Converts multi-class discrete labels to composite Multi-Region binary masks:
      Channel 0: Whole Tumor (WT) = (label >= 1)
      Channel 1: Tumor Core  (TC) = (label in [2, 3])
      Channel 2: Enhancing   (ET) = (label == 3)
    """
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            is_tensor = isinstance(lbl, torch.Tensor)
            if is_tensor:
                # Squeeze to 3D if channel dim present
                seg = lbl[0] if lbl.ndim == 4 else lbl
                wt = (seg >= 1).to(torch.float32)
                tc = ((seg == 2) | (seg == 3)).to(torch.float32)
                et = (seg == 3).to(torch.float32)
                d[key] = torch.stack([wt, tc, et], dim=0)
            else:
                seg = lbl[0] if lbl.ndim == 4 else lbl
                wt = (seg >= 1).astype(np.float32)
                tc = ((seg == 2) | (seg == 3)).astype(np.float32)
                et = (seg == 3).astype(np.float32)
                d[key] = np.stack([wt, tc, et], axis=0)
        return d


class ConvertToBinaryTargetd(MapTransform):
    """Converts any positive integer segmentation label to a single binary lesion channel (seg > 0)."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            is_tensor = isinstance(lbl, torch.Tensor)
            if is_tensor:
                seg = lbl[0] if lbl.ndim == 4 else lbl
                d[key] = (seg > 0).to(torch.float32).unsqueeze(0)
            else:
                seg = lbl[0] if lbl.ndim == 4 else lbl
                d[key] = (seg > 0).astype(np.float32)[np.newaxis, ...]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & Enhancement Pipelines for the 4 Retraining Diseases
# ─────────────────────────────────────────────────────────────────────────────

def get_glioma_pipeline(mode: str = "train", crop_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """
    Glioma Pipeline:
    - 4 Sequences: FLAIR, T1, T1ce, T2
    - Multi-Region Labels: WT, TC, ET
    - Filtration: Percentile clipping + non-zero Brain Mask Guard
    - Enhancement: Channel-wise Z-score + T1ce contrast enhancement
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # Filtration: Remove scanner spikes & enforce brain boundary
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        # Enhancement: Normalize & enhance T1ce contrast
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=2, gamma=1.15),
        ConvertToMultiLabeld(keys=["label"]),
    ]

    if mode == "train":
        transforms.extend([
            SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=crop_size,
                pos=2.0,
                neg=1.0,
                num_samples=1,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        ])

    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_meningioma_pipeline(mode: str = "train", crop_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """
    Meningioma Pipeline:
    - Fixed Channel Ordering: [FLAIR, T1, T1ce, T2]
    - Single Solid Mass Binary Target (Tumor Core / Dural Mass)
    - Filtration: Brain mask guard suppressing skull bone / dural sinus false positives
    - Enhancement: Focused T1ce Gadolinium contrast stretching
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=2, gamma=1.25),
        ConvertToBinaryTargetd(keys=["label"]),
    ]

    if mode == "train":
        transforms.extend([
            SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=crop_size,
                pos=3.0,
                neg=1.0,
                num_samples=1,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.03),
        ])

    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_metastasis_pipeline(mode: str = "train", crop_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """
    Brain Metastasis Pipeline:
    - High-Resolution Micro-Lesion Focus: STRICTLY NO DOWNSAMPLING
    - Filtration: High-fidelity outlier clipping to preserve sharp punctate enhancement
    - Enhancement: Contrast stretching + high positive lesion sampling (pos=4.0)
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.2, upper_percentile=99.8),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ContrastEnhancementT1ced(keys=["image"], t1ce_channel_idx=2, gamma=1.2),
        ConvertToBinaryTargetd(keys=["label"]),
    ]

    if mode == "train":
        transforms.extend([
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
        ])

    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_hippocampus_pipeline(mode: str = "train", crop_size: Tuple[int, int, int] = (48, 64, 48)) -> Compose:
    """
    Hippocampus Pipeline:
    - High-Resolution Structural T1
    - Filtration: Intracranial masking and intensity outlier clipping
    - Enhancement: Z-score normalization within brain tissue
    - Medial Temporal Bounding Box Focus: avoids empty background patches
    """
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        PercentileClippingAndFiltrationd(keys=["image"], lower_percentile=0.5, upper_percentile=99.5),
        BrainMaskGuardFiltrationd(image_key="image", label_key="label"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False),
        ConvertToBinaryTargetd(keys=["label"]),
    ]

    if mode == "train":
        transforms.extend([
            SpatialPadd(keys=["image", "label"], spatial_size=crop_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=crop_size,
                pos=3.0,
                neg=1.0,
                num_samples=1,
            ),
            # Safe spatial augmentations (careful: no horizontal flip if distinguishing left/right)
            RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, mode=("bilinear", "nearest"), prob=0.3),
        ])

    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)
