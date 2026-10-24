#!/usr/bin/env python3
"""
quantify_multiple_sclerosis.py
==============================
NeuroScope AI — Multiple Sclerosis (التصلب المتعدد / التصلب اللويحي) Volumetric & Lesion Load Engine.
Calculates 3D Demyelinating lesion burden (mm^3) and lesion count via connected-component analysis.
Generates publication-quality 3-view visual overlays (Coronal, Sagittal, Axial) on FLAIR MRI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from monai.inferers import sliding_window_inference

from train_multiple_sclerosis import create_model, get_ms_transforms, compute_dice_score

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_MultipleSclerosis"
DEFAULT_MODEL = SCRIPT_DIR / "models" / "segresnet_multiple_sclerosis_best.pt"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "multiple_sclerosis_analysis"
REPORTS_DIR = SCRIPT_DIR / "reports"


def plot_ms_overlays(
    flair_data: np.ndarray,
    pred_mask: np.ndarray,
    pid: str,
    total_vol: float,
    num_lesions: int,
    dice: float | None,
    save_path: Path,
):
    """Generates a high-resolution 3-view panel (Coronal, Sagittal, Axial) with MS overlays."""
    # Find slice with maximum lesion voxels
    lesion_sum_z = (pred_mask > 0).sum(axis=(0, 1))
    axial_idx = int(np.argmax(lesion_sum_z)) if lesion_sum_z.max() > 0 else flair_data.shape[2] // 2

    lesion_sum_y = (pred_mask > 0).sum(axis=(0, 2))
    coronal_idx = int(np.argmax(lesion_sum_y)) if lesion_sum_y.max() > 0 else flair_data.shape[1] // 2

    lesion_sum_x = (pred_mask > 0).sum(axis=(1, 2))
    sagittal_idx = int(np.argmax(lesion_sum_x)) if lesion_sum_x.max() > 0 else flair_data.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#060c18")

    views = [
        ("Coronal (Y=" + str(coronal_idx) + ")", np.rot90(flair_data[:, coronal_idx, :]), np.rot90(pred_mask[:, coronal_idx, :])),
        ("Sagittal (X=" + str(sagittal_idx) + ")", np.rot90(flair_data[sagittal_idx, :, :]), np.rot90(pred_mask[sagittal_idx, :, :])),
        ("Axial (Z=" + str(axial_idx) + ")", np.rot90(flair_data[:, :, axial_idx]), np.rot90(pred_mask[:, :, axial_idx])),
    ]

    for ax, (title, mri_sl, pred_sl) in zip(axes, views):
        norm_sl = np.clip((mri_sl - mri_sl.min()) / (mri_sl.max() - mri_sl.min() + 1e-5), 0, 1)
        ax.imshow(norm_sl, cmap="gray", origin="lower")

        # Color overlay: Electric Cyan / Amber for MS Demyelinating Plaque
        overlay = np.zeros((*norm_sl.shape, 4), dtype=np.float32)
        overlay[pred_sl > 0] = [0.0, 0.90, 0.95, 0.75]
        ax.imshow(overlay, origin="lower")

        ax.set_title(title, color="#f1f5f9", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    dice_str = f" | Dice: {dice:.3f}" if dice is not None else ""
    fig.suptitle(
        f"NeuroScope AI — Multiple Sclerosis Analysis: Patient {pid}\n"
        f"Total Lesion Load (TLL): {total_vol:,.1f} mm³ | Plaque Count: {num_lesions} lesion(s){dice_str}",
        color="#38bdf8",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate_patient_case(
    case_dict: Dict,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: Tuple[int, int, int] = (96, 96, 96),
) -> Dict:
    pid = case_dict["patient_id"]
    mri_nii = nib.load(case_dict["mri_path"])
    mri_data = mri_nii.get_fdata(dtype=np.float32)
    zooms = mri_nii.header.get_zooms()[:3]
    voxel_vol = float(zooms[0] * zooms[1] * zooms[2])

    _, val_pipe = get_ms_transforms()
    item = val_pipe({
        "image": torch.from_numpy(mri_data).unsqueeze(0),
        "label": torch.zeros((1, *mri_data.shape)),
    })

    x = item["image"].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = sliding_window_inference(
                inputs=x,
                roi_size=crop_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )
        pred_prob = torch.sigmoid(out).squeeze(0).squeeze(0)
        brain_mask = (item["image"].abs().sum(dim=0) > 0).to(device)
        pred_bin = ((pred_prob > 0.5) & brain_mask).cpu().numpy().astype(np.uint8)

    # If spatial padding occurred, unpad back to original MRI shape
    slices = []
    for d, orig_len in enumerate(mri_data.shape):
        if pred_bin.shape[d] > orig_len:
            pad_before = (pred_bin.shape[d] - orig_len) // 2
            slices.append(slice(pad_before, pad_before + orig_len))
        else:
            slices.append(slice(0, orig_len))
    pred_bin = pred_bin[tuple(slices)]

    # Connected components for MS plaque lesion counting
    labeled_lesions, num_lesions = ndi.label(pred_bin)
    total_voxels = int(pred_bin.sum())
    total_vol = total_voxels * voxel_vol

    dice = None
    gt_vol = None
    if "lesion_mask_path" in case_dict and Path(case_dict["lesion_mask_path"]).is_file():
        mask_nii = nib.load(case_dict["lesion_mask_path"])
        gt_bin = (mask_nii.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
        gt_vol = float((gt_bin > 0).sum() * voxel_vol)
        dice = compute_dice_score(torch.from_numpy(pred_bin).float(), torch.from_numpy(gt_bin).float())

    vis_path = OUTPUT_DIR / f"{pid}_ms_visual.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ms_overlays(
        flair_data=mri_data,
        pred_mask=pred_bin,
        pid=pid,
        total_vol=total_vol,
        num_lesions=num_lesions,
        dice=dice,
        save_path=vis_path,
    )

    print(f"[{pid}] MS Lesion Load: {total_vol:,.1f} mm³ | Plaques: {num_lesions}" + (f" | Dice: {dice:.4f}" if dice is not None else ""))

    res = {
        "patient_id": pid,
        "voxel_spacing_mm": [float(z) for z in zooms],
        "total_lesion_load_mm3": float(total_vol),
        "ground_truth_volume_mm3": float(gt_vol) if gt_vol is not None else None,
        "plaque_count": int(num_lesions),
        "dice_score": float(dice) if dice is not None else None,
        "visual_path": str(vis_path),
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multiple Sclerosis Cohort")
    parser.add_argument("--num-patients", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = create_model(device)
    if DEFAULT_MODEL.is_file():
        ckpt = torch.load(str(DEFAULT_MODEL), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Loaded weights from {DEFAULT_MODEL.name} (Val Dice: {ckpt.get('val_dice', 0):.4f})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {DEFAULT_MODEL}")

    manifest_path = DEFAULT_DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Sample evenly across cohort
    step = max(1, len(manifest) // args.num_patients)
    eval_cohort = [manifest[i * step] for i in range(min(args.num_patients, len(manifest)))]

    print(f"\n======================================================================")
    print(f"  RUNNING MULTIPLE SCLEROSIS COHORT QUANTIFICATION ({len(eval_cohort)} scans)")
    print(f"======================================================================")

    results = []
    for item in eval_cohort:
        res = evaluate_patient_case(item, model, device)
        results.append(res)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json = REPORTS_DIR / "multiple_sclerosis_cohort_report.json"
    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# Multiple Sclerosis (التصلب المتعدد / التصلب اللويحي) — Cohort Volumetric Report",
        "",
        "**Date**: 2026-10-24 | **Model**: SegResNet 3D (Trained on RTX 2050 GPU) | **Dataset**: MSLesSeg-2024 & WMH Challenge (240 Cases Cohort)",
        "",
        "| Scan / Patient ID | Total Lesion Load (mm³) | Ground Truth (mm³) | Plaque Count | Dice Similarity | Clinical Stage / Burden |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in results:
        gt_s = f"{r['ground_truth_volume_mm3']:,.1f}" if r["ground_truth_volume_mm3"] is not None else "N/A"
        dice_s = f"{r['dice_score']:.3f}" if r["dice_score"] is not None else "N/A"
        vol = r["total_lesion_load_mm3"]
        if vol < 5000:
            stage = "🟢 **Mild Burden (<5k mm³)**"
        elif vol <= 15000:
            stage = "🟡 **Moderate Burden (5k-15k mm³)**"
        else:
            stage = "🔴 **Severe Burden (>15k mm³)**"

        md_lines.append(f"| `{r['patient_id']}` | {vol:,.1f} | {gt_s} | {r['plaque_count']} | {dice_s} | {stage} |")

    md_lines.extend([
        "",
        "## Diagnostic Overlays",
        f"High-resolution 3-plane views (Coronal, Sagittal, Axial) saved to `{OUTPUT_DIR.relative_to(SCRIPT_DIR)}/`.",
    ])

    report_md = REPORTS_DIR / "multiple_sclerosis_cohort_report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[DONE] Saved Multiple Sclerosis Cohort Report to:")
    print(f"  - Markdown: {report_md.relative_to(SCRIPT_DIR)}")
    print(f"  - JSON    : {report_json.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
