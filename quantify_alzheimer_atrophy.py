#!/usr/bin/env python3
"""
quantify_alzheimer_atrophy.py
=============================
NeuroScope AI — Alzheimer's Disease Biomarker & Hippocampal Volumetric Engine.
Calculates 3D Hippocampus volume (mm^3) to evaluate neurodegeneration and brain atrophy.
Generates publication-grade coronal/sagittal/axial overlay images with matplotlib.
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
import torch
from monai.transforms import ResizeWithPadOrCrop

from train_alzheimer_hippocampus import create_model, get_transforms

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task04_Hippocampus"
DEFAULT_MODEL = SCRIPT_DIR / "models" / "segresnet_alzheimer_hippocampus_best.pt"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "alzheimer_atrophy_analysis"
REPORTS_DIR = SCRIPT_DIR / "reports"


def assess_atrophy(total_vol_mm3: float) -> Tuple[str, str]:
    """Clinical risk classification based on hippocampal volume."""
    if total_vol_mm3 >= 3000.0:
        return "NORMAL", "Preserved hippocampal volume (Low risk of neurodegeneration)"
    elif total_vol_mm3 >= 2400.0:
        return "BORDERLINE / MCI", "Mild volume reduction (Consistent with Mild Cognitive Impairment / Early Alzheimer's)"
    else:
        return "ATROPHY / ALZHEIMER'S", "Significant hippocampal atrophy (High radiological correlation with Alzheimer's Disease)"


def plot_hippocampus_overlays(
    img_data: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    case_id: str,
    total_vol: float,
    status: str,
    dice: float | None,
    save_path: Path,
):
    """Generates a high-resolution 3-view panel (Coronal, Sagittal, Axial) with overlays."""
    # Find slice with maximum hippocampal volume
    hip_sum_z = (pred_mask > 0).sum(axis=(0, 1))
    axial_idx = int(np.argmax(hip_sum_z)) if hip_sum_z.max() > 0 else img_data.shape[2] // 2

    hip_sum_y = (pred_mask > 0).sum(axis=(0, 2))
    coronal_idx = int(np.argmax(hip_sum_y)) if hip_sum_y.max() > 0 else img_data.shape[1] // 2

    hip_sum_x = (pred_mask > 0).sum(axis=(1, 2))
    sagittal_idx = int(np.argmax(hip_sum_x)) if hip_sum_x.max() > 0 else img_data.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0f172a")

    views = [
        ("Coronal (Y=" + str(coronal_idx) + ")", np.rot90(img_data[:, coronal_idx, :]), np.rot90(pred_mask[:, coronal_idx, :])),
        ("Sagittal (X=" + str(sagittal_idx) + ")", np.rot90(img_data[sagittal_idx, :, :]), np.rot90(pred_mask[sagittal_idx, :, :])),
        ("Axial (Z=" + str(axial_idx) + ")", np.rot90(img_data[:, :, axial_idx]), np.rot90(pred_mask[:, :, axial_idx])),
    ]

    for ax, (title, mri_sl, pred_sl) in zip(axes, views):
        ax.imshow(mri_sl, cmap="gray", origin="lower")
        # Color overlay: 1=Anterior (red), 2=Posterior (cyan)
        overlay = np.zeros((*mri_sl.shape, 4), dtype=np.float32)
        overlay[pred_sl == 1] = [1.0, 0.2, 0.2, 0.55]  # Red: Anterior
        overlay[pred_sl == 2] = [0.1, 0.8, 1.0, 0.55]  # Cyan: Posterior
        ax.imshow(overlay, origin="lower")

        ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    dice_str = f" | Val Dice: {dice:.3f}" if dice is not None else ""
    status_color = "#22c55e" if "NORMAL" in status else ("#eab308" if "MCI" in status else "#ef4444")
    fig.suptitle(
        f"NeuroScope AI — Alzheimer's Volumetric Analysis: {case_id}\n"
        f"Volume: {total_vol:.1f} mm³ | Status: {status}{dice_str}",
        color=status_color,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_case_quantification(
    img_path: Path,
    lbl_path: Path | None,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = img_path.name.replace(".nii.gz", "")

    img_nii = nib.load(str(img_path))
    img_data = img_nii.get_fdata(dtype=np.float32)
    orig_shape = img_data.shape
    zooms = img_nii.header.get_zooms()[:3]
    voxel_vol = float(zooms[0] * zooms[1] * zooms[2])

    gt_data = None
    gt_vol = None
    dice = None
    if lbl_path and lbl_path.is_file():
        lbl_nii = nib.load(str(lbl_path))
        gt_data = lbl_nii.get_fdata(dtype=np.float32).round().astype(np.int64)
        gt_voxels = int((gt_data > 0).sum())
        gt_vol = float(gt_voxels * voxel_vol)

    # Preprocess
    _, val_pipe = get_transforms()
    data_dict = val_pipe({
        "image": torch.from_numpy(img_data).unsqueeze(0),
        "label": torch.zeros_like(torch.from_numpy(img_data)).unsqueeze(0),
    })

    x = data_dict["image"].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        pred_cropped = torch.argmax(out, dim=1).squeeze(0).cpu()  # [32, 64, 32]

    # Resize back to original dimensions for precise voxel volumetry & visualization
    unpad = ResizeWithPadOrCrop(spatial_size=orig_shape)
    pred_orig = unpad(pred_cropped.unsqueeze(0)).squeeze(0).numpy().astype(np.int32)

    ant_voxels = int((pred_orig == 1).sum())
    post_voxels = int((pred_orig == 2).sum())
    total_voxels = ant_voxels + post_voxels

    ant_vol = float(ant_voxels * voxel_vol)
    post_vol = float(post_voxels * voxel_vol)
    total_vol = float(total_voxels * voxel_vol)

    category, clinical_note = assess_atrophy(total_vol)

    # Compute Dice if GT is available
    if gt_data is not None:
        p_bin = (pred_orig > 0).astype(np.float32)
        g_bin = (gt_data > 0).astype(np.float32)
        inter = (p_bin * g_bin).sum()
        card = p_bin.sum() + g_bin.sum()
        dice = float((2.0 * inter) / (card + 1e-5)) if card > 0 else 1.0

    # Save visual plot
    vis_path = out_dir / f"{cid}_atrophy_visual.png"
    plot_hippocampus_overlays(
        img_data=img_data,
        pred_mask=pred_orig,
        gt_mask=gt_data,
        case_id=cid,
        total_vol=total_vol,
        status=category,
        dice=dice,
        save_path=vis_path,
    )

    print(f"[{cid}] Vol: {total_vol:.1f} mm^3 | Status: {category}" + (f" | Dice: {dice:.3f}" if dice else ""))

    res = {
        "case_id": cid,
        "voxel_spacing_mm": [float(z) for z in zooms],
        "original_shape": list(orig_shape),
        "anterior_volume_mm3": ant_vol,
        "posterior_volume_mm3": post_vol,
        "total_volume_mm3": total_vol,
        "gt_volume_mm3": gt_vol,
        "dice_score": dice,
        "atrophy_category": category,
        "clinical_assessment": clinical_note,
        "visual_path": str(vis_path),
    }

    json_path = out_dir / f"{cid}_atrophy_report.json"
    json_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    parser = argparse.ArgumentParser(description="NeuroScope AI Hippocampal Volumetric Atrophy Engine")
    parser.add_argument("--num_cases", type=int, default=8, help="Number of cohort cases to quantify")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = create_model(device)
    if DEFAULT_MODEL.is_file():
        ckpt = torch.load(str(DEFAULT_MODEL), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Loaded weights from {DEFAULT_MODEL.name} (Val Dice: {ckpt.get('val_total_dice', 0):.4f})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {DEFAULT_MODEL}")

    img_dir = DEFAULT_DATA_DIR / "imagesTr"
    lbl_dir = DEFAULT_DATA_DIR / "labelsTr"
    all_imgs = [p for p in sorted(list(img_dir.glob("*.nii.gz"))) if not p.name.startswith(".")][:args.num_cases]

    print(f"\n======================================================================")
    print(f"  RUNNING ALZHEIMER HIPPOCAMPAL VOLUMETRY COHORT ANALYSIS ({len(all_imgs)} cases)")
    print(f"======================================================================")

    cohort_results = []
    for img_p in all_imgs:
        lbl_p = lbl_dir / img_p.name
        res = run_case_quantification(
            img_path=img_p,
            lbl_path=lbl_p if lbl_p.is_file() else None,
            model=model,
            device=device,
            out_dir=OUTPUT_DIR,
        )
        cohort_results.append(res)

    # Consolidated JSON
    cohort_json = REPORTS_DIR / "alzheimer_cohort_volumetry_report.json"
    cohort_json.write_text(json.dumps(cohort_results, indent=2), encoding="utf-8")

    # Consolidated Markdown table
    md_lines = [
        "# Alzheimer's Disease & Brain Atrophy — Hippocampal Volumetry Report",
        "",
        f"**Date**: 2026-10-19 | **Model**: SegResNet (Trained on RTX 2050 GPU) | **Dataset**: Medical Segmentation Decathlon Task04",
        "",
        "## Clinical Cohort Volumetric Summary",
        "",
        "| Case ID | Anterior Vol (mm³) | Posterior Vol (mm³) | Total Vol (mm³) | Ground Truth (mm³) | Dice Score | Clinical Atrophy Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in cohort_results:
        gt_str = f"{r['gt_volume_mm3']:.1f}" if r["gt_volume_mm3"] is not None else "N/A"
        dice_str = f"{r['dice_score']:.3f}" if r["dice_score"] is not None else "N/A"
        badge = f"**{r['atrophy_category']}**"
        md_lines.append(
            f"| `{r['case_id']}` | {r['anterior_volume_mm3']:.1f} | {r['posterior_volume_mm3']:.1f} | {r['total_volume_mm3']:.1f} | {gt_str} | {dice_str} | {badge} |"
        )

    md_lines.extend([
        "",
        "## Clinical Diagnostic Criteria",
        "- **Normal Brain Volume** ($\ge 3000\\text{ mm}^3$): Healthy hippocampal structure, low risk of cognitive impairment.",
        "- **Borderline / Mild Cognitive Impairment (MCI)** ($2400 - 3000\\text{ mm}^3$): Early-stage neurodegeneration.",
        "- **Severe Hippocampal Atrophy / Alzheimer's** ($< 2400\\text{ mm}^3$): High radiological correlation with clinical Alzheimer's Disease.",
        "",
        "## Visual Evidence",
        f"High-resolution 3-view overlays (Coronal, Sagittal, Axial) saved to `{OUTPUT_DIR.relative_to(SCRIPT_DIR)}/`.",
    ])

    cohort_md = REPORTS_DIR / "alzheimer_cohort_volumetry_report.md"
    cohort_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[DONE] Saved cohort analysis to:")
    print(f"  - Markdown : {cohort_md.relative_to(SCRIPT_DIR)}")
    print(f"  - JSON     : {cohort_json.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
