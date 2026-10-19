#!/usr/bin/env python3
"""
evaluate_meningioma_cases.py
============================
NeuroScope AI — Meningioma (BraTS-MEN 2023) Evaluation & Visualization Engine.
Performs sliding-window inference on 3D 4-channel MRI scans.
Measures Whole Tumor (WT) volume (mm^3) and calculates Dice score.
Generates publication-quality 3-view visual overlays (Axial, Coronal, Sagittal).
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
from monai.inferers import sliding_window_inference

from train_meningioma_transfer import create_model, get_meningioma_transforms, compute_region_dice

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Task_Meningioma" / "BraTS_MEN"
DEFAULT_MODEL = SCRIPT_DIR / "models" / "segresnet_meningioma_best.pt"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "meningioma_analysis"
REPORTS_DIR = SCRIPT_DIR / "reports"


def plot_meningioma_overlays(
    mri_t1c: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    case_id: str,
    tumor_vol_mm3: float,
    dice: float | None,
    save_path: Path,
):
    """Generates a high-contrast 3-view panel (Coronal, Sagittal, Axial) with tumor mask."""
    # Find slice with maximum tumor presence
    tumor_sum_z = (pred_mask > 0).sum(axis=(0, 1))
    axial_idx = int(np.argmax(tumor_sum_z)) if tumor_sum_z.max() > 0 else mri_t1c.shape[2] // 2

    tumor_sum_y = (pred_mask > 0).sum(axis=(0, 2))
    coronal_idx = int(np.argmax(tumor_sum_y)) if tumor_sum_y.max() > 0 else mri_t1c.shape[1] // 2

    tumor_sum_x = (pred_mask > 0).sum(axis=(1, 2))
    sagittal_idx = int(np.argmax(tumor_sum_x)) if tumor_sum_x.max() > 0 else mri_t1c.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#090d16")

    views = [
        ("Coronal (Y=" + str(coronal_idx) + ")", np.rot90(mri_t1c[:, coronal_idx, :]), np.rot90(pred_mask[:, coronal_idx, :])),
        ("Sagittal (X=" + str(sagittal_idx) + ")", np.rot90(mri_t1c[sagittal_idx, :, :]), np.rot90(pred_mask[sagittal_idx, :, :])),
        ("Axial (Z=" + str(axial_idx) + ")", np.rot90(mri_t1c[:, :, axial_idx]), np.rot90(pred_mask[:, :, axial_idx])),
    ]

    for ax, (title, mri_sl, pred_sl) in zip(axes, views):
        # Normalize MRI slice for clear visualization
        norm_sl = np.clip((mri_sl - mri_sl.min()) / (mri_sl.max() - mri_sl.min() + 1e-5), 0, 1)
        ax.imshow(norm_sl, cmap="gray", origin="lower")

        # Color overlay: Amber/Red for Meningioma Whole Tumor
        overlay = np.zeros((*norm_sl.shape, 4), dtype=np.float32)
        overlay[pred_sl > 0] = [1.0, 0.35, 0.05, 0.65]  # Bright orange-red
        ax.imshow(overlay, origin="lower")

        ax.set_title(title, color="#f1f5f9", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    dice_str = f" | Dice: {dice:.3f}" if dice is not None else ""
    fig.suptitle(
        f"NeuroScope AI — Meningioma Segmentation & Volumetry: {case_id}\n"
        f"Whole Tumor Volume: {tumor_vol_mm3:,.1f} mm³{dice_str}",
        color="#fb923c",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate_single_case(
    case_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: Tuple[int, int, int] = (96, 96, 96),
) -> Dict:
    cid = case_dir.name
    t1n = case_dir / f"{cid}-t1n.nii.gz"
    t1c = case_dir / f"{cid}-t1c.nii.gz"
    t2f = case_dir / f"{cid}-t2f.nii.gz"
    t2w = case_dir / f"{cid}-t2w.nii.gz"
    seg = case_dir / f"{cid}-seg.nii.gz"

    t1c_nii = nib.load(str(t1c))
    t1c_data = t1c_nii.get_fdata(dtype=np.float32)
    zooms = t1c_nii.header.get_zooms()[:3]
    voxel_vol = float(zooms[0] * zooms[1] * zooms[2])

    # Stack 4 modalities
    mods = [nib.load(str(p)).get_fdata(dtype=np.float32) for p in [t1n, t1c, t2f, t2w]]
    img_4d = np.stack(mods, axis=0)  # [4, H, W, D]

    _, val_pipe = get_meningioma_transforms()
    item = val_pipe({
        "image": torch.from_numpy(img_4d),
        "label": torch.zeros((1, *img_4d.shape[1:])),
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
        prob = torch.sigmoid(out).squeeze(0)
        brain_mask = (item["image"].abs().sum(dim=0) > 0).to(device)
        pred_wt = ((prob[0] > 0.5) & brain_mask).cpu().numpy().astype(np.uint8)

    tumor_voxels = int(pred_wt.sum())
    tumor_vol = tumor_voxels * voxel_vol

    dice = None
    if seg.is_file():
        seg_nii = nib.load(str(seg))
        gt = (seg_nii.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
        p_t = torch.from_numpy(pred_wt).float()
        g_t = torch.from_numpy(gt).float()
        dice = compute_region_dice(p_t, g_t)

    vis_path = OUTPUT_DIR / f"{cid}_meningioma_visual.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_meningioma_overlays(
        mri_t1c=t1c_data,
        pred_mask=pred_wt,
        gt_mask=None,
        case_id=cid,
        tumor_vol_mm3=tumor_vol,
        dice=dice,
        save_path=vis_path,
    )

    print(f"[{cid}] Meningioma Vol: {tumor_vol:,.1f} mm³" + (f" | Dice: {dice:.4f}" if dice is not None else ""))

    res = {
        "case_id": cid,
        "voxel_spacing_mm": [float(z) for z in zooms],
        "tumor_volume_mm3": float(tumor_vol),
        "tumor_voxels": tumor_voxels,
        "dice_score": float(dice) if dice is not None else None,
        "visual_path": str(vis_path),
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate Meningioma cohort")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of cases to evaluate and visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = create_model(device)
    if DEFAULT_MODEL.is_file():
        ckpt = torch.load(str(DEFAULT_MODEL), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Loaded weights from {DEFAULT_MODEL.name} (Val Dice: {ckpt.get('val_wt_dice', 0):.4f})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {DEFAULT_MODEL}")

    case_dirs = sorted([d for d in DEFAULT_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("BraTS-MEN")])
    eval_cases = case_dirs[-args.num_cases:]

    print(f"\n======================================================================")
    print(f"  RUNNING MENINGIOMA VOLUMETRIC EVALUATION ({len(eval_cases)} validation cases)")
    print(f"======================================================================")

    results = []
    for cd in eval_cases:
        res = evaluate_single_case(cd, model, device)
        results.append(res)

    # Save summary markdown
    md_lines = [
        "# Meningioma (BraTS-MEN 2023) — Volumetric Evaluation Report",
        "",
        "**Date**: 2026-10-19 | **Model**: SegResNet Transfer Learning | **Dataset**: BraTS-MEN 2023",
        "",
        "| Case ID | Tumor Volume (mm³) | Tumor Voxels | Dice Similarity | Clinical Finding |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    for r in results:
        dice_s = f"{r['dice_score']:.3f}" if r["dice_score"] is not None else "N/A"
        finding = "**Detected Meningioma**" if r["tumor_volume_mm3"] > 100 else "Minimal / Residual"
        md_lines.append(f"| `{r['case_id']}` | {r['tumor_volume_mm3']:,.1f} | {r['tumor_voxels']:,} | {dice_s} | {finding} |")

    md_lines.extend([
        "",
        "## Diagnostic Overlays",
        f"High-contrast 3-plane views (Coronal, Sagittal, Axial) saved to `{OUTPUT_DIR.relative_to(SCRIPT_DIR)}/`.",
    ])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_md = REPORTS_DIR / "meningioma_cohort_report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    report_json = REPORTS_DIR / "meningioma_cohort_report.json"
    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[DONE] Saved Meningioma Cohort Report to:")
    print(f"  - Markdown: {report_md.relative_to(SCRIPT_DIR)}")
    print(f"  - JSON    : {report_json.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
