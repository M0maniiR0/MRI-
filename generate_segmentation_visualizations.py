#!/usr/bin/env python3
"""
generate_segmentation_visualizations.py
======================================
Publication-Grade Visual Comparison Generator for Brain Tumor Segmentation (BraTS / MONAI SegResNet 3D).
Generates 4-panel visual comparisons:
  [Original MRI FLAIR] | [Ground Truth Mask] | [Predicted SegResNet Mask] | [Alpha-Blended Overlay]

Selects representative cases across the cohort performance spectrum:
- High / Good cases (top quartile)
- Median / Average cases (around median)
- Challenging / Poor cases (lower quartile)

Uses PIL and NumPy (100% offline, zero external internet dependencies).
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "outputs" / "segmentation_finetuned"
    if (SCRIPT_DIR / "outputs" / "segmentation_finetuned").is_dir()
    else SCRIPT_DIR / "outputs" / "segmentation_full_volume"
)
VIS_DIR = DEFAULT_OUTPUT_DIR / "visualizations"
CSV_METRICS = SCRIPT_DIR / "reports" / "segmentation_per_case_metrics.csv"
JSON_METRICS = (
    SCRIPT_DIR / "reports" / "finetuned_evaluation_metrics.json"
    if (SCRIPT_DIR / "reports" / "finetuned_evaluation_metrics.json").is_file()
    else SCRIPT_DIR / "reports" / "full_volume_evaluation_metrics.json"
)

# Color Palette: MSD Task01 labels {1: edema, 2: NET, 3: ET}
COLORS = {
    0: (0, 0, 0),        # Background
    1: (46, 204, 113),   # Edema (Emerald Green)
    2: (52, 152, 219),   # Non-Enhancing Core (Sky Blue)
    3: (231, 76, 60),    # Enhancing Tumor (Coral Red)
}


def to_grayscale_u8(arr: np.ndarray) -> np.ndarray:
    """Clips robust 1st-99th percentiles of non-zero MRI brain intensities to 8-bit grayscale."""
    nonzeros = arr[arr > 0]
    if len(nonzeros) > 0:
        p1, p99 = np.percentile(nonzeros, (1.0, 99.0))
    else:
        p1, p99 = 0.0, 1.0
    if p99 <= p1:
        p99 = p1 + 1e-5
    clipped = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def create_color_mask(mask_slice: np.ndarray) -> np.ndarray:
    """Converts 2D integer mask {0, 1, 2, 3} to RGB array."""
    h, w = mask_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, col in COLORS.items():
        if lbl > 0:
            rgb[mask_slice == lbl] = col
    return rgb


def create_overlay(gray_u8: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Alpha blends colored segmentation mask on top of grayscale MRI image."""
    base_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    overlay = base_rgb.copy()
    
    for lbl, col in COLORS.items():
        if lbl > 0:
            match = (mask_slice == lbl)
            if np.any(match):
                c_arr = np.array(col, dtype=np.float32)
                overlay[match] = (1.0 - alpha) * base_rgb[match] + alpha * c_arr

    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_panel(img_u8: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    """Renders a single visual panel with header title bar."""
    # Rotate 90 deg counter-clockwise so axial slice is upright (radiological orientation)
    upright = np.rot90(img_u8)
    h, w, c = upright.shape
    
    header_h = 44
    panel_img = Image.new("RGB", (w, h + header_h), color=(24, 28, 36))
    draw = ImageDraw.Draw(panel_img)
    
    # Draw header text
    draw.text((10, 6), title, fill=(240, 245, 255))
    if subtitle:
        draw.text((10, 24), subtitle, fill=(160, 175, 200))
        
    # Paste image
    pil_sub = Image.fromarray(upright)
    panel_img.paste(pil_sub, (0, header_h))
    
    return panel_img


def generate_single_case_figure(
    case_id: str,
    data_dir: Path,
    output_dir: Path,
    vis_dir: Path,
    metrics: Dict[str, Any],
) -> Path:
    """Generates and saves the 4-panel comparison for a single case."""
    img_file = data_dir / f"imagesTr/{case_id}.nii.gz"
    gt_file = data_dir / f"labelsTr/{case_id}.nii.gz"
    pred_file = output_dir / case_id / "mask.nii.gz"

    if not img_file.is_file() or not gt_file.is_file() or not pred_file.is_file():
        raise FileNotFoundError(f"Missing files for {case_id}: img={img_file.is_file()}, gt={gt_file.is_file()}, pred={pred_file.is_file()}")

    # Load NIfTI volumes
    img_nii = nib.load(str(img_file))
    gt_nii = nib.load(str(gt_file))
    pred_nii = nib.load(str(pred_file))

    img_data = img_nii.get_fdata()
    gt_data = gt_nii.get_fdata().round().astype(np.int32)
    pred_data = pred_nii.get_fdata().round().astype(np.int32)

    # Channel 0 of 4D image is FLAIR
    flair_vol = img_data[..., 0]

    # Find optimal axial slice (maximum tumor area in GT or Pred)
    tumor_vox_per_slice = (gt_data > 0).sum(axis=(0, 1))
    if tumor_vox_per_slice.max() == 0:
        tumor_vox_per_slice = (pred_data > 0).sum(axis=(0, 1))
    best_slice_idx = int(np.argmax(tumor_vox_per_slice))

    flair_slice = flair_vol[:, :, best_slice_idx]
    gt_slice = gt_data[:, :, best_slice_idx]
    pred_slice = pred_data[:, :, best_slice_idx]

    flair_u8 = to_grayscale_u8(flair_slice)
    flair_rgb = np.stack([flair_u8, flair_u8, flair_u8], axis=-1)

    gt_color = create_color_mask(gt_slice)
    pred_color = create_color_mask(pred_slice)
    overlay_pred = create_overlay(flair_u8, pred_slice, alpha=0.55)

    # Subtitles with Dice scores
    dice_wt = metrics.get("WT_Dice", 0.0)
    dice_tc = metrics.get("TC_Dice", 0.0)
    dice_et = metrics.get("ET_Dice", 0.0)
    mean_dice = metrics.get("Mean_Dice", 0.0)

    p1 = render_panel(flair_rgb, "1. MRI Modality", f"FLAIR (Axial Slice z={best_slice_idx})")
    p2 = render_panel(gt_color, "2. Ground Truth", "Manual Radiologist Annotations")
    p3 = render_panel(pred_color, "3. SegResNet 3D Pred", f"Dice: WT={dice_wt:.2f} | TC={dice_tc:.2f} | ET={dice_et:.2f}")
    p4 = render_panel(overlay_pred, "4. Overlay on FLAIR", f"Overall Mean Dice = {mean_dice:.4f}")

    # Stitch the 4 panels horizontally with vertical dividers
    panel_w, panel_h = p1.size
    divider_w = 4
    total_w = 4 * panel_w + 3 * divider_w
    top_banner_h = 60
    bottom_legend_h = 45
    total_h = panel_h + top_banner_h + bottom_legend_h

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 18, 25))
    draw = ImageDraw.Draw(canvas)

    # Banner text
    title_text = f"BrainAgenticthon 2026 - Glioma 3D Segmentation | Case: {case_id}"
    score_text = f"Mean Dice: {mean_dice:.4f}  (WT: {dice_wt:.4f}, TC: {dice_tc:.4f}, ET: {dice_et:.4f})"
    draw.text((20, 10), title_text, fill=(255, 255, 255))
    draw.text((20, 32), score_text, fill=(80, 200, 255))

    # Paste panels
    x_offset = 0
    for idx, p in enumerate([p1, p2, p3, p4]):
        canvas.paste(p, (x_offset, top_banner_h))
        x_offset += panel_w
        if idx < 3:
            # Draw divider
            draw.rectangle([x_offset, top_banner_h, x_offset + divider_w, top_banner_h + panel_h], fill=(45, 55, 75))
            x_offset += divider_w

    # Bottom Legend
    leg_y = top_banner_h + panel_h + 12
    draw.text((20, leg_y), "Legend:", fill=(200, 210, 230))
    
    # Legend items
    leg_items = [
        ((46, 204, 113), "Green: Peritumoral Edema (ED, Label 1)"),
        ((52, 152, 219), "Blue: Non-Enhancing Core / Necrosis (NET, Label 2)"),
        ((231, 76, 60),  "Red: Enhancing Active Tumor (ET, Label 3)"),
    ]
    cur_x = 90
    for col, text in leg_items:
        draw.rectangle([cur_x, leg_y + 2, cur_x + 14, leg_y + 16], fill=col)
        draw.text((cur_x + 20, leg_y), text, fill=(230, 235, 245))
        cur_x += 320

    vis_dir.mkdir(parents=True, exist_ok=True)
    out_file = vis_dir / f"{case_id}_comparison.png"
    canvas.save(str(out_file), quality=95)
    print(f"  [SAVED] {out_file.name} (Mean Dice: {mean_dice:.4f})")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate publication visual comparisons")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Task01_BrainTumour root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to segmentation predictions")
    parser.add_argument("--report-file", type=Path, default=None, help="Path to evaluation metrics JSON or CSV")
    parser.add_argument("--vis-dir", type=Path, default=None, help="Path to save visualization PNGs")
    args = parser.parse_args()

    data_dir = args.data.resolve()
    output_dir = args.output_dir.resolve()
    vis_dir = args.vis_dir.resolve() if args.vis_dir else (output_dir / "visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Determine report file
    report_candidate = args.report_file.resolve() if args.report_file else None
    if report_candidate is None:
        if (SCRIPT_DIR / "reports" / "finetuned_evaluation_metrics.json").is_file():
            report_candidate = SCRIPT_DIR / "reports" / "finetuned_evaluation_metrics.json"
        elif CSV_METRICS.is_file():
            report_candidate = CSV_METRICS
        else:
            report_candidate = JSON_METRICS

    # 1. Read metrics from CSV or JSON
    cases = []
    if report_candidate and report_candidate.suffix == ".csv" and report_candidate.is_file():
        with open(report_candidate, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "SUCCESS":
                    cases.append({
                        "case_id": row["case_id"],
                        "Mean_Dice": float(row.get("Mean_Dice", 0.0)),
                        "WT_Dice": float(row.get("WT_Dice", 0.0)),
                        "TC_Dice": float(row.get("TC_Dice", 0.0)),
                        "ET_Dice": float(row.get("ET_Dice", 0.0)),
                    })
    elif report_candidate and report_candidate.is_file():
        with open(report_candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data.get("per_case_results", []):
                d = r["dice"]
                cases.append({
                    "case_id": r["case_id"],
                    "Mean_Dice": float(d["Mean_Dice"]),
                    "WT_Dice": float(d["Whole_Tumor_WT"]),
                    "TC_Dice": float(d["Tumor_Core_TC"]),
                    "ET_Dice": float(d["Enhancing_Tumor_ET"]),
                })

    if not cases:
        print(f"[ERROR] No evaluated cases found in {report_candidate}. Ensure full evaluation completes first.")
        return

    # Sort cases by Mean_Dice
    cases.sort(key=lambda x: x["Mean_Dice"], reverse=True)
    n = len(cases)
    print(f"Total evaluated cases available: {n}")

    # Select representative cases:
    top_indices = [0, max(1, n // 8)]
    median_indices = [n // 2 - 1, n // 2]
    lower_indices = [int(n * 0.8), int(n * 0.95)]
    
    selected_indices = list(dict.fromkeys(top_indices + median_indices + lower_indices))[:6]
    selected_cases = [cases[idx] for idx in selected_indices if idx < n]

    print(f"Selected {len(selected_cases)} representative cases for visualization:")
    for idx, c in enumerate(selected_cases, start=1):
        tier = "HIGH/GOOD" if idx <= 2 else ("MEDIAN/AVERAGE" if idx <= 4 else "CHALLENGING/POOR")
        print(f"  {idx}. [{tier}] {c['case_id']}: Mean Dice = {c['Mean_Dice']:.4f} (WT={c['WT_Dice']:.4f}, TC={c['TC_Dice']:.4f}, ET={c['ET_Dice']:.4f})")

    generated_paths = []
    for c in selected_cases:
        p = generate_single_case_figure(
            case_id=c["case_id"],
            data_dir=data_dir,
            output_dir=output_dir,
            vis_dir=vis_dir,
            metrics=c,
        )
        generated_paths.append(str(p))

    print(f"\n[DONE] Successfully generated {len(generated_paths)} visual comparisons in {vis_dir}")


if __name__ == "__main__":
    main()
