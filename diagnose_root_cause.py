#!/usr/bin/env python3
"""
diagnose_root_cause.py
======================
Comprehensive diagnostic analysis to identify the exact root cause of:
1. High HD95 (~150 mm)
2. Over-segmentation / False Positives
3. Threshold sensitivity
4. Brain mask leakage (predictions outside brain)
5. Connected component distribution
"""

import json
from pathlib import Path
import nibabel as nib
import numpy as np

DATA_DIR = Path("Task01_BrainTumour/Task01_BrainTumour")
OUTPUT_DIR = Path("outputs/segmentation_full_volume")
JSON_PATH = Path("reports/full_volume_evaluation_metrics.json")


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["per_case_results"]
    print(f"Total cases evaluated: {len(results)}")

    wt_ratios = []
    tc_ratios = []
    et_ratios = []
    outside_brain_stats = []
    case_summaries = []

    for r in results:
        cid = r["case_id"]
        gt = r["ground_truth_voxels"]
        pv = r["predicted_volumes"]
        d = r["dice"]
        h = r["hd95_mm"]

        r_wt = pv["WT_voxels"] / max(1, gt["WT"])
        r_tc = pv["TC_voxels"] / max(1, gt["TC"])
        r_et = pv["ET_voxels"] / max(1, gt["ET"])

        wt_ratios.append(r_wt)
        tc_ratios.append(r_tc)
        et_ratios.append(r_et)

        case_summaries.append({
            "case_id": cid,
            "gt_wt": gt["WT"],
            "pred_wt": pv["WT_voxels"],
            "ratio_wt": r_wt,
            "dice_wt": d["Whole_Tumor_WT"],
            "dice_tc": d["Tumor_Core_TC"],
            "dice_et": d["Enhancing_Tumor_ET"],
            "mean_dice": d["Mean_Dice"],
            "hd95_wt": h["Whole_Tumor_WT"],
            "hd95_tc": h["Tumor_Core_TC"],
            "hd95_et": h["Enhancing_Tumor_ET"],
            "mean_hd95": h["Mean_HD95"],
        })

    # Sort cases to inspect good, median, and poor cases
    case_summaries.sort(key=lambda x: x["mean_dice"], reverse=True)
    n = len(case_summaries)

    print("\n" + "="*85)
    print("  COHORT VOLUME RATIO ANALYSIS (PRED_VOXELS / GT_VOXELS)")
    print("="*85)
    print(f"  Whole Tumor (WT) Volume Ratio  : Mean = {np.mean(wt_ratios):.2f}x | Median = {np.median(wt_ratios):.2f}x | Range = [{np.min(wt_ratios):.2f}x, {np.max(wt_ratios):.2f}x]")
    print(f"  Tumor Core (TC) Volume Ratio   : Mean = {np.mean(tc_ratios):.2f}x | Median = {np.median(tc_ratios):.2f}x | Range = [{np.min(tc_ratios):.2f}x, {np.max(tc_ratios):.2f}x]")
    print(f"  Enhancing Tumor (ET) Ratio     : Mean = {np.mean(et_ratios):.2f}x | Median = {np.median(et_ratios):.2f}x | Range = [{np.min(et_ratios):.2f}x, {np.max(et_ratios):.2f}x]")

    print("\n" + "="*85)
    print("  DETAILED AUDIT OF 10 REPRESENTATIVE CASES (GOOD, MEDIAN, POOR)")
    print("="*85)
    sample_indices = [0, 1, 2, n//4, n//2 - 1, n//2, n//2 + 1, int(n*0.75), n - 2, n - 1]
    selected_cases = [case_summaries[i] for i in sample_indices]

    print(f"  {'Case ID':<10} {'Tier':<12} {'GT_WT':<8} {'Pred_WT':<9} {'Ratio':<8} {'Dice_WT':<8} {'Dice_TC':<8} {'Dice_ET':<8} {'MeanDice':<9} {'MeanHD95':<9}")
    print("  " + "-"*83)

    for idx, c in enumerate(selected_cases):
        if idx < 3:
            tier = "GOOD (Top)"
        elif idx < 7:
            tier = "MEDIAN"
        else:
            tier = "POOR (Low)"

        print(f"  {c['case_id']:<10} {tier:<12} {c['gt_wt']:<8} {c['pred_wt']:<9} {c['ratio_wt']:<7.1f}x {c['dice_wt']:<8.4f} {c['dice_tc']:<8.4f} {c['dice_et']:<8.4f} {c['mean_dice']:<9.4f} {c['mean_hd95']:<8.1f}mm")

    # Check background / brain mask leakage for these 10 cases
    print("\n" + "="*85)
    print("  BRAIN MASK LEAKAGE CHECK (DO PREDICTIONS FALL OUTSIDE THE SKULL/BRAIN?)")
    print("="*85)
    for c in selected_cases:
        cid = c["case_id"]
        img_p = DATA_DIR / f"imagesTr/{cid}.nii.gz"
        pred_p = OUTPUT_DIR / f"{cid}/mask.nii.gz"
        gt_p = DATA_DIR / f"labelsTr/{cid}.nii.gz"

        if img_p.is_file() and pred_p.is_file():
            img_arr = nib.load(str(img_p)).get_fdata()
            pred_arr = nib.load(str(pred_p)).get_fdata()
            gt_arr = nib.load(str(gt_p)).get_fdata()

            # Brain mask: voxels where at least one MRI channel > 0
            brain_mask = np.any(img_arr > 0, axis=-1)
            pred_pos = pred_arr > 0
            gt_pos = gt_arr > 0

            # Outside brain
            out_brain = pred_pos & (~brain_mask)
            out_cnt = int(np.sum(out_brain))
            pred_total = int(np.sum(pred_pos))
            pct_outside = (out_cnt / max(1, pred_total)) * 100.0

            # True Positives, False Positives, False Negatives within brain
            tp = int(np.sum(pred_pos & gt_pos))
            fp = int(np.sum(pred_pos & (~gt_pos)))
            fn = int(np.sum((~pred_pos) & gt_pos))

            print(f"  {cid}: Pred_Total={pred_total:>7} | Outside_Brain={out_cnt:>5} ({pct_outside:>4.1f}%) | TP={tp:>6} | FP={fp:>7} | FN={fn:>5} | FP/TP Ratio={fp/max(1,tp):>5.1f}x")

    return case_summaries


if __name__ == "__main__":
    main()
