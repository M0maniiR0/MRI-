#!/usr/bin/env python3
"""
calibrate_native_t1.py
======================
Calibrates 3D DenseNet-121 on the validation split (n=133) using Native T1 cached data:
1. Computes logits on validation set (n=133).
2. Optimizes temperature T > 0 by minimizing ECE.
3. Tunes uncertainty/abstain thresholds on validation set.
4. Generates reliability diagrams (before & after calibration).
5. Saves calibration_params.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
CACHE_DIR = BASE_DIR / "cache_preprocessed"
from monai.networks.nets import DenseNet121

UNCERTAIN_CLASS_ID = 9  # Official index for 'Uncertain/Other'
PRESENT_CLASSES = [1, 2, 3]  # Glioma, Meningioma, Metastasis


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_sizes.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_sizes.append(0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


def run_native_calibration():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[CALIBRATION]: Using device {device}")

    splits_path = CACHE_DIR / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_items = splits["val"]
    print(f"[CALIBRATION]: Validation cohort support n={len(val_items)} subjects")

    ckpt_path = BASE_DIR / "best_model.pt"
    assert ckpt_path.exists(), f"Model checkpoint not found at {ckpt_path}"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=10).to(device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for item in val_items:
            vol = np.load(item["cache_file"]).astype(np.float32)
            tensor_in = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(tensor_in).cpu().numpy().squeeze(0)
            all_logits.append(logits)
            all_targets.append(item["class_id"])

    val_logits = np.array(all_logits)
    val_targets = np.array(all_targets)

    # We evaluate calibration across the present classes (1, 2, 3) mapped to [0, 1, 2]
    # or across the 10 classes
    target_idx = np.array([PRESENT_CLASSES.index(t) for t in val_targets])
    sub_logits = val_logits[:, PRESENT_CLASSES]

    # Probabilities before calibration
    p_uncal = np.exp(sub_logits) / np.sum(np.exp(sub_logits), axis=1, keepdims=True)
    ece_before, acc_b, conf_b, _ = compute_ece(p_uncal, target_idx)
    print(f"[CALIBRATION]: Uncalibrated ECE (Validation n={len(val_items)}) = {ece_before:.4f}")

    # Optimize Temperature T in [0.2, 5.0]
    best_t = 1.0
    best_ece = ece_before
    for cand_t in np.linspace(0.2, 4.0, 77):
        p_cand = np.exp(sub_logits / cand_t) / np.sum(np.exp(sub_logits / cand_t), axis=1, keepdims=True)
        cand_ece, _, _, _ = compute_ece(p_cand, target_idx)
        if cand_ece < best_ece:
            best_ece = cand_ece
            best_t = float(cand_t)

    print(f"[CALIBRATION]: Optimal Temperature T = {best_t:.4f} (ECE reduced to {best_ece:.4f})")

    p_cal = np.exp(sub_logits / best_t) / np.sum(np.exp(sub_logits / best_t), axis=1, keepdims=True)
    ece_after, acc_a, conf_a, _ = compute_ece(p_cal, target_idx)

    # Uncertainty thresholds (abstain threshold on max probability)
    # Analyze max confidence on correct vs incorrect validation predictions
    val_preds = np.argmax(p_cal, axis=1)
    confs = np.max(p_cal, axis=1)
    wrong_mask = (val_preds != target_idx)

    if np.any(wrong_mask):
        max_wrong_conf = float(np.max(confs[wrong_mask]))
        abstain_thresh = min(0.90, max(0.50, max_wrong_conf))
    else:
        abstain_thresh = 0.50

    print(f"[CALIBRATION]: Tuned Abstain / Confidence Threshold = {abstain_thresh:.3f}")

    # Reliability diagram
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_path = reports_dir / "calibration_curve.png"

    bins = np.linspace(0.05, 0.95, 10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax1.bar(bins, acc_b, width=0.08, alpha=0.7, color="salmon", edgecolor="black", label=f"Uncalibrated (ECE={ece_before:.4f})")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability (Before Calibration)")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    ax2.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax2.bar(bins, acc_a, width=0.08, alpha=0.7, color="mediumseagreen", edgecolor="black", label=f"Calibrated (ECE={ece_after:.4f})")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Reliability (After T={best_t:.2f})")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=200)
    plt.close()
    print(f"[CALIBRATION]: Saved reliability diagram to {fig_path}")

    # Save calibration_params.json
    params = {
        "temperature": best_t,
        "ece_before_val": ece_before,
        "ece_after_val": ece_after,
        "confidence_threshold": abstain_thresh,
        "uncertain_class_id": UNCERTAIN_CLASS_ID,
        "present_classes": PRESENT_CLASSES,
        "val_n": len(val_items)
    }
    cal_json_path = BASE_DIR / "calibration_params.json"
    with open(cal_json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"[CALIBRATION]: Parameters saved to {cal_json_path}")
    return params


if __name__ == "__main__":
    run_native_calibration()
