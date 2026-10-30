#!/usr/bin/env python3
"""
evaluate_native_t1_bootstrap.py
===============================
Evaluates 3D DenseNet-121 on the held-out test split (n=132 native T1 subjects):
- Loads calibrated temperature and abstain threshold from calibration_params.json (fit on validation only).
- Computes Balanced Accuracy, Macro-F1, per-class Recall, Precision, and OvR AUROC.
- Performs 1,000 bootstrap resamples to report empirical 95% Confidence Intervals.
- Computes Expected Calibration Error (ECE) before and after temperature scaling.
- Generates confusion matrix and calibration impact.
- Reports only factual numbers without subjective adjectives ("PASS", "excellent").
- Exports results to test_evaluation_results.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from monai.networks.nets import DenseNet121

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
CACHE_DIR = BASE_DIR / "cache_preprocessed"


def compute_binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    matches = (pos[:, None] > neg[None, :]).sum()
    ties = 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float((matches + ties) / (n_pos * n_neg))


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc_bin = np.mean(accuracies[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += prop * np.abs(acc_bin - conf_bin)
    return float(ece)


def run_evaluation(num_bootstraps: int = 1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================ TEST EVALUATION (BOOTSTRAP B={num_bootstraps}) ================")

    splits_path = CACHE_DIR / "splits.json"
    assert splits_path.exists(), "splits.json not found"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_items = splits["test"]
    n_test = len(test_items)
    print(f"Total Test Set Support: n={n_test} subjects")

    # Class ID mapping: Glioma=1, Meningioma=2, Metastasis=3
    class_map = {1: "Glioma", 2: "Meningioma", 3: "Metastasis"}
    present_classes = [1, 2, 3]

    for cid in present_classes:
        cnt = sum(1 for it in test_items if it["class_id"] == cid)
        print(f"  - Class {cid} ({class_map[cid]}): n={cnt}")

    # Load calibration parameters fitted on validation set
    cal_params_path = BASE_DIR / "calibration_params.json"
    if cal_params_path.exists():
        with open(cal_params_path, "r", encoding="utf-8") as f:
            cal_params = json.load(f)
        T = cal_params.get("temperature", 1.0)
        conf_thresh = cal_params.get("confidence_threshold", 0.50)
        print(f"Loaded validation-fitted parameters: Temperature T={T:.4f}, Abstain Threshold={conf_thresh:.3f}")
    else:
        T = 1.0
        conf_thresh = 0.50
        print("Warning: calibration_params.json not found, using default T=1.0, conf_thresh=0.50")

    # Load trained model
    ckpt_path = BASE_DIR / "best_model.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=10).to(device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Forward pass on all test items
    all_logits = []
    all_targets = []
    all_sids = []

    with torch.no_grad():
        for it in test_items:
            vol = np.load(it["cache_file"]).astype(np.float32)
            tensor_in = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(tensor_in).cpu().numpy().squeeze(0)
            all_logits.append(logits)
            all_targets.append(it["class_id"])
            all_sids.append(it["subject_id"])

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)

    # Sub-logits for present classes [1, 2, 3]
    sub_logits = all_logits[:, present_classes]
    target_idx = np.array([present_classes.index(t) for t in all_targets])

    # Uncalibrated and Calibrated probabilities over the active classes
    p_uncal = np.exp(sub_logits) / np.sum(np.exp(sub_logits), axis=1, keepdims=True)
    p_cal = np.exp(sub_logits / T) / np.sum(np.exp(sub_logits / T), axis=1, keepdims=True)

    ece_test_before = compute_ece(p_uncal, target_idx)
    ece_test_after = compute_ece(p_cal, target_idx)

    # Predictions
    pred_idx = np.argmax(p_cal, axis=1)
    active_preds = np.array([present_classes[i] for i in pred_idx])

    # Abstain check
    max_p = np.max(p_cal, axis=1)
    abstained_mask = (max_p < conf_thresh)
    n_abstained = int(np.sum(abstained_mask))

    # Confusion matrix on present classes
    conf_matrix = np.zeros((len(present_classes), len(present_classes)), dtype=int)
    for t_i, p_i in zip(target_idx, pred_idx):
        conf_matrix[t_i, p_i] += 1

    # Point Estimates
    recalls, precisions, f1s, aurocs = {}, {}, {}, {}
    for cid in present_classes:
        cname = class_map[cid]
        tp = np.sum((active_preds == cid) & (all_targets == cid))
        fn = np.sum((active_preds != cid) & (all_targets == cid))
        fp = np.sum((active_preds == cid) & (all_targets != cid))

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec + 1e-8)) if (prec + rec) > 0 else 0.0

        # OvR AUROC using calibrated probability
        y_bin = (all_targets == cid).astype(int)
        score_bin = p_cal[:, present_classes.index(cid)]
        auroc = compute_binary_auroc(y_bin, score_bin)

        recalls[cname] = rec
        precisions[cname] = prec
        f1s[cname] = f1
        aurocs[cname] = auroc

    point_bal_acc = float(np.mean(list(recalls.values())))
    point_macro_f1 = float(np.mean(list(f1s.values())))

    # Bootstrap 95% Confidence Intervals (B = num_bootstraps)
    rng = np.random.default_rng(42)
    boot_bal_acc = []
    boot_macro_f1 = []
    boot_recalls = {cname: [] for cname in class_map.values()}
    boot_precisions = {cname: [] for cname in class_map.values()}
    boot_aurocs = {cname: [] for cname in class_map.values()}

    for _ in range(num_bootstraps):
        b_idx = rng.choice(n_test, size=n_test, replace=True)
        b_targets = all_targets[b_idx]
        b_preds = active_preds[b_idx]
        b_p_cal = p_cal[b_idx]

        b_recs, b_precs, b_f1_list = [], [], []
        for cid in present_classes:
            cname = class_map[cid]
            tp = np.sum((b_preds == cid) & (b_targets == cid))
            fn = np.sum((b_preds != cid) & (b_targets == cid))
            fp = np.sum((b_preds == cid) & (b_targets != cid))

            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f = float(2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0

            y_b = (b_targets == cid).astype(int)
            s_b = b_p_cal[:, present_classes.index(cid)]
            auc = compute_binary_auroc(y_b, s_b)

            boot_recalls[cname].append(r)
            boot_precisions[cname].append(p)
            if not np.isnan(auc):
                boot_aurocs[cname].append(auc)

            b_recs.append(r)
            b_precs.append(p)
            b_f1_list.append(f)

        boot_bal_acc.append(np.mean(b_recs))
        boot_macro_f1.append(np.mean(b_f1_list))

    bal_acc_ci = [float(np.percentile(boot_bal_acc, 2.5)), float(np.percentile(boot_bal_acc, 97.5))]
    macro_f1_ci = [float(np.percentile(boot_macro_f1, 2.5)), float(np.percentile(boot_macro_f1, 97.5))]

    print("\n---------------- TEST METRICS (Point Estimate & Empirical 95% CI) ----------------")
    print(f"Overall Balanced Accuracy : {point_bal_acc*100:.2f}% (95% CI: [{bal_acc_ci[0]*100:.2f}%, {bal_acc_ci[1]*100:.2f}%])")
    print(f"Overall Macro-F1 Score    : {point_macro_f1*100:.2f}% (95% CI: [{macro_f1_ci[0]*100:.2f}%, {macro_f1_ci[1]*100:.2f}%])")
    print(f"Expected Calibration Error: Before T = {ece_test_before:.4f} | After T = {ece_test_after:.4f}")
    print(f"Abstained Scans (P < {conf_thresh:.2f}): {n_abstained}/{n_test} ({n_abstained/n_test*100:.1f}%)")

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Prediction):")
    print("                Glioma  Meningioma  Metastasis")
    for i, cname in enumerate(["Glioma", "Meningioma", "Metastasis"]):
        print(f"  {cname:<12}:   {conf_matrix[i, 0]:<7} {conf_matrix[i, 1]:<11} {conf_matrix[i, 2]}")

    per_class_summary = {}
    print("\nPer-Class Breakdown:")
    for cid in present_classes:
        cname = class_map[cid]
        cnt = sum(1 for it in test_items if it["class_id"] == cid)
        r_ci = [float(np.percentile(boot_recalls[cname], 2.5)), float(np.percentile(boot_recalls[cname], 97.5))]
        p_ci = [float(np.percentile(boot_precisions[cname], 2.5)), float(np.percentile(boot_precisions[cname], 97.5))]
        a_ci = [float(np.percentile(boot_aurocs[cname], 2.5)), float(np.percentile(boot_aurocs[cname], 97.5))] if boot_aurocs[cname] else [0.0, 0.0]

        print(f"\n  Class: {cname} (Test Support n={cnt})")
        print(f"    - Recall (Sensitivity) : {recalls[cname]*100:.2f}% (95% CI: [{r_ci[0]*100:.2f}%, {r_ci[1]*100:.2f}%])")
        print(f"    - Precision            : {precisions[cname]*100:.2f}% (95% CI: [{p_ci[0]*100:.2f}%, {p_ci[1]*100:.2f}%])")
        print(f"    - F1-Score             : {f1s[cname]*100:.2f}%")
        print(f"    - One-vs-Rest AUROC    : {aurocs[cname]:.4f} (95% CI: [{a_ci[0]:.4f}, {a_ci[1]:.4f}])")

        per_class_summary[cname] = {
            "support_n": cnt,
            "recall": recalls[cname],
            "recall_95ci": r_ci,
            "precision": precisions[cname],
            "precision_95ci": p_ci,
            "f1": f1s[cname],
            "auroc": aurocs[cname],
            "auroc_95ci": a_ci
        }

    # Document Coverage Gaps
    print("\nDocumented Coverage Gaps (No Local Whole-Brain Data, Routed to Uncertain/Other):")
    gaps = ["Normal", "Alzheimer's", "FCD epilepsy", "Post-stroke lesion", "Parkinson's", "ASD", "Uncertain/Other"]
    for g in gaps:
        print(f"  - {g}: n=0 (Coverage Gap)")

    # Save detailed JSON output
    out_results = {
        "test_support_n": n_test,
        "balanced_accuracy": point_bal_acc,
        "balanced_accuracy_95ci": bal_acc_ci,
        "macro_f1": point_macro_f1,
        "macro_f1_95ci": macro_f1_ci,
        "ece_before_cal": ece_test_before,
        "ece_after_cal": ece_test_after,
        "temperature_used": T,
        "abstain_threshold_used": conf_thresh,
        "n_abstained": n_abstained,
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": per_class_summary,
        "coverage_gaps": gaps
    }
    res_path = BASE_DIR / "test_evaluation_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)

    print(f"\n[OK] Results exported to {res_path}")


if __name__ == "__main__":
    run_evaluation(num_bootstraps=1000)
