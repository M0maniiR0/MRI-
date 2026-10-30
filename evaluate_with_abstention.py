#!/usr/bin/env python3
"""
evaluate_with_abstention.py
===========================
Evaluates 3D DenseNet-121 performance taking into account the safety abstention policy:
- Reuses the exact same held-out test split (n=132 patients).
- Calibrated probabilities using temperature T = 0.4000 (from validation).
- Abstain threshold = 0.900 (from validation).
- Computes:
  1. Per-class and total abstention rates.
  2. Selective performance on non-abstained cases only (coverage, accuracy, macro-F1, balanced accuracy).
  3. Deployment-view metrics: treating abstained scans as 'Uncertain/Other' (a missed call / false negative
     for the underlying tumor class). Computes Balanced Accuracy, Macro-F1, and the 3x4 confusion matrix
     with B=1000 bootstrap 95% Confidence Intervals.
  4. Side-by-side comparison of argmax-only vs. deployment-view metrics.
Saves results to abstention_evaluation_results.json.
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

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache_preprocessed"


def run_evaluation(num_bootstraps: int = 1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================ EVALUATION WITH ABSTENTION (B={num_bootstraps}) ================")

    splits_path = CACHE_DIR / "splits.json"
    assert splits_path.exists(), f"splits.json not found at {splits_path}"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_items = splits["test"]
    n_test = len(test_items)
    print(f"Total Test Set Support: n={n_test} patients (1 scan per unique patient)")

    class_map = {1: "Glioma", 2: "Meningioma", 3: "Metastasis"}
    present_classes = [1, 2, 3]

    # Load calibration parameters
    cal_params_path = BASE_DIR / "calibration_params.json"
    assert cal_params_path.exists(), f"Missing {cal_params_path}"
    with open(cal_params_path, "r", encoding="utf-8") as f:
        cal_params = json.load(f)

    T = float(cal_params.get("temperature", 0.4000))
    conf_thresh = float(cal_params.get("confidence_threshold", 0.900))
    print(f"Calibration Parameters: T={T:.4f}, Abstain Threshold={conf_thresh:.3f}")

    # Load model
    ckpt_path = BASE_DIR / "best_model.pt"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=10).to(device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_logits, all_targets, all_sids = [], [], []
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

    # Sub-logits for the 3 active classes [1, 2, 3]
    sub_logits = all_logits[:, present_classes]
    p_cal = np.exp(sub_logits / T) / np.sum(np.exp(sub_logits / T), axis=1, keepdims=True)

    # Argmax prediction and max confidence
    argmax_pred_idx = np.argmax(p_cal, axis=1)
    argmax_preds = np.array([present_classes[i] for i in argmax_pred_idx])
    max_confs = np.max(p_cal, axis=1)

    # Abstention mask (confidence < threshold)
    abstained_mask = (max_confs < conf_thresh)
    n_abstained = int(np.sum(abstained_mask))
    coverage = float(1.0 - (n_abstained / n_test))

    # --- Part A: Abstention Breakdown ---
    print("\n--- Part A: Abstention Breakdown ---")
    print(f"Total Abstained Scans: {n_abstained}/{n_test} ({n_abstained/n_test*100:.2f}%)")
    print(f"Cohort Coverage      : {coverage*100:.2f}% ({n_test - n_abstained}/{n_test} decided)")

    per_class_abstention = {}
    for cid in present_classes:
        cname = class_map[cid]
        c_mask = (all_targets == cid)
        c_tot = int(np.sum(c_mask))
        c_abs = int(np.sum(abstained_mask[c_mask]))
        per_class_abstention[cname] = {
            "total_n": c_tot,
            "abstained_n": c_abs,
            "abstention_rate": float(c_abs / c_tot) if c_tot > 0 else 0.0
        }
        print(f"  - {cname:<12}: {c_abs}/{c_tot} abstained ({c_abs/c_tot*100:.2f}%)")

    # --- Part B: Selective Performance on Non-Abstained Cases ---
    print("\n--- Part B: Selective Performance (Non-Abstained Cases Only) ---")
    non_abs_idx = np.where(~abstained_mask)[0]
    n_non_abs = len(non_abs_idx)

    y_true_sel = all_targets[non_abs_idx]
    y_pred_sel = argmax_preds[non_abs_idx]

    sel_acc = float(np.mean(y_true_sel == y_pred_sel)) if n_non_abs > 0 else 0.0

    sel_recs, sel_precs, sel_f1s = {}, {}, {}
    for cid in present_classes:
        cname = class_map[cid]
        tp = np.sum((y_pred_sel == cid) & (y_true_sel == cid))
        fn = np.sum((y_pred_sel != cid) & (y_true_sel == cid))
        fp = np.sum((y_pred_sel == cid) & (y_true_sel != cid))
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f = float(2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0
        sel_recs[cname] = r
        sel_precs[cname] = p
        sel_f1s[cname] = f

    sel_bal_acc = float(np.mean(list(sel_recs.values())))
    sel_macro_f1 = float(np.mean(list(sel_f1s.values())))

    print(f"Non-Abstained Support   : n={n_non_abs}")
    print(f"Selective Accuracy      : {sel_acc*100:.2f}% ({int(np.sum(y_true_sel == y_pred_sel))}/{n_non_abs})")
    print(f"Selective Balanced Acc  : {sel_bal_acc*100:.2f}%")
    print(f"Selective Macro-F1      : {sel_macro_f1*100:.2f}%")

    # --- Part C: Deployment-View Metrics (Abstained -> Uncertain/Other) ---
    print("\n--- Part C: Deployment-View Metrics (Abstained Counted as 'Uncertain/Other') ---")
    # For deployment view, an abstained case is assigned label 9 ("Uncertain/Other").
    # It is a False Negative for the true tumor class.
    UNCERTAIN_CLASS = 9
    deploy_preds = argmax_preds.copy()
    deploy_preds[abstained_mask] = UNCERTAIN_CLASS

    # 3x4 Confusion matrix: Rows = True (Glioma, Meningioma, Metastasis), Cols = Pred (Glioma, Meningioma, Metastasis, Uncertain/Other)
    cm_deploy = np.zeros((3, 4), dtype=int)
    col_labels = ["Glioma", "Meningioma", "Metastasis", "Uncertain/Other"]
    col_map = {1: 0, 2: 1, 3: 2, UNCERTAIN_CLASS: 3}

    for true_c, pred_c in zip(all_targets, deploy_preds):
        row_idx = present_classes.index(true_c)
        col_idx = col_map[pred_c]
        cm_deploy[row_idx, col_idx] += 1

    print("Confusion Matrix (Rows = True Class, Columns = Predicted Class):")
    print(f"{'':<14} | {'Glioma':<10} | {'Meningioma':<10} | {'Metastasis':<10} | {'Uncertain/Other':<15}")
    print("-" * 72)
    for i, cname in enumerate(["Glioma", "Meningioma", "Metastasis"]):
        print(f"{cname:<14} | {cm_deploy[i, 0]:<10} | {cm_deploy[i, 1]:<10} | {cm_deploy[i, 2]:<10} | {cm_deploy[i, 3]:<15}")

    # Compute point estimates for deployment view over the 3 true classes
    dep_recs, dep_precs, dep_f1s = {}, {}, {}
    for i, cid in enumerate(present_classes):
        cname = class_map[cid]
        tp = np.sum((deploy_preds == cid) & (all_targets == cid))
        fn = np.sum((deploy_preds != cid) & (all_targets == cid))  # Includes cases routed to Uncertain/Other!
        fp = np.sum((deploy_preds == cid) & (all_targets != cid))
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f = float(2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0
        dep_recs[cname] = r
        dep_precs[cname] = p
        dep_f1s[cname] = f

    dep_bal_acc = float(np.mean(list(dep_recs.values())))
    dep_macro_f1 = float(np.mean(list(dep_f1s.values())))

    # Bootstrap 95% Confidence Intervals (B=num_bootstraps resamples of patients)
    rng = np.random.default_rng(42)
    boot_dep_bal_acc = []
    boot_dep_macro_f1 = []
    boot_dep_recs = {cname: [] for cname in class_map.values()}
    boot_dep_precs = {cname: [] for cname in class_map.values()}

    for _ in range(num_bootstraps):
        b_idx = rng.choice(n_test, size=n_test, replace=True)
        b_targets = all_targets[b_idx]
        b_deploy_preds = deploy_preds[b_idx]

        b_r_list, b_f_list = [], []
        for cid in present_classes:
            cname = class_map[cid]
            tp = np.sum((b_deploy_preds == cid) & (b_targets == cid))
            fn = np.sum((b_deploy_preds != cid) & (b_targets == cid))
            fp = np.sum((b_deploy_preds == cid) & (b_targets != cid))
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            f = float(2 * p * r / (p + r + 1e-8)) if (p + r) > 0 else 0.0

            boot_dep_recs[cname].append(r)
            boot_dep_precs[cname].append(p)
            b_r_list.append(r)
            b_f_list.append(f)

        boot_dep_bal_acc.append(np.mean(b_r_list))
        boot_dep_macro_f1.append(np.mean(b_f_list))

    dep_bal_acc_ci = [float(np.percentile(boot_dep_bal_acc, 2.5)), float(np.percentile(boot_dep_bal_acc, 97.5))]
    dep_macro_f1_ci = [float(np.percentile(boot_dep_macro_f1, 2.5)), float(np.percentile(boot_dep_macro_f1, 97.5))]

    print(f"\nDeployment Balanced Accuracy : {dep_bal_acc*100:.2f}% (95% CI: [{dep_bal_acc_ci[0]*100:.2f}%, {dep_bal_acc_ci[1]*100:.2f}%])")
    print(f"Deployment Macro-F1 Score    : {dep_macro_f1*100:.2f}% (95% CI: [{dep_macro_f1_ci[0]*100:.2f}%, {dep_macro_f1_ci[1]*100:.2f}%])")

    print("\nPer-Class Deployment Performance (Sensitivity with Abstention Penalized):")
    per_class_deploy = {}
    for cid in present_classes:
        cname = class_map[cid]
        r_ci = [float(np.percentile(boot_dep_recs[cname], 2.5)), float(np.percentile(boot_dep_recs[cname], 97.5))]
        p_ci = [float(np.percentile(boot_dep_precs[cname], 2.5)), float(np.percentile(boot_dep_precs[cname], 97.5))]
        print(f"  - {cname:<12}: Recall = {dep_recs[cname]*100:.2f}% (95% CI: [{r_ci[0]*100:.2f}%, {r_ci[1]*100:.2f}%]), Precision = {dep_precs[cname]*100:.2f}% (95% CI: [{p_ci[0]*100:.2f}%, {p_ci[1]*100:.2f}%]), F1 = {dep_f1s[cname]*100:.2f}%")
        per_class_deploy[cname] = {
            "recall": dep_recs[cname],
            "recall_95ci": r_ci,
            "precision": dep_precs[cname],
            "precision_95ci": p_ci,
            "f1": dep_f1s[cname]
        }

    # --- Part D: Side-by-Side Comparison ---
    print("\n--- Part D: Side-by-Side Comparison (Argmax-Only vs Deployment-View) ---")
    # Existing argmax metrics from test_evaluation_results.json
    existing_res_path = BASE_DIR / "test_evaluation_results.json"
    with open(existing_res_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    argmax_bal_acc = existing_data["balanced_accuracy"]
    argmax_bal_acc_ci = existing_data["balanced_accuracy_95ci"]
    argmax_macro_f1 = existing_data["macro_f1"]
    argmax_macro_f1_ci = existing_data["macro_f1_95ci"]

    print(f"{'Metric':<28} | {'Argmax-Only (Existing)':<26} | {'Deployment-View (With Abstention)':<32}")
    print("-" * 92)
    print(f"{'Balanced Accuracy':<28} | {argmax_bal_acc*100:.2f}% [{argmax_bal_acc_ci[0]*100:.2f}%, {argmax_bal_acc_ci[1]*100:.2f}%] | {dep_bal_acc*100:.2f}% [{dep_bal_acc_ci[0]*100:.2f}%, {dep_bal_acc_ci[1]*100:.2f}%]")
    print(f"{'Macro-F1 Score':<28} | {argmax_macro_f1*100:.2f}% [{argmax_macro_f1_ci[0]*100:.2f}%, {argmax_macro_f1_ci[1]*100:.2f}%] | {dep_macro_f1*100:.2f}% [{dep_macro_f1_ci[0]*100:.2f}%, {dep_macro_f1_ci[1]*100:.2f}%]")
    print(f"{'Glioma Recall':<28} | {existing_data['per_class']['Glioma']['recall']*100:.2f}%                   | {dep_recs['Glioma']*100:.2f}% [{boot_dep_recs['Glioma'][0]*100:.2f}%, {boot_dep_recs['Glioma'][-1]*100:.2f}%]")
    print(f"{'Meningioma Recall':<28} | {existing_data['per_class']['Meningioma']['recall']*100:.2f}%                   | {dep_recs['Meningioma']*100:.2f}% [{boot_dep_recs['Meningioma'][0]*100:.2f}%, {boot_dep_recs['Meningioma'][-1]*100:.2f}%]")
    print(f"{'Metastasis Recall':<28} | {existing_data['per_class']['Metastasis']['recall']*100:.2f}%                   | {dep_recs['Metastasis']*100:.2f}% [{boot_dep_recs['Metastasis'][0]*100:.2f}%, {boot_dep_recs['Metastasis'][-1]*100:.2f}%]")

    # Save to abstention_evaluation_results.json
    out_results = {
        "test_support_n": n_test,
        "temperature_used": T,
        "abstain_threshold": conf_thresh,
        "abstention_summary": {
            "total_abstained_n": n_abstained,
            "abstention_rate": float(n_abstained / n_test),
            "coverage": coverage,
            "per_class": per_class_abstention
        },
        "selective_performance_non_abstained": {
            "support_n": n_non_abs,
            "accuracy": sel_acc,
            "balanced_accuracy": sel_bal_acc,
            "macro_f1": sel_macro_f1,
            "per_class_recall": sel_recs,
            "per_class_precision": sel_precs,
            "per_class_f1": sel_f1s
        },
        "deployment_view_with_abstention": {
            "balanced_accuracy": dep_bal_acc,
            "balanced_accuracy_95ci": dep_bal_acc_ci,
            "macro_f1": dep_macro_f1,
            "macro_f1_95ci": dep_macro_f1_ci,
            "confusion_matrix_3x4": {
                "rows_true": ["Glioma", "Meningioma", "Metastasis"],
                "cols_pred": col_labels,
                "matrix": cm_deploy.tolist()
            },
            "per_class": per_class_deploy
        },
        "side_by_side_comparison": {
            "argmax_balanced_accuracy": argmax_bal_acc,
            "argmax_balanced_accuracy_95ci": argmax_bal_acc_ci,
            "deployment_balanced_accuracy": dep_bal_acc,
            "deployment_balanced_accuracy_95ci": dep_bal_acc_ci,
            "argmax_macro_f1": argmax_macro_f1,
            "argmax_macro_f1_95ci": argmax_macro_f1_ci,
            "deployment_macro_f1": dep_macro_f1,
            "deployment_macro_f1_95ci": dep_macro_f1_ci
        }
    }

    out_file = BASE_DIR / "abstention_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)

    print(f"\n[OK] Results saved to {out_file}")


if __name__ == "__main__":
    run_evaluation(num_bootstraps=1000)
