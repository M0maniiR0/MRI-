#!/usr/bin/env python3
"""
run_source_recognition_check.py
===============================
Source Recognition Check on Cached Preprocessed Volumes.
Evaluates whether acquisition source dataset (MSD Task01, BraTS-MEN, Brain-Mets)
can be identified using ONLY intensity-histogram and noise-level features (zero geometry).

Rules:
- ONLY intensity-histogram and noise features.
- NO geometry: no shape, no dimensions, no spacing, no voxel counts, no volumes, no padding fraction.
- Normalized probability densities so volume differences cannot leak through histogram counts.
- Strict patient-level split (Train: 619, Test: 132).
- Report Balanced Accuracy with empirical bootstrap 95% CI (B=1000).
- Report only factual numbers and what they imply.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
CACHE_DIR = BASE_DIR / "cache_preprocessed"


def extract_intensity_and_noise_features(vol_path: str) -> np.ndarray:
    """
    Extracts purely intensity distribution and noise texture features from a 3D volume.
    Zero geometry: no shape, no voxel counts, no spatial coordinates.
    All histogram features are normalized probability distributions.
    """
    vol = np.load(vol_path).astype(np.float32)  # Shape (128, 128, 128)
    
    # Brain mask: non-zero voxels
    brain_mask = (vol != 0.0)
    nz_vals = vol[brain_mask]
    
    if len(nz_vals) == 0:
        return np.zeros(40, dtype=np.float32)

    # 1. Intensity distribution summary statistics (Scale-free / normalized moments)
    mean_val = float(np.mean(nz_vals))
    std_val = float(np.std(nz_vals))
    skew_val = float(skew(nz_vals))
    kurt_val = float(kurtosis(nz_vals))

    # 2. Intensity Percentiles (13 features)
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    perc_vals = [float(p) for p in np.percentile(nz_vals, percentiles)]
    iqr = perc_vals[-3] - perc_vals[2]  # p90 - p10

    # 3. Normalized Histogram Density (15 fixed bins between -3.0 and +3.0)
    # density=True ensures the histogram integrates to 1 and does NOT reflect voxel count
    hist_counts, _ = np.histogram(nz_vals, bins=15, range=(-3.0, 3.0), density=True)
    # Shannon entropy of the intensity distribution
    bin_probs = hist_counts / (np.sum(hist_counts) + 1e-12)
    entropy_val = float(-np.sum(bin_probs * np.log(bin_probs + 1e-12)))

    # 4. Noise-Level Features (High-frequency variations inside brain)
    # A. 3D Finite difference gradient magnitude where both voxels are within brain
    diff_x = np.abs(vol[1:, :, :] - vol[:-1, :, :])
    mask_x = brain_mask[1:, :, :] & brain_mask[:-1, :, :]
    noise_dx = diff_x[mask_x]
    
    diff_y = np.abs(vol[:, 1:, :] - vol[:, :-1, :])
    mask_y = brain_mask[:, 1:, :] & brain_mask[:, :-1, :]
    noise_dy = diff_y[mask_y]

    diff_z = np.abs(vol[:, :, 1:] - vol[:, :, :-1])
    mask_z = brain_mask[:, :, 1:] & brain_mask[:, :, :-1]
    noise_dz = diff_z[mask_z]

    all_diffs = np.concatenate([noise_dx, noise_dy, noise_dz])
    noise_mean_grad = float(np.mean(all_diffs)) if len(all_diffs) > 0 else 0.0
    noise_median_grad = float(np.median(all_diffs)) if len(all_diffs) > 0 else 0.0
    noise_mad = float(np.median(np.abs(all_diffs - noise_median_grad))) if len(all_diffs) > 0 else 0.0
    noise_p90_grad = float(np.percentile(all_diffs, 90)) if len(all_diffs) > 0 else 0.0

    # B. Local standard deviation (texture patch noise)
    # Compute local variance in 3x3x3 neighborhood: E[X^2] - (E[X])^2
    mean_local = uniform_filter(vol, size=3)
    mean_sq_local = uniform_filter(vol ** 2, size=3)
    var_local = np.maximum(mean_sq_local - mean_local ** 2, 0.0)
    std_local = np.sqrt(var_local)[brain_mask]
    local_noise_mean = float(np.mean(std_local))
    local_noise_std = float(np.std(std_local))

    # Feature vector: 4 moments + 13 percentiles + 1 IQR + 15 histogram densities + 1 entropy + 6 noise features = 40 features
    feat_vector = np.array([
        mean_val, std_val, skew_val, kurt_val,
        *perc_vals, iqr,
        *hist_counts.tolist(), entropy_val,
        noise_mean_grad, noise_median_grad, noise_mad, noise_p90_grad,
        local_noise_mean, local_noise_std
    ], dtype=np.float32)

    return feat_vector


def run_source_check():
    splits_file = CACHE_DIR / "splits.json"
    assert splits_file.exists(), f"Missing {splits_file}"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_recs = splits["train"]
    test_recs = splits["test"]

    # Target class: dataset source (1: MSD Task01, 2: BraTS-MEN, 3: Brain-Mets)
    source_names = {1: "MSD Task01 (Glioma)", 2: "BraTS-MEN (Meningioma)", 3: "Brain-Mets (Metastasis)"}

    print("================ SOURCE-RECOGNITION EXPERIMENT (INTENSITY & NOISE ONLY) ================")
    print("Features extracted: 40 features")
    print("  - Intensity Statistics: Mean, Std, Skewness, Kurtosis (4)")
    print("  - Intensity Percentiles: p1, p5, p10, p20, p30, p40, p50, p60, p70, p80, p90, p95, p99, IQR (14)")
    print("  - Intensity Histogram Densities: 15 fixed bins [-3.0, 3.0], normalized sum=1.0 (15)")
    print("  - Distribution Entropy: Shannon entropy of normalized intensity histogram (1)")
    print("  - High-Frequency Noise Texture: Mean grad, Median grad, MAD, p90 grad, Local 3x3x3 mean std, Local 3x3x3 std std (6)")
    print("  - Geometric Features: ZERO (No shape, no spacing, no counts, no volume, no padding fraction)")
    print(f"\nCohort Sizes (Patient-Level Split):")
    print(f"  - Train Set : n={len(train_recs)} subjects (MSD={sum(1 for r in train_recs if r['class_id']==1)}, BraTS-MEN={sum(1 for r in train_recs if r['class_id']==2)}, Brain-Mets={sum(1 for r in train_recs if r['class_id']==3)})")
    print(f"  - Test Set  : n={len(test_recs)} subjects (MSD={sum(1 for r in test_recs if r['class_id']==1)}, BraTS-MEN={sum(1 for r in test_recs if r['class_id']==2)}, Brain-Mets={sum(1 for r in test_recs if r['class_id']==3)})")

    t0 = time.time()
    print("\nExtracting features for Train cohort (6 workers)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        X_train_list = list(ex.map(lambda r: extract_intensity_and_noise_features(r["cache_file"]), train_recs))
    y_train = np.array([r["class_id"] for r in train_recs], dtype=np.int64)
    X_train = np.array(X_train_list, dtype=np.float32)

    print(f"Extracting features for Test cohort (6 workers)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        X_test_list = list(ex.map(lambda r: extract_intensity_and_noise_features(r["cache_file"]), test_recs))
    y_test = np.array([r["class_id"] for r in test_recs], dtype=np.int64)
    X_test = np.array(X_test_list, dtype=np.float32)
    print(f"Feature extraction completed in {time.time() - t0:.2f}s.")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Regularized Multinomial Logistic Regression
    clf_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_lr.fit(X_train_scaled, y_train)
    y_pred_lr = clf_lr.predict(X_test_scaled)
    bal_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)

    # Model 2: Random Forest Classifier
    clf_rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    bal_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)

    # Bootstrap 95% Confidence Intervals (B=1000 resamples on test set)
    rng = np.random.default_rng(42)
    boot_accs_lr, boot_accs_rf = [], []
    n_test = len(y_test)

    for _ in range(1000):
        b_idx = rng.choice(n_test, size=n_test, replace=True)
        b_y_true = y_test[b_idx]
        b_pred_lr = y_pred_lr[b_idx]
        b_pred_rf = y_pred_rf[b_idx]
        boot_accs_lr.append(balanced_accuracy_score(b_y_true, b_pred_lr))
        boot_accs_rf.append(balanced_accuracy_score(b_y_true, b_pred_rf))

    ci_lr = [float(np.percentile(boot_accs_lr, 2.5)), float(np.percentile(boot_accs_lr, 97.5))]
    ci_rf = [float(np.percentile(boot_accs_rf, 2.5)), float(np.percentile(boot_accs_rf, 97.5))]

    print("\n---------------- SOURCE RECOGNITION TEST RESULTS ----------------")
    print(f"Theoretical Chance Baseline : 33.33% (1/3)")
    print(f"Logistic Regression Balanced Accuracy : {bal_acc_lr * 100:.2f}% (Bootstrap 95% CI: [{ci_lr[0] * 100:.2f}%, {ci_lr[1] * 100:.2f}%])")
    print(f"Random Forest Balanced Accuracy       : {bal_acc_rf * 100:.2f}% (Bootstrap 95% CI: [{ci_rf[0] * 100:.2f}%, {ci_rf[1] * 100:.2f}%])")

    # Confusion Matrix
    print("\nConfusion Matrix (Logistic Regression, Rows: Ground Truth Source, Cols: Predicted Source):")
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[1, 2, 3])
    print("                              Pred MSD      Pred BraTS-MEN     Pred Brain-Mets")
    for i, cid in enumerate([1, 2, 3]):
        print(f"  Actual {source_names[cid]:<24}: {cm_lr[i, 0]:<14} {cm_lr[i, 1]:<18} {cm_lr[i, 2]}")

    print("\nConfusion Matrix (Random Forest, Rows: Ground Truth Source, Cols: Predicted Source):")
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[1, 2, 3])
    print("                              Pred MSD      Pred BraTS-MEN     Pred Brain-Mets")
    for i, cid in enumerate([1, 2, 3]):
        print(f"  Actual {source_names[cid]:<24}: {cm_rf[i, 0]:<14} {cm_rf[i, 1]:<18} {cm_rf[i, 2]}")

    # Top Features
    feat_names = [
        "mean", "std", "skewness", "kurtosis",
        "p1", "p5", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "p95", "p99", "iqr",
        *[f"hist_bin_{i}" for i in range(15)], "entropy",
        "noise_mean_grad", "noise_median_grad", "noise_mad", "noise_p90_grad",
        "local_noise_mean", "local_noise_std"
    ]
    importances = clf_rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\nTop 10 Most Discriminative Intensity/Noise Features (Random Forest):")
    for idx in top_indices:
        print(f"  - {feat_names[idx]:<20}: relative importance = {importances[idx]:.4f}")

    # Per-class breakdown
    print("\nPer-Source Recall Breakdown (Logistic Regression):")
    for i, cid in enumerate([1, 2, 3]):
        rec = float(cm_lr[i, i] / np.sum(cm_lr[i, :])) if np.sum(cm_lr[i, :]) > 0 else 0.0
        print(f"  - {source_names[cid]:<24}: Recall = {rec * 100:.2f}% ({cm_lr[i, i]}/{np.sum(cm_lr[i, :])})")

    # Save results
    out_results = {
        "features_used": "intensity_histogram_and_noise_only",
        "geometry_included": False,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "chance_level": 1.0 / 3.0,
        "logistic_regression": {
            "balanced_accuracy": float(bal_acc_lr),
            "ci_95": ci_lr,
            "confusion_matrix": cm_lr.tolist()
        },
        "random_forest": {
            "balanced_accuracy": float(bal_acc_rf),
            "ci_95": ci_rf,
            "confusion_matrix": cm_rf.tolist(),
            "top_features": [(feat_names[i], float(importances[i])) for i in top_indices]
        }
    }
    out_json = BASE_DIR / "source_recognition_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    print(f"\n[OK] Results saved to {out_json}")


if __name__ == "__main__":
    run_source_check()
