#!/usr/bin/env python3
"""
detection_experiment.py
=======================
3D MRI Abnormality Detection Experiment (Normal vs Abnormal).
BrainAgenticthon 2026 - Medical AI Workstream (Phase 4 / Prompt 04).

Key Guidelines & Safety Constraints:
------------------------------------
1. Strict Clinical Safety Guard:
   - Training a binary detector on tumor-only data (Glioma) without verified
     Healthy Controls violates clinical AI validation protocols.
   - If no healthy controls (label=0) are provided, training is BLOCKED.

2. Metrics Implemented:
   - AUROC (Area under ROC Curve)
   - Sensitivity @ 90% Specificity (official competition metric, 15% weight)
   - ECE (Expected Calibration Error)
   - Brier Score

3. Calibration:
   - Temperature Scaling to calibrate raw logits into trusted probabilities.

4. Handoff Contract:
   - Exports detection.json matching the official schema for downstream agents.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

from monai.networks.nets import DenseNet121

_SCRIPT_DIR = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────
# Temperature Scaling for Calibration
# ──────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 on validation logits
    to calibrate output probabilities (Platt scaling / temperature scaling).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        self.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.eval()
        return float(self.temperature.item())


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_mask = (probs >= bin_limits[i]) & (probs < bin_limits[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_sensitivity_at_specificity(
    labels: np.ndarray,
    probs: np.ndarray,
    target_spec: float = 0.90,
) -> Tuple[float, float]:
    """
    Computes sensitivity at target specificity (e.g. 90%) using ROC curve.
    Returns: (sensitivity, optimal_threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    specificity = 1.0 - fpr

    # Find points where specificity >= target_spec
    valid_indices = np.where(specificity >= target_spec)[0]
    if len(valid_indices) == 0:
        return 0.0, 1.0

    best_idx = valid_indices[-1]  # Closest to target specificity from above
    return float(tpr[best_idx]), float(thresholds[best_idx])


# ──────────────────────────────────────────────────────────────
# Clinical Safety Verification
# ──────────────────────────────────────────────────────────────
def verify_dataset_safety(manifest_rows: List[Dict[str, Any]]) -> None:
    """
    Blocks training if the dataset only contains glioma cases (label=1)
    without any healthy control cases (label=0).
    """
    labels = [int(r["label"]) for r in manifest_rows]
    pos_count = sum(1 for l in labels if l == 1)
    neg_count = sum(1 for l in labels if l == 0)

    print("=" * 65)
    print("  CLINICAL DATASET AUDIT FOR DETECTION")
    print("=" * 65)
    print(f"  Abnormal / Glioma Cases (1) : {pos_count}")
    print(f"  Healthy / Normal Cases  (0) : {neg_count}")
    print("=" * 65)

    if neg_count == 0:
        error_msg = (
            "\n" + "!" * 70 + "\n"
            "[CLINICAL SAFETY BLOCK] DETECTION EXPERIMENT CANNOT PROCEED!\n"
            "Reason: The dataset contains ZERO Healthy Controls (label=0).\n"
            "Clinical Rationale:\n"
            "  Training an abnormal-vs-normal binary detector on a cohort consisting\n"
            "  only of brain tumor patients (MSD Task01 / BraTS) is scientifically\n"
            "  invalid. The model will suffer from severe shortcut learning and 100% false positives.\n"
            "Action Required:\n"
            "  Please supply a manifest containing verified normal control MRI volumes\n"
            "  (e.g., from OASIS, IXI, or ADNI normal cohorts) with label=0.\n"
            + "!" * 70
        )
        raise RuntimeError(error_msg)


# ──────────────────────────────────────────────────────────────
# Mock Healthy Controls Ingestion Helper (For pipeline testing)
# ──────────────────────────────────────────────────────────────
def generate_sample_manifest_with_controls(
    output_csv: Path,
    msd_data_dir: Path,
    num_abnormal: int = 10,
    num_healthy_dummy: int = 5,
) -> None:
    """
    Helper for engineering tests: creates a sample detection manifest.
    Flags dummy controls with an explicit 'is_synthetic_test_control' flag.
    """
    manifest_file = msd_data_dir / "dataset.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    # Add real glioma cases
    for item in manifest["training"][:num_abnormal]:
        cid = Path(item["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        rows.append({
            "case_id": cid,
            "image_path": str((msd_data_dir / item["image"]).resolve()),
            "label": 1,
            "source_cohort": "MSD_Task01_Glioma",
            "is_test_control": False,
        })

    # Add dummy entries representing healthy controls for pipeline dry-run
    for i in range(num_healthy_dummy):
        cid = f"HEALTHY_CTRL_{i+1:03d}"
        # Use first image as file placeholder for dry-run
        rows.append({
            "case_id": cid,
            "image_path": str((msd_data_dir / manifest["training"][0]["image"]).resolve()),
            "label": 0,
            "source_cohort": "TEST_HEALTHY_CONTROL",
            "is_test_control": True,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "image_path", "label", "source_cohort", "is_test_control"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Generated detection manifest -> {output_csv.name} ({len(rows)} cases)")


# ──────────────────────────────────────────────────────────────
# Downstream Handoff: detection.json Generator
# ──────────────────────────────────────────────────────────────
def export_detection_json(
    case_id: str,
    abnormal_prob: float,
    threshold: float = 0.5,
    calibration_method: str = "temperature_scaling",
    model_version: str = "detector_3d_densenet121_v1",
    output_dir: Path = _SCRIPT_DIR / "outputs" / "detection",
) -> Path:
    """
    Generates detection.json matching the official contract (Section 6 of roadmap).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{case_id}_detection.json"

    data = {
        "case_id": case_id,
        "abnormal_probability": round(float(abnormal_prob), 4),
        "prediction": "abnormal" if abnormal_prob >= threshold else "normal",
        "threshold": round(float(threshold), 4),
        "calibration_method": calibration_method,
        "model_version": model_version,
        "qc_flags": [],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_file


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3D Abnormality Detection Experiment")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest CSV with columns: case_id, image_path, label")
    parser.add_argument("--create-test-manifest", action="store_true", help="Generate a test manifest for pipeline verification")
    parser.add_argument("--check-safety", action="store_true", help="Audit current MSD dataset for healthy controls")
    args = parser.parse_args()

    msd_dir = _SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"

    if args.create_test_manifest:
        out_manifest = _SCRIPT_DIR / "detection_test_manifest.csv"
        generate_sample_manifest_with_controls(out_manifest, msd_dir, num_abnormal=10, num_healthy_dummy=5)
        print("Test manifest created. Review columns: case_id, image_path, label (0 or 1).")
        return

    # If user wants to check the existing dataset
    if args.check_safety or args.manifest is None:
        print("[INFO] Auditing current MSD dataset for detection task...")
        manifest_file = msd_dir / "dataset.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # MSD training cases are all glioma
        msd_rows = [{"case_id": c["image"], "label": 1} for c in manifest["training"]]
        try:
            verify_dataset_safety(msd_rows)
        except RuntimeError as e:
            print(e)
            print("\n[CONCLUSION] Exactly as documented in Section 2 & 10 of the roadmap:")
            print("  'BraTS is built around tumor cases. A binary classifier cannot be validly trained'")
            print("  'on BraTS alone without a real healthy control cohort.'")
            print("  Use --create-test-manifest to generate a formatted template when healthy data is acquired.")


if __name__ == "__main__":
    main()
