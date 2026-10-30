#!/usr/bin/env python3
"""
tests/test_task2_preprocessing.py
=================================
Acceptance Test Suite for Task 2 Preprocessing:
1. Tests 1 scan from each available source cohort:
   - Glioma (MSD Task01 BraTS)
   - Meningioma (BraTS-MEN 2023)
   - Brain Metastasis (Pakawat / SCI Cohort)
   - Alzheimer's / Hippocampus (MSD Task04)
2. Produces before/after PNG visualizations of middle slices in 3 planes.
3. Asserts target shape [128, 128, 128], spacing [1.0, 1.0, 1.0], finite values, and value ranges.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from preprocess import preprocess_mri, save_orthogonal_slices_png

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
REPORT_VIZ_DIR = BASE_DIR / "reports" / "qc_preproc_visualizations"

SAMPLE_SCANS = {
    "glioma": {
        "path": BASE_DIR / "Task01_BrainTumour" / "Task01_BrainTumour" / "imagesTr" / "BRATS_001.nii.gz",
        "channel_idx": 1  # T1 channel in 4D BraTS
    },
    "meningioma": {
        "path": BASE_DIR / "Task_Meningioma" / "BraTS_MEN" / "BraTS-MEN-00004-000" / "BraTS-MEN-00004-000-t1c.nii.gz",
        "channel_idx": 0
    },
    "metastasis": {
        "path": BASE_DIR / "Task_BrainMetastasis" / "example_input" / "case-a" / "t1c.nii.gz",
        "channel_idx": 0
    },
    "hippocampus": {
        "path": BASE_DIR / "Task04_Hippocampus" / "imagesTr" / "hippocampus_001.nii.gz",
        "channel_idx": 0
    }
}

def test_unified_preprocessing():
    REPORT_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print("================ RUNNING TASK 2 ACCEPTANCE SUITE ================")

    results = {}
    for cohort_name, info in SAMPLE_SCANS.items():
        scan_path = info["path"]
        ch_idx = info["channel_idx"]
        print(f"\n--- Testing cohort: {cohort_name.upper()} ---")
        print(f"File: {scan_path.name}")
        assert scan_path.exists(), f"Sample file not found: {scan_path}"

        # Run unified preprocessing
        res = preprocess_mri(scan_path, channel_index=ch_idx)
        results[cohort_name] = res

        proc_vol = res["processed_volume"]
        raw_vol = res["raw_volume_before"]
        qc = res["qc_report"]

        # 1. Assert Shape
        assert proc_vol.shape == (128, 128, 128), f"Expected shape (128, 128, 128), got {proc_vol.shape}"
        print(f"  [PASS] Shape verified: {proc_vol.shape}")

        # 2. Assert Spacing & Orientation
        assert res["spacing"] == [1.0, 1.0, 1.0], f"Expected 1.0 mm isotropic spacing, got {res['spacing']}"
        assert res["orientation"] == "RAS", f"Expected RAS orientation, got {res['orientation']}"
        print(f"  [PASS] Spacing & Orientation verified: {res['spacing']} mm isotropic, {res['orientation']}")

        # 3. Assert Quality Control & Finite Values
        assert np.isfinite(proc_vol).all(), "Processed volume contains NaN or Inf"
        assert qc["passed"] is True or qc["brain_voxels"] > 500, f"QC failed unexpectedly: {qc['issues']}"
        print(f"  [PASS] QC status: brain voxels={qc['brain_voxels']}, issues={qc['issues']}")

        # 4. Assert Value Range & Brain Z-score
        brain_mask = res["brain_mask"]
        if np.sum(brain_mask) > 0:
            brain_vals = proc_vol[brain_mask]
            mean_b = np.mean(brain_vals)
            std_b = np.std(brain_vals)
            print(f"  [PASS] Intensity Range: min={proc_vol.min():.2f}, max={proc_vol.max():.2f}, brain_mean={mean_b:.2f}, brain_std={std_b:.2f}")
            assert -15.0 <= proc_vol.min() and proc_vol.max() <= 15.0, "Voxel values out of expected bounds"
            assert abs(mean_b) < 1.0, f"Brain mean intensity {mean_b} should be near 0 after Z-score"
            assert 0.5 < std_b < 1.5, f"Brain std intensity {std_b} should be near 1 after Z-score"

        # 5. Assert Invertibility metadata
        inv_meta = res["inverse_transform_info"]
        assert "original_shape" in inv_meta and "pads" in inv_meta and "crops" in inv_meta
        print(f"  [PASS] Invertibility metadata preserved: original_shape={inv_meta['original_shape']}")

        # 6. Generate and save orthogonal PNG visualisations (before & after)
        before_png = REPORT_VIZ_DIR / f"{cohort_name}_before.png"
        after_png = REPORT_VIZ_DIR / f"{cohort_name}_after.png"
        save_orthogonal_slices_png(raw_vol, before_png, title=f"{cohort_name} Before")
        save_orthogonal_slices_png(proc_vol, after_png, title=f"{cohort_name} After")
        assert before_png.exists() and after_png.exists()
        print(f"  [PASS] Visualizations generated:\n         {before_png}\n         {after_png}")

    print("\n[ALL ACCEPTANCE CRITERIA FOR TASK 2 FULLY SATISFIED]")

if __name__ == "__main__":
    test_unified_preprocessing()
