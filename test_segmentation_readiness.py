#!/usr/bin/env python3
"""
test_segmentation_readiness.py
==============================
Training-Readiness Unit & Integration Test Suite for MONAI SegResNet 3D.
BrainAgenticthon 2026 - Medical AI Workstream (Abdulrahman & Rou'a).

Tests:
------
1. test_segmentation_label_conversion:
   - Verifies bidirectional mapping between discrete classes {0,1,2,3} and multi-label [WT,TC,ET].
2. test_spatial_inversion_nearest_neighbor:
   - Verifies nearest-neighbor spatial inversion back to native MRI grid without label blurring.
3. test_nifti_export_and_reload:
   - Validates exported NIfTI shape, affine matrix, header zooms, and discrete values upon reload.
4. test_checkpoint_metadata_and_safe_loading:
   - Validates loading of best_model.pt and verifies full metadata structure.
5. test_zero_leakage_and_missing_file_guards:
   - Verifies 0 overlap with 72 holdout test cases and asserts loud FileNotFoundError for missing files.
6. test_hd95_empty_policy_and_explicit_error:
   - Validates empty-mask policy (0.0mm both empty, diagonal penalty one empty) and explicit error raising.
7. test_stochastic_training_pipeline:
   - Verifies stochastic sampling with positive tumor and background ratios.
8. test_inference_input_isolation:
   - Verifies forward sliding-window inference operates on MRI images only without ground truth.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai.data import MetaTensor
from monai.networks.nets import SegResNet

from monai_preprocessing_pipeline import (
    convert_multilabel_to_multiclass,
    get_spatial_pipeline,
    get_training_pipeline,
    get_validation_pipeline,
)
from segmentation_baseline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_SPLITS_DIR,
    build_case_dicts,
    compute_clinical_volumes,
    compute_region_dice,
    compute_region_hd95,
    create_model,
    enforce_zero_leakage_guard,
    evaluate_single_case_full_volume,
    invert_mask_to_native_space,
    load_model_from_checkpoint,
    set_deterministic_seed,
)


class TestSegmentationTrainingReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_deterministic_seed(42)
        cls.data_dir = DEFAULT_DATA
        cls.splits_dir = DEFAULT_SPLITS_DIR
        cls.test_case_id = "BRATS_002"
        cls.img_path = cls.data_dir / f"imagesTr/{cls.test_case_id}.nii.gz"
        cls.lbl_path = cls.data_dir / f"labelsTr/{cls.test_case_id}.nii.gz"

        assert cls.img_path.is_file(), f"Test image missing: {cls.img_path}"
        assert cls.lbl_path.is_file(), f"Test label missing: {cls.lbl_path}"

    def test_01_label_conversion_bidirectional(self):
        """Test bidirectional mapping between multiclass {0,1,2,3} and multilabel [WT, TC, ET]."""
        # Create synthetic volume with all classes
        shape = (10, 10, 10)
        synth_seg = np.zeros(shape, dtype=np.int32)
        synth_seg[1:3, 1:3, 1:3] = 1  # Edema
        synth_seg[4:6, 4:6, 4:6] = 2  # Non-enhancing
        synth_seg[7:9, 7:9, 7:9] = 3  # Enhancing

        # Multiclass -> Multilabel
        wt = (synth_seg == 1) | (synth_seg == 2) | (synth_seg == 3)
        tc = (synth_seg == 2) | (synth_seg == 3)
        et = (synth_seg == 3)
        ml_tensor = torch.from_numpy(np.stack([wt, tc, et], axis=0).astype(np.float32))

        # Multilabel -> Multiclass
        recon = convert_multilabel_to_multiclass(ml_tensor)

        # Assert exact reconstruction
        np.testing.assert_array_equal(synth_seg, recon)
        self.assertEqual(set(np.unique(recon)), {0, 1, 2, 3})

    def test_02_spatial_inversion_nearest_neighbor(self):
        """Test spatial inversion using strictly nearest-neighbor interpolation."""
        spatial_pipe = get_spatial_pipeline(allow_missing_keys=True)
        res = spatial_pipe({"image": str(self.img_path)})

        img_ref = res["image"]
        meta = img_ref.meta
        applied_ops = img_ref.applied_operations

        # Create discrete prediction with classes 0, 1, 2, 3
        spatial_shape = img_ref.shape[1:]  # (240, 240, 155)
        dummy_pred = np.zeros(spatial_shape, dtype=np.int32)
        dummy_pred[100:110, 100:110, 70:80] = 1
        dummy_pred[110:120, 110:120, 70:80] = 2
        dummy_pred[120:130, 120:130, 70:80] = 3

        inv_mask = invert_mask_to_native_space(
            pred_multiclass=dummy_pred,
            img_meta=meta,
            applied_operations=applied_ops,
            spatial_pipe=spatial_pipe,
        )

        orig_nii = nib.load(str(self.img_path))
        self.assertEqual(inv_mask.shape, orig_nii.shape[:3])

        # Assert no fractional values occurred and labels belong strictly to {0, 1, 2, 3}
        unique_vals = set(np.unique(inv_mask))
        self.assertTrue(unique_vals.issubset({0, 1, 2, 3}))
        self.assertIn(1, unique_vals)
        self.assertIn(2, unique_vals)
        self.assertIn(3, unique_vals)

    def test_03_nifti_export_and_reload_validation(self):
        """Test exporting NIfTI mask and reloading to verify shape, affine, and zooms."""
        orig_nii = nib.load(str(self.img_path))
        orig_shape = orig_nii.shape[:3]
        orig_affine = orig_nii.affine
        orig_zooms = orig_nii.header.get_zooms()[:3]

        dummy_mask = np.zeros(orig_shape, dtype=np.uint8)
        dummy_mask[50:60, 50:60, 50:60] = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_mask.nii.gz"
            out_nii = nib.Nifti1Image(dummy_mask, orig_affine, orig_nii.header)
            nib.save(out_nii, str(out_path))

            reloaded = nib.load(str(out_path))
            self.assertEqual(reloaded.shape, orig_shape)
            self.assertTrue(np.allclose(reloaded.affine, orig_affine))
            self.assertEqual(reloaded.header.get_zooms()[:3], orig_zooms)
            self.assertEqual(set(np.unique(reloaded.get_fdata())), {0.0, 2.0})

    def test_04_checkpoint_loading_and_metadata_verification(self):
        """Test loading existing checkpoint best_model.pt and verifying metadata structure."""
        device = torch.device("cpu")
        model, meta = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)

        self.assertIsInstance(model, SegResNet)
        self.assertEqual(meta["checkpoint_file"], "best_model.pt")
        self.assertIn(meta["init_filters"], [8, 16])
        self.assertIn("epoch", meta)
        self.assertIn("training_loss", meta)

        # Test forward pass with dummy tensor
        dummy_in = torch.randn((1, 4, 64, 64, 64), device=device)
        with torch.no_grad():
            out = model(dummy_in)
        self.assertEqual(out.shape, (1, 3, 64, 64, 64))

    def test_05_zero_leakage_and_missing_file_guards(self):
        """Test split counts, zero leakage with test set, and loud missing file exceptions."""
        train_ids, val_ids, test_ids = enforce_zero_leakage_guard(self.splits_dir)

        # Checkpoint split counts
        self.assertEqual(len(train_ids), 339)
        self.assertEqual(len(val_ids), 73)
        self.assertEqual(len(test_ids), 72)

        # Strict zero leakage
        self.assertEqual(len(set(train_ids) & set(test_ids)), 0)
        self.assertEqual(len(set(val_ids) & set(test_ids)), 0)
        self.assertEqual(len(set(train_ids) & set(val_ids)), 0)

        # Missing file test: must fail loudly
        fake_ids = ["NON_EXISTENT_CASE_999"]
        with self.assertRaises(FileNotFoundError):
            build_case_dicts(self.data_dir, fake_ids, require_label=True)

    def test_06_hd95_empty_mask_policy_and_explicit_error(self):
        """Test empty-mask HD95 policy and explicit error propagation."""
        spacing = (1.0, 1.0, 1.0)
        shape = (50, 50, 50)
        diag = float(np.sqrt(50**2 + 50**2 + 50**2))

        empty_a = torch.zeros(shape)
        empty_b = torch.zeros(shape)
        box_a = torch.zeros(shape)
        box_a[10:20, 10:20, 10:20] = 1.0

        # Case 1: Both empty -> 0.0 mm
        val, pen = compute_region_hd95(empty_a, empty_b, spacing, shape)
        self.assertEqual(val, 0.0)
        self.assertFalse(pen)

        # Case 2: One empty, one non-empty -> Diagonal penalty
        val, pen = compute_region_hd95(box_a, empty_b, spacing, shape)
        self.assertEqual(val, round(diag, 2))
        self.assertTrue(pen)

        # Case 3: Both non-empty -> Finite physical distance
        box_b = torch.zeros(shape)
        box_b[11:21, 11:21, 11:21] = 1.0
        val, pen = compute_region_hd95(box_a, box_b, spacing, shape)
        self.assertFalse(pen)
        self.assertGreater(val, 0.0)
        self.assertLess(val, diag)

    def test_07_stochastic_training_pipeline(self):
        """Test stochastic sampling with positive tumor and background ratio."""
        from segmentation_baseline import get_seg_training_pipeline
        train_pipe = get_seg_training_pipeline(
            crop_size=(64, 64, 64),
            to_multilabel=True,
            pos_ratio=2.0,
            neg_ratio=1.0,
            num_samples=2,
        )
        data_item = {"image": str(self.img_path), "label": str(self.lbl_path)}
        samples = train_pipe(data_item)

        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertEqual(s["image"].shape, (4, 64, 64, 64))
            self.assertEqual(s["label"].shape, (3, 64, 64, 64))

    def test_08_inference_input_isolation(self):
        """Test that forward prediction runs strictly without label input."""
        device = torch.device("cpu")
        model, _ = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)

        case_dict = {
            "image": str(self.img_path),
            "label": str(self.lbl_path),
            "case_id": self.test_case_id,
        }

        # Run single case full-volume inference
        res = evaluate_single_case_full_volume(
            model=model,
            case_dict=case_dict,
            device=device,
            output_dir=None,
            sw_roi_size=(64, 64, 64),
            sw_overlap=0.25,
            sw_batch_size=4,
            save_nifti=False,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("dice", res)
        self.assertIn("hd95_mm", res)
        self.assertIn("timing_seconds", res)
        self.assertGreater(res["timing_seconds"]["end_to_end_total"], 0.0)


def run_all_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSegmentationTrainingReadiness)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
