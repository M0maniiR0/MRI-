#!/usr/bin/env python3
"""
tests/test_unified_inference.py
===============================
Unit and integration test suite for NeuroScope AI Unified Inference Module.
Verifies:
1. Architecture loading and weight caching
2. Offline execution guarantees
3. Error handling and uncertainty states for invalid/missing inputs
4. Native-space mask generation (shape, affine, header preservation)
5. Geometric volumetry calculation without ground-truth masks
6. Discrete label compliance
7. Competition alignment flags (Core vs. Research-only)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from unified_inference import UnifiedInferenceEngine


class TestUnifiedInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = UnifiedInferenceEngine()
        cls.test_out_dir = Path(tempfile.mkdtemp(prefix="test_inference_"))

    @classmethod
    def tearDownClass(cls):
        if cls.test_out_dir.exists():
            shutil.rmtree(cls.test_out_dir)

    def test_01_registry_and_device(self):
        """Verifies registry configuration and offline hardware initialization."""
        self.assertIsNotNone(self.engine.device)
        self.assertEqual(len(self.engine.MODEL_REGISTRY), 5)
        for k in ["glioma", "meningioma", "hippocampus", "brain_metastasis", "multiple_sclerosis"]:
            self.assertIn(k, self.engine.MODEL_REGISTRY)
            cfg = self.engine.MODEL_REGISTRY[k]
            self.assertTrue(cfg["checkpoint"].is_file(), f"Checkpoint missing for {k}: {cfg['checkpoint']}")

    def test_02_invalid_inputs_error_handling(self):
        """Confirms system does not crash on missing files; returns explicit error state."""
        res = self.engine.predict(
            target_model="glioma",
            input_paths={"flair": "non_existent_file.nii.gz"}
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["error_code"], "INVALID_OR_MISSING_INPUTS")
        self.assertEqual(res["uncertainty_state"], "HIGH_UNCERTAINTY")
        self.assertIn("Missing required modalities", res["message"])

    def test_03_unsupported_model_error_handling(self):
        """Verifies error reporting when an unknown model key is requested."""
        res = self.engine.predict(
            target_model="unknown_pathology_model",
            input_paths={"flair": "some_file.nii.gz"}
        )
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["uncertainty_state"], "HIGH_UNCERTAINTY")

    def test_04_hippocampus_inference(self):
        """Tests end-to-end Hippocampus inference without ground-truth masks."""
        hippo_case = Path("Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
        if not hippo_case.is_file():
            self.skipTest(f"Sample case {hippo_case} not found.")

        orig_nii = nib.load(str(hippo_case))
        orig_shape = list(orig_nii.shape)

        res = self.engine.predict(
            target_model="hippocampus",
            input_paths=hippo_case,
            output_dir=self.test_out_dir,
            case_id="TEST_HIPPO_001",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["native_shape"], orig_shape)
        self.assertIn("total_hippocampus_volume_mm3", res["volumes"])
        self.assertGreater(res["volumes"]["total_hippocampus_volume_mm3"], 0)
        self.assertLess(res["inference_time_sec"], 2.0)

        # Check mask on disk
        mask_path = Path(res["mask_path"])
        self.assertTrue(mask_path.is_file())
        saved_nii = nib.load(str(mask_path))
        self.assertEqual(list(saved_nii.shape), orig_shape)
        np.testing.assert_allclose(saved_nii.affine, orig_nii.affine, rtol=1e-5)
        unique_labels = set(np.unique(saved_nii.get_fdata()))
        self.assertTrue(unique_labels.issubset({0, 1, 2}))

    def test_05_metastasis_inference(self):
        """Tests Brain Metastasis single-channel inference."""
        sample_case = Path("Task_BrainMetastasis/P01/mri.nii.gz")
        if not sample_case.is_file():
            # Try alternate from manifest
            manifest = Path("Task_BrainMetastasis/dataset_manifest.json")
            if manifest.is_file():
                with open(manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sample_case = Path(data[0]["mri_path"])

        if not sample_case.is_file():
            self.skipTest("No metastasis sample file found.")

        orig_nii = nib.load(str(sample_case))
        res = self.engine.predict(
            target_model="brain_metastasis",
            input_paths={"t1c": sample_case},
            output_dir=self.test_out_dir,
            case_id="TEST_METASTASIS_001",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["native_shape"], list(orig_nii.shape))
        self.assertIn("metastatic_volume_mm3", res["volumes"])
        self.assertTrue(Path(res["mask_path"]).is_file())

    def test_06_research_flag_multiple_sclerosis(self):
        """Verifies that Multiple Sclerosis is flagged as a research model."""
        sample_case = Path("Task_MultipleSclerosis/WMH_Cohort/wmh_Utrecht_1.nii.gz")
        if not sample_case.is_file():
            manifest = Path("Task_MultipleSclerosis/dataset_manifest.json")
            if manifest.is_file():
                with open(manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sample_case = Path(data[0]["mri_path"])

        if not sample_case.is_file():
            self.skipTest("No MS sample file found.")

        orig_nii = nib.load(str(sample_case))
        res = self.engine.predict(
            target_model="multiple_sclerosis",
            input_paths=sample_case,
            output_dir=self.test_out_dir,
            case_id="TEST_MS_001",
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["research_only"], "MS model must be flagged as research_only=True")
        self.assertFalse(res["competition_core"], "MS model must have competition_core=False")
        self.assertEqual(res["native_shape"], list(orig_nii.shape))


if __name__ == "__main__":
    unittest.main()
