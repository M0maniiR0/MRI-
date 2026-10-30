import unittest
import json
import tempfile
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nibabel as nib
import torch

from inference import TaskCInferenceEngine, OFFICIAL_10_CLASSES, run_inference

class TestTask7Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.engine = TaskCInferenceEngine()
        
        # Find a real test scan
        glioma_files = sorted((cls.base_dir / "Task01_BrainTumour" / "Task01_BrainTumour" / "imagesTr").glob("*.nii.gz"))
        cls.real_scan_path = glioma_files[0] if glioma_files else None

    def test_official_10_classes_exact_names(self):
        expected_classes = [
            "Normal",
            "Glioma",
            "Meningioma",
            "Metastasis",
            "Alzheimer's",
            "FCD epilepsy",
            "Post-stroke lesion",
            "Parkinson's",
            "ASD",
            "Uncertain/Other"
        ]
        self.assertEqual(OFFICIAL_10_CLASSES, expected_classes)

    def test_real_scan_inference_contract(self):
        if self.real_scan_path is None or not self.real_scan_path.exists():
            self.skipTest("No real scan found for testing")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "classification.json"
            res = run_inference(self.real_scan_path, output_path=out_json, channel_index=1)

            # 1. Output file written
            self.assertTrue(out_json.exists())
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)

            # 2. Check top-level keys
            required_keys = [
                "model_name", "predicted_class", "top_probability",
                "probabilities", "is_uncertain_category", "entropy",
                "margin", "qc_status", "offline_verified"
            ]
            for k in required_keys:
                self.assertIn(k, saved, f"Missing key in classification.json: {k}")

            # 3. Check official 10 classes
            probs = saved["probabilities"]
            self.assertEqual(list(probs.keys()), OFFICIAL_10_CLASSES)

            # 4. Probabilities sum to 1.0
            prob_sum = sum(probs.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=4, msg=f"Probabilities sum to {prob_sum}, expected 1.0")

            # 5. Top probability matches predicted class
            pred_class = saved["predicted_class"]
            self.assertIn(pred_class, OFFICIAL_10_CLASSES)
            self.assertAlmostEqual(saved["top_probability"], probs[pred_class], places=4)

            # 6. Offline flag
            self.assertTrue(saved["offline_verified"])

    def test_corrupted_or_zero_scan_routes_to_uncertain(self):
        """A synthetic empty/corrupted scan must fail QC or route to Uncertain/Other."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_nii_path = Path(tmp_dir) / "corrupt_empty_scan.nii.gz"
            # Zero volume -> empty brain mask -> QC FAIL
            empty_arr = np.zeros((32, 32, 32), dtype=np.float32)
            nii = nib.Nifti1Image(empty_arr, affine=np.eye(4))
            nib.save(nii, str(bad_nii_path))

            out_json = Path(tmp_dir) / "classification_corrupt.json"
            res = run_inference(bad_nii_path, output_path=out_json)

            self.assertEqual(res["predicted_class"], "Uncertain/Other")
            self.assertTrue(res["is_uncertain_category"])
            self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
