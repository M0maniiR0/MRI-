import unittest
import json
from pathlib import Path

class TestTask6CalibrationAcceptance(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.params_path = self.base_dir / "calibration_params.json"
        self.plot_path = self.base_dir / "reports" / "calibration_curve.png"

    def test_calibration_artifacts_exist(self):
        self.assertTrue(self.params_path.exists(), "calibration_params.json must exist")
        self.assertTrue(self.plot_path.exists(), "reports/calibration_curve.png must exist")
        self.assertGreater(self.plot_path.stat().st_size, 1000, "Calibration curve PNG must not be empty")

    def test_calibration_parameters_valid(self):
        with open(self.params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("temperature", params)
        self.assertIn("ece_before", params)
        self.assertIn("ece_after", params)
        self.assertIn("confidence_threshold", params)
        self.assertIn("entropy_threshold", params)

        self.assertGreater(params["temperature"], 0.0, "Temperature must be positive")
        self.assertLessEqual(params["ece_after"], params["ece_before"] + 1e-4, 
                             "Calibrated ECE should be <= uncalibrated ECE")
        self.assertGreaterEqual(params["confidence_threshold"], 0.1)
        self.assertLessEqual(params["confidence_threshold"], 1.0)

if __name__ == "__main__":
    unittest.main()
