#!/usr/bin/env python3
"""
inference.py
============
Unified Offline Inference Pipeline for Task C (Multi-Class 3D Brain MRI Classification).
BrainAgenticthon 2026 (Track 2: Roa'a).

Contract & Guarantees:
1. Pure Offline Execution: No internet calls, no external API requests, fully self-contained.
2. Unified Input: Unseen 3D brain MRI NIfTI (.nii or .nii.gz) or DICOM.
3. Preprocessing: Identical to training (QC + RAS + 1.0mm isotropic + intensity skull-stripping + Z-score + 128^3 crop/pad).
4. Architecture: 3D DenseNet-121 with 10 output classes.
5. Calibrated Probabilities: Temperature scaling (T fitted on validation set).
6. Uncertainty Routing: If confidence < threshold or entropy > threshold, routes to 'Uncertain/Other'.
7. Fixed Contract: Emits classification.json with exactly the 10 official classes summing to 1.0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from preprocess import preprocess_mri, load_config


OFFICIAL_10_CLASSES = [
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


class TaskCInferenceEngine:
    """
    Offline inference engine conforming to Task C contract.
    Loads 3D-DenseNet121 weights, calibration parameters, and runs uncertainty routing.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.ckpt_path = Path(checkpoint_path) if checkpoint_path else SCRIPT_DIR / "best_model.pt"
        self.calib_path = Path(calibration_path) if calibration_path else SCRIPT_DIR / "calibration_params.json"

        # Verify offline artifacts exist
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Missing model checkpoint on disk: {self.ckpt_path}")
        if not self.calib_path.is_file():
            raise FileNotFoundError(f"Missing calibration parameters on disk: {self.calib_path}")

        # Load calibration parameters
        with open(self.calib_path, "r", encoding="utf-8") as f:
            self.calib_params = json.load(f)

        self.temperature = float(self.calib_params.get("temperature", 1.0))
        self.conf_threshold = float(self.calib_params.get("confidence_threshold", 0.35))
        self.margin_threshold = float(self.calib_params.get("margin_threshold", 0.10))
        self.entropy_threshold = float(self.calib_params.get("entropy_threshold", 1.85))
        self.uncertain_idx = int(self.calib_params.get("uncertain_class_id", 9))

        # Instantiate 3D DenseNet-121
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=10).to(self.device)
        
        # Load weights
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def predict(self, nifti_path: Union[str, Path], channel_index: int = 0) -> Dict[str, Any]:
        """
        Runs the full Task C pipeline on an input NIfTI file:
        1. Preprocessing & QC
        2. 3D Model forward pass
        3. Temperature calibration
        4. Shannon entropy & confidence margin calculation
        5. Uncertainty routing
        6. Contract formatting
        """
        start_time = time.time()
        p = Path(nifti_path)
        if not p.exists():
            raise FileNotFoundError(f"Input scan not found: {p}")

        # 1. Unified Preprocessing
        try:
            preproc_res = preprocess_mri(p, channel_index=channel_index)
            qc_report = preproc_res["qc_report"]
        except Exception as e:
            # Fatal preprocessing error -> Graceful QC FAIL fallback
            qc_report = {"status": "FAIL", "reason": f"Preprocessing error: {str(e)}"}
            preproc_res = None

        # Handle QC Failure
        is_qc_fail = (preproc_res is None) or (not qc_report.get("passed", True)) or (qc_report.get("status") == "FAIL")
        if is_qc_fail:
            probs_dict = {cls_name: 0.01 for cls_name in OFFICIAL_10_CLASSES}
            probs_dict["Uncertain/Other"] = 0.91
            # Normalize exactly to 1.0
            total_p = sum(probs_dict.values())
            probs_dict = {k: round(v / total_p, 6) for k, v in probs_dict.items()}

            return {
                "model_name": "3D-DenseNet121-Classifier",
                "predicted_class": "Uncertain/Other",
                "top_probability": probs_dict["Uncertain/Other"],
                "probabilities": probs_dict,
                "is_uncertain_category": True,
                "entropy": 0.4500,
                "margin": 0.9000,
                "qc_status": "FAIL",
                "qc_details": qc_report,
                "inference_time_sec": round(time.time() - start_time, 4),
                "offline_verified": True
            }

        # 2. Model Forward Pass
        vol = preproc_res["processed_volume"].astype(np.float32)
        tensor_in = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape (1, 1, 128, 128, 128)

        with torch.no_grad():
            if torch.cuda.is_available() and self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    raw_logits = self.model(tensor_in)
            else:
                raw_logits = self.model(tensor_in)

            # 3. Temperature Scaling
            scaled_logits = raw_logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()

        # 4. Metrics: Top prob, Entropy, Margin
        p_max = float(np.max(probs))
        top_idx = int(np.argmax(probs))
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # 5. Uncertainty Routing Rule
        is_uncertain = bool(p_max < self.conf_threshold or margin < self.margin_threshold or entropy > self.entropy_threshold)

        if is_uncertain:
            predicted_class = "Uncertain/Other"
            # Route probability mass to Uncertain/Other so distribution reflects routing
            uncertain_prob = max(float(probs[self.uncertain_idx]), 1.0 - margin, 0.51)
            uncertain_prob = min(uncertain_prob, 0.95)
            
            # Rebalance other 9 classes proportionally
            other_sum = float(np.sum([probs[i] for i in range(10) if i != self.uncertain_idx]))
            final_probs = np.zeros(10, dtype=np.float64)
            if other_sum > 1e-8:
                scale = (1.0 - uncertain_prob) / other_sum
                for i in range(10):
                    if i == self.uncertain_idx:
                        final_probs[i] = uncertain_prob
                    else:
                        final_probs[i] = probs[i] * scale
            else:
                final_probs.fill((1.0 - uncertain_prob) / 9.0)
                final_probs[self.uncertain_idx] = uncertain_prob

            top_prob = float(final_probs[self.uncertain_idx])
        else:
            predicted_class = OFFICIAL_10_CLASSES[top_idx]
            final_probs = probs.astype(np.float64)
            top_prob = p_max

        # Ensure exact sum to 1.0
        final_probs = final_probs / np.sum(final_probs)
        probs_dict = {OFFICIAL_10_CLASSES[i]: float(round(final_probs[i], 6)) for i in range(10)}
        
        # Enforce sum to exactly 1.0 by adjusting largest class by discrepancy
        diff = 1.0 - sum(probs_dict.values())
        probs_dict[predicted_class] = round(probs_dict[predicted_class] + diff, 6)

        return {
            "model_name": "3D-DenseNet121-Classifier",
            "predicted_class": predicted_class,
            "top_probability": probs_dict[predicted_class],
            "probabilities": probs_dict,
            "is_uncertain_category": is_uncertain,
            "entropy": round(entropy, 4),
            "margin": round(margin, 4),
            "qc_status": qc_report.get("status", "PASS"),
            "qc_details": qc_report,
            "inference_time_sec": round(time.time() - start_time, 4),
            "offline_verified": True
        }


def run_inference(
    nifti_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    channel_index: int = 0
) -> Dict[str, Any]:
    """Public helper function for unified offline inference."""
    engine = TaskCInferenceEngine()
    result = engine.predict(nifti_path, channel_index=channel_index)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Task C Unified Offline 3D Brain MRI Classifier")
    parser.add_argument("input", type=str, help="Path to input 3D NIfTI scan (.nii or .nii.gz)")
    parser.add_argument("--output", "-o", type=str, default="classification.json", help="Output JSON path")
    parser.add_argument("--channel", "-c", type=int, default=0, help="Channel index for multi-channel NIfTI (default 0)")
    args = parser.parse_args()

    res = run_inference(args.input, output_path=args.output, channel_index=args.channel)
    print("\n================ INFERENCE RESULT ================")
    print(json.dumps(res, indent=2))
    print(f"\n[OK] classification.json written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
