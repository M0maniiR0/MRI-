#!/usr/bin/env python3
"""
package_clean_deliverables.py
=============================
Packages all final deliverables for Task C (Track 2) into a clean, self-contained ZIP:
- Trained model weights (best_model.pt)
- Calibration parameters (calibration_params.json)
- Full source code (preprocess.py, inference.py, tests, evaluation scripts)
- Detailed verification & evaluation reports (markdown + figures)
- Contract tests & configuration files
Saves directly to Desktop: NeuroScope_Track2_Deliverables.zip
"""

import sys
import zipfile
from pathlib import Path

# Force UTF-8 encoding for console output on Windows
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
DESKTOP_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop")
ZIP_PATH = DESKTOP_DIR / "NeuroScope_Track2_Deliverables.zip"

FILES_TO_PACK = [
    # Core Model & Calibration Weights (Root level for instant runnable inference)
    ("best_model.pt", "best_model.pt"),
    ("calibration_params.json", "calibration_params.json"),
    ("preprocessing_config.yaml", "preprocessing_config.yaml"),
    ("label_map.json", "label_map.json"),
    ("requirements.txt", "requirements.txt"),
    ("README.md", "README.md"),
    
    # Core Pipeline & Execution Scripts (Root level)
    ("preprocess.py", "preprocess.py"),
    ("inference.py", "inference.py"),
    ("train_native_t1_classifier.py", "train_native_t1_classifier.py"),
    ("calibrate_native_t1.py", "calibrate_native_t1.py"),
    ("evaluate_native_t1_bootstrap.py", "evaluate_native_t1_bootstrap.py"),
    ("evaluate_with_abstention.py", "evaluate_with_abstention.py"),
    ("run_shortcut_diagnosis_native_t1.py", "run_shortcut_diagnosis_native_t1.py"),
    ("run_source_recognition_check.py", "run_source_recognition_check.py"),
    ("cache_dataset.py", "cache_dataset.py"),

    # Contract Tests
    ("tests/test_task7_contract.py", "tests/test_task7_contract.py"),
    
    # Reports, Logs & Results
    ("reports/task_c_evaluation_report.md", "reports/task_c_evaluation_report.md"),
    ("reports/calibration_curve.png", "reports/calibration_curve.png"),
    ("coverage_report.md", "reports/coverage_report.md"),
    ("data_registry.csv", "reports/data_registry.csv"),
    ("test_evaluation_results.json", "reports/test_evaluation_results.json"),
    ("source_recognition_results.json", "reports/source_recognition_results.json"),
    ("abstention_evaluation_results.json", "reports/abstention_evaluation_results.json"),
    ("training_log.csv", "reports/training_log.csv"),
]

print(f"Creating ZIP archive at: {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for src_rel, arc_name in FILES_TO_PACK:
        src_path = BASE_DIR / src_rel
        if src_path.exists():
            zf.write(src_path, arc_name)
            print(f"  [+] Added: {src_rel} -> {arc_name} ({src_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [-] Warning: File not found: {src_path}")

print(f"\n[OK] Deliverables successfully packaged into: {ZIP_PATH}")
print(f"Final archive size: {ZIP_PATH.stat().st_size / (1024 * 1024):.2f} MB")
