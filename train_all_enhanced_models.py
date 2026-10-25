#!/usr/bin/env python3
"""
train_all_enhanced_models.py
============================
NeuroScope AI — Master Retraining Orchestrator for Local NVIDIA RTX 2050 GPU.
Sequentially trains the 4 medically redesigned segmentation models:
  1. Meningioma (BraTS-MEN 2023 - Solid Tumor Core with Skull Filtration)
  2. Brain Metastasis (Multi-Focal Micro-Lesions, No Downsampling)
  3. Hippocampus (Task04_Hippocampus - Bilateral Medial Temporal Volumetry)
  4. Glioma (BraTS / MSD Task01 - 4-Sequence Multi-Region SegResNet)

Strictly obeys clinical rules:
  - Multiple Sclerosis is PERMANENTLY EXCLUDED (preventing overfitting & hallucinations).
  - Employs Automatic Mixed Precision (AMP FP16) to keep peak VRAM under 2.5 GB (safe for 4GB RTX 2050).
  - Clears CUDA memory cache between models to prevent fragmentation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent

# Import individual training routines
from train_meningioma_enhanced import run_meningioma_retraining
from train_brain_metastasis_enhanced import run_metastasis_retraining
from train_hippocampus_enhanced import run_hippocampus_retraining
from finetune_segresnet import run_finetuning, DEFAULT_DATA, DEFAULT_SPLITS_DIR, DEFAULT_INITIAL_CHECKPOINT


def clean_gpu_memory():
    """Forces garbage collection and empties PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="NeuroScope AI Master GPU Retraining Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "meningioma", "metastasis", "hippocampus", "glioma"],
        help="Target model to retrain or 'all' for full sequential execution",
    )
    parser.add_argument("--epochs-meningioma", type=int, default=12)
    parser.add_argument("--epochs-metastasis", type=int, default=12)
    parser.add_argument("--epochs-hippocampus", type=int, default=12)
    parser.add_argument("--epochs-glioma", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    total_start = time.time()

    print("=" * 85, flush=True)
    print("  NEUROSCOPE AI — MASTER ENHANCED RETRAINING PIPELINE (LOCAL GPU)", flush=True)
    print("=" * 85, flush=True)
    print(f"  Target Hardware  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  Target Scope     : {args.model.upper()}", flush=True)
    print("  Multiple Sclerosis: EXCLUDED (Per clinical directive)", flush=True)
    print("=" * 85, flush=True)

    summary_records = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MENINGIOMA RETRAINING
    # ─────────────────────────────────────────────────────────────────────────
    if args.model in ["all", "meningioma"]:
        clean_gpu_memory()
        print("\n" + "#" * 85, flush=True)
        print("  [PHASE 1/4] STARTING ENHANCED MENINGIOMA RETRAINING (Solid Tumor Core Ch1)", flush=True)
        print("#" * 85, flush=True)
        t_m = time.time()
        try:
            run_meningioma_retraining(
                epochs=args.epochs_meningioma,
                lr=3e-4,
                device_str=args.device,
            )
            summary_records["meningioma"] = {
                "status": "COMPLETED",
                "duration_min": round((time.time() - t_m) / 60.0, 2),
                "checkpoint": "models/segresnet_meningioma_enhanced_best.pt",
            }
        except Exception as e:
            print(f"[ERROR in Meningioma]: {e}", flush=True)
            summary_records["meningioma"] = {"status": "FAILED", "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # 2. BRAIN METASTASIS RETRAINING
    # ─────────────────────────────────────────────────────────────────────────
    if args.model in ["all", "metastasis"]:
        clean_gpu_memory()
        print("\n" + "#" * 85, flush=True)
        print("  [PHASE 2/4] STARTING ENHANCED BRAIN METASTASIS RETRAINING (No Downsample)", flush=True)
        print("#" * 85, flush=True)
        t_meta = time.time()
        try:
            run_metastasis_retraining(
                epochs=args.epochs_metastasis,
                lr=2e-4,
                device_str=args.device,
            )
            summary_records["metastasis"] = {
                "status": "COMPLETED",
                "duration_min": round((time.time() - t_meta) / 60.0, 2),
                "checkpoint": "models/segresnet_brain_metastasis_enhanced_best.pt",
            }
        except Exception as e:
            print(f"[ERROR in Metastasis]: {e}", flush=True)
            summary_records["metastasis"] = {"status": "FAILED", "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # 3. HIPPOCAMPUS RETRAINING
    # ─────────────────────────────────────────────────────────────────────────
    if args.model in ["all", "hippocampus"]:
        clean_gpu_memory()
        print("\n" + "#" * 85, flush=True)
        print("  [PHASE 3/4] STARTING ENHANCED HIPPOCAMPUS RETRAINING (Medial Temporal ROI)", flush=True)
        print("#" * 85, flush=True)
        t_h = time.time()
        try:
            run_hippocampus_retraining(
                epochs=args.epochs_hippocampus,
                lr=3e-4,
                device_str=args.device,
            )
            summary_records["hippocampus"] = {
                "status": "COMPLETED",
                "duration_min": round((time.time() - t_h) / 60.0, 2),
                "checkpoint": "models/segresnet_hippocampus_enhanced_best.pt",
            }
        except Exception as e:
            print(f"[ERROR in Hippocampus]: {e}", flush=True)
            summary_records["hippocampus"] = {"status": "FAILED", "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # 4. GLIOMA RETRAINING
    # ─────────────────────────────────────────────────────────────────────────
    if args.model in ["all", "glioma"]:
        clean_gpu_memory()
        print("\n" + "#" * 85, flush=True)
        print("  [PHASE 4/4] STARTING ENHANCED GLIOMA MULTI-REGION RETRAINING (339 Patients)", flush=True)
        print("#" * 85, flush=True)
        t_g = time.time()
        try:
            target_ckpt = SCRIPT_DIR / "models" / "segresnet_glioma_finetuned_best.pt"
            report_file = SCRIPT_DIR / "reports" / "glioma_finetuning_report.json"
            res = run_finetuning(
                data_dir=DEFAULT_DATA.resolve(),
                splits_dir=DEFAULT_SPLITS_DIR.resolve(),
                initial_checkpoint=DEFAULT_INITIAL_CHECKPOINT.resolve(),
                output_checkpoint=target_ckpt.resolve(),
                epochs=args.epochs_glioma,
                lr=5e-5,
                crop_size=(96, 96, 96),
                batch_size=1,
                val_num_cases=10,
                device=device,
                seed=42,
                report_file=report_file.resolve(),
            )
            summary_records["glioma"] = {
                "status": "COMPLETED",
                "duration_min": round((time.time() - t_g) / 60.0, 2),
                "best_dice": res.get("best_dice"),
                "checkpoint": str(target_ckpt.name),
            }
        except Exception as e:
            print(f"[ERROR in Glioma]: {e}", flush=True)
            summary_records["glioma"] = {"status": "FAILED", "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Report
    # ─────────────────────────────────────────────────────────────────────────
    total_dur_min = round((time.time() - total_start) / 60.0, 2)
    master_summary_file = SCRIPT_DIR / "reports" / "enhanced_retraining_master_summary.json"
    with open(master_summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": str(device),
            "total_duration_minutes": total_dur_min,
            "models": summary_records,
        }, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print(f"  ALL REQUESTED RETRAINING TASKS COMPLETED IN {total_dur_min} MINUTES!", flush=True)
    print(f"  Summary Report Saved -> {master_summary_file}", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
