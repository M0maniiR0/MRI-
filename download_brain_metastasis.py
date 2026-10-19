#!/usr/bin/env python3
"""
download_brain_metastasis.py
============================
Downloads official 3D Brain Metastases MRI dataset from Hugging Face:
Repository: Pakawat-Phasook/brain_metastases_dataset
Includes 40 preprocessed 3D patient scans:
  - mri.nii.gz (High-resolution 3D T1-weighted CE-MRI, shape [240, 240, 155])
  - lesion_mask.nii.gz (Multi-lesion metastatic ground-truth segmentation)
  - brain_mask.nii.gz (Extracted brain mask)
Fast concurrent multithreaded streaming. Zero authentication required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple

REPO_ID = "Pakawat-Phasook/brain_metastases_dataset"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/preproc"
RAW_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/preproc"

FILES_TO_DOWNLOAD = ["mri.nii.gz", "lesion_mask.nii.gz", "brain_mask.nii.gz"]


def list_available_patients() -> List[str]:
    """Fetches patient IDs from Hugging Face API."""
    print(f"Connecting to Hugging Face API for {REPO_ID}...")
    req = urllib.request.Request(API_URL, headers={"User-Agent": "NeuroScope-MedicalAI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    patients = [item["path"].split("/")[-1] for item in data if item.get("type") == "directory"]
    patients.sort()
    print(f"[DISCOVERY] Found {len(patients)} Brain Metastasis patient cohorts on remote.")
    return patients


def download_file_worker(args: Tuple[str, Path, int]) -> bool:
    """Worker function for concurrent downloads with retries."""
    remote_url, local_path, max_retries = args
    if local_path.is_file() and local_path.stat().st_size > 1000:
        return True  # Already cached

    local_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(512 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"    [FAIL] {local_path.name}: {e}")
                if local_path.exists():
                    local_path.unlink()
                return False
            time.sleep(1.0)
    return False


def download_metastasis_dataset(target_dir: Path, max_patients: int = 40, workers: int = 6) -> List[Dict[str, Any]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    all_patients = list_available_patients()[:max_patients]
    print(f"\nTarget download: {len(all_patients)} patients into: {target_dir.resolve()}\n")

    tasks = []
    patient_map = {}
    for pid in all_patients:
        p_dir = target_dir / pid
        patient_map[pid] = {}
        for fname in FILES_TO_DOWNLOAD:
            local_path = p_dir / fname
            remote_url = f"{RAW_BASE}/{pid}/{fname}"
            patient_map[pid][fname] = local_path
            tasks.append((remote_url, local_path, 3))

    t0 = time.time()
    total_files = len(tasks)
    print(f"Starting parallel download ({total_files} files, {workers} threads)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(download_file_worker, tasks))

    success = sum(1 for r in results if r)
    manifest = []
    for pid in all_patients:
        files = patient_map[pid]
        if all(p.is_file() and p.stat().st_size > 500 for p in files.values()):
            manifest.append({
                "patient_id": pid,
                "mri_path": str(files["mri.nii.gz"].resolve()),
                "lesion_mask_path": str(files["lesion_mask.nii.gz"].resolve()),
                "brain_mask_path": str(files["brain_mask.nii.gz"].resolve()),
            })

    elapsed = time.time() - t0
    manifest_path = target_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=======================================================")
    print(f"  BRAIN METASTASIS DOWNLOAD COMPLETE: {len(manifest)}/{len(all_patients)} ready in {elapsed:.1f}s")
    print(f"  Saved Manifest: {manifest_path.resolve()}")
    print(f"=======================================================")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download Brain Metastasis MRI Dataset")
    parser.add_argument("--dest", type=Path, default=Path("Task_BrainMetastasis"))
    parser.add_argument("--num-patients", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    download_metastasis_dataset(args.dest, args.num_patients, args.workers)


if __name__ == "__main__":
    main()
