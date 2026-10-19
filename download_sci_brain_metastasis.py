#!/usr/bin/env python3
"""
download_sci_brain_metastasis.py
================================
Downloads and extracts the official SCI Brain Metastases multi-modal MRI cohort (1.79 GB)
from Hugging Face: Emulated-Inc/sci-brain-metastases-new-centre
Each patient contains:
  - t1c.nii.gz (Contrast-enhanced T1)
  - t1n.nii.gz (Native T1)
  - t2f.nii.gz (T2-FLAIR)
  - t2w.nii.gz (T2-weighted)
  - seg.nii.gz (Ground-truth Metastatic Lesion mask)
Zero authentication required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

TAR_URL = "https://huggingface.co/datasets/Emulated-Inc/sci-brain-metastases-new-centre/resolve/main/train.tar"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEST = SCRIPT_DIR / "Task_BrainMetastasis"


def download_with_progress(url: str, output_path: Path):
    print(f"Connecting to: {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2 MB chunks
        t0 = time.time()
        last_log = t0

        print(f"Total archive size: {total_size / (1024 * 1024):.1f} MB")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_log >= 5.0 or downloaded == total_size:
                pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                mb = downloaded / (1024 * 1024)
                speed = mb / max(0.1, now - t0)
                print(f"  [Progress] {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1f}%) | {speed:.1f} MB/s", flush=True)
                last_log = now

    print(f"\n[OK] Downloaded archive to: {output_path.name}")


def extract_archive(tar_path: Path, extract_to: Path):
    print(f"Extracting {tar_path.name} to {extract_to} ...", flush=True)
    t0 = time.time()
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)
    dur = time.time() - t0
    print(f"[OK] Extracted archive in {dur:.1f}s", flush=True)


def build_consolidated_manifest(data_dir: Path) -> List[Dict]:
    """Scans all patient folders in data_dir (both Pakawat P* and SCI train/p*) and builds manifest."""
    manifest = []

    # 1. Check SCI train patient folders
    sci_train = data_dir / "train"
    if sci_train.is_dir():
        for p_dir in sorted(sci_train.iterdir()):
            if p_dir.is_dir():
                t1c = p_dir / "t1c.nii.gz"
                t1n = p_dir / "t1n.nii.gz"
                t2f = p_dir / "t2f.nii.gz"
                t2w = p_dir / "t2w.nii.gz"
                seg = p_dir / "seg.nii.gz"
                if t1c.is_file() and seg.is_file():
                    manifest.append({
                        "patient_id": f"sci_{p_dir.name}",
                        "mri_path": str(t1c.resolve()),
                        "t1n_path": str(t1n.resolve()) if t1n.is_file() else None,
                        "t2f_path": str(t2f.resolve()) if t2f.is_file() else None,
                        "t2w_path": str(t2w.resolve()) if t2w.is_file() else None,
                        "lesion_mask_path": str(seg.resolve()),
                    })

    # 2. Check original Pakawat cases
    for p_dir in sorted(data_dir.iterdir()):
        if p_dir.is_dir() and p_dir.name.startswith("P"):
            mri = p_dir / "mri.nii.gz"
            seg = p_dir / "lesion_mask.nii.gz"
            if mri.is_file() and seg.is_file():
                manifest.append({
                    "patient_id": p_dir.name,
                    "mri_path": str(mri.resolve()),
                    "lesion_mask_path": str(seg.resolve()),
                })

    manifest_file = data_dir / "dataset_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n=======================================================")
    print(f"  CONSOLIDATED BRAIN METASTASIS COHORT: {len(manifest)} patients")
    print(f"  Manifest: {manifest_file.resolve()}")
    print(f"=======================================================")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download SCI Brain Metastases Dataset")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    temp_tar = args.dest / "train_metastases.tar"

    # 1. Download archive if not already extracted
    if not (args.dest / "train").is_dir():
        if not temp_tar.is_file() or temp_tar.stat().st_size < 1_000_000_000:
            download_with_progress(TAR_URL, temp_tar)

        # 2. Extract
        extract_archive(temp_tar, args.dest)

        # 3. Cleanup tar to save disk space
        if temp_tar.is_file():
            temp_tar.unlink()
            print("[CLEANUP] Deleted temporary tar archive.")
    else:
        print("[INFO] 'train' directory already exists in target. Skipping download.")

    # 4. Build manifest
    build_consolidated_manifest(args.dest)


if __name__ == "__main__":
    main()
