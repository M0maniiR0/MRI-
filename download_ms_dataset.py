#!/usr/bin/env python3
"""
download_ms_dataset.py
======================
NeuroScope AI — Multiple Sclerosis (التصلب المتعدد / التصلب اللويحي) Dataset Pipeline.
Downloads official MSLesSeg-2024 benchmark dataset (1.37 GB) from Figshare / Nature Scientific Data:
  - 115 3D scans of 75 Multiple Sclerosis patients.
  - Modalities: High-resolution FLAIR, T1, T2 co-registered to MNI152 1mm isotropic space.
  - Ground Truth: Expert-annotated Multiple Sclerosis demyelinating lesion masks.
Zero authentication required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

FIGSHARE_URL = "https://ndownloader.figshare.com/files/52771814"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEST = SCRIPT_DIR / "Task_MultipleSclerosis"


def download_with_progress(url: str, output_path: Path):
    print(f"Connecting to: {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2 MB chunks
        t0 = time.time()
        last_log = t0

        print(f"Total archive size: {total_size / (1024 * 1024):.1f} MB", flush=True)
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

    print(f"\n[OK] Downloaded archive to: {output_path.name}", flush=True)


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name} to {extract_to} ...", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    dur = time.time() - t0
    print(f"[OK] Extracted archive in {dur:.1f}s", flush=True)


def build_ms_manifest(data_dir: Path) -> List[Dict[str, Any]]:
    """Accurately indexes all Multiple Sclerosis sessions with FLAIR, T1, T2 and MASK."""
    manifest = []
    all_masks = sorted(data_dir.rglob("*_MASK.nii.gz"))
    print(f"Indexing Multiple Sclerosis cases from {len(all_masks)} ground-truth masks ...", flush=True)

    for mask_path in all_masks:
        prefix = mask_path.name.replace("_MASK.nii.gz", "")
        parent = mask_path.parent
        flair = parent / f"{prefix}_FLAIR.nii.gz"
        t1 = parent / f"{prefix}_T1.nii.gz"
        t2 = parent / f"{prefix}_T2.nii.gz"

        if flair.is_file():
            manifest.append({
                "patient_id": prefix,
                "mri_path": str(flair.resolve()),
                "flair_path": str(flair.resolve()),
                "t1_path": str(t1.resolve()) if t1.is_file() else None,
                "t2_path": str(t2.resolve()) if t2.is_file() else None,
                "lesion_mask_path": str(mask_path.resolve()),
            })

    manifest_file = data_dir / "dataset_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"  MULTIPLE SCLEROSIS COHORT: {len(manifest)} 3D scans/sessions")
    print(f"  Manifest: {manifest_file.resolve()}")
    print("=" * 80, flush=True)
    return manifest

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download Multiple Sclerosis MSLesSeg Dataset")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    temp_zip = args.dest / "MSLesSeg_Dataset.zip"

    # 1. Download
    extracted_flag = args.dest / ".extracted"
    if not extracted_flag.is_file():
        if not temp_zip.is_file() or temp_zip.stat().st_size < 1_000_000_000:
            download_with_progress(FIGSHARE_URL, temp_zip)
        
        # 2. Extract
        extract_zip(temp_zip, args.dest)
        extracted_flag.write_text("ok", encoding="utf-8")

        # 3. Clean up zip
        if temp_zip.is_file():
            temp_zip.unlink()
            print("[CLEANUP] Deleted temporary zip archive.", flush=True)
    else:
        print("[INFO] Already extracted. Skipping download.", flush=True)

    # 4. Build manifest
    build_ms_manifest(args.dest)


if __name__ == "__main__":
    main()
