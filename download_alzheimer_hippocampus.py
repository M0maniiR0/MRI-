#!/usr/bin/env python3
"""
download_alzheimer_hippocampus.py
=================================
Downloads and extracts Medical Segmentation Decathlon Task04_Hippocampus (27 MB).
Clinical Target: Alzheimer's Disease & Brain Atrophy (Hippocampal Volumetry).
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
DEFAULT_DEST = Path("Task04_Hippocampus")


def download_and_extract(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "Task04_Hippocampus.tar"

    if (dest_dir / "imagesTr").is_dir() and (dest_dir / "dataset.json").is_file():
        print(f"[OK] Task04_Hippocampus already exists on disk in: {dest_dir.resolve()}")
        return

    print("=" * 70)
    print("  DOWNLOADING ALZHEIMER'S HIPPOCAMPUS DATASET (MSD Task04)")
    print("=" * 70)
    print(f"  URL: {DATASET_URL}")
    print(f"  Target Destination: {dest_dir.resolve()}")

    t0 = time.time()
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "NeuroScope-MedicalAI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tar_path, "wb") as f:
        downloaded = 0
        total_size = int(resp.headers.get("Content-Length", 28430000))
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size) * 100
            print(f"\r  Downloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)

    print(f"\n  [OK] Download completed in {time.time() - t0:.1f}s.")
    print("  Extracting archive...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_dir.parent)

    # Clean up archive file
    if tar_path.is_file():
        tar_path.unlink()

    # Load and print dataset manifest summary
    json_path = dest_dir / "dataset.json"
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print("\n=======================================================")
        print(f"  DATASET READY: {manifest.get('name')}")
        print(f"  Modality     : {manifest.get('modality')}")
        print(f"  Labels       : {manifest.get('labels')}")
        print(f"  Training     : {len(manifest.get('training', []))} cases")
        print(f"  Testing      : {len(manifest.get('test', []))} cases")
        print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Download Alzheimer's Hippocampus dataset")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination directory")
    args = parser.parse_args()

    download_and_extract(args.dest)


if __name__ == "__main__":
    main()
