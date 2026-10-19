#!/usr/bin/env python3
"""
download_meningioma_sample.py
==============================
Downloads official BraTS 2023 Meningioma (BraTS-MEN) 3D MRI cases from Hugging Face.
Dataset: MedOtter/brats2023-men-dataset
Each patient includes:
  - t1n.nii.gz (T1 native)
  - t1c.nii.gz (T1 contrast-enhanced)
  - t2f.nii.gz (T2-FLAIR)
  - t2w.nii.gz (T2-weighted)
  - seg.nii.gz (Ground-truth segmentation mask)
Zero authentication required. Pure HTTP streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

REPO_ID = "MedOtter/brats2023-men-dataset"
TREE_API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/BraTS-MEN-Train"
RAW_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

MODALITIES = ["t1n", "t1c", "t2f", "t2w", "seg"]


def list_available_cases(max_cases: int = 50) -> List[str]:
    """Fetches list of patient folder names from Hugging Face API."""
    print(f"Connecting to Hugging Face API for {REPO_ID}...")
    req = urllib.request.Request(TREE_API, headers={"User-Agent": "NeuroScope-MedicalAI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    cases = [Path(item["path"]).name for item in data if item.get("type") == "directory"]
    cases.sort()
    print(f"Total Meningioma cases discovered on remote: {len(cases)}")
    return cases[:max_cases]


def download_file(remote_url: str, local_path: Path, max_retries: int = 3) -> bool:
    """Downloads a single file with streaming and retry logic."""
    if local_path.is_file() and local_path.stat().st_size > 1000:
        return True  # already exists

    local_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"    [FAIL] Failed after {max_retries} attempts: {e}")
                if local_path.exists():
                    local_path.unlink()
                return False
            time.sleep(2)
    return False


def download_cohort(target_dir: Path, num_cases: int = 20) -> List[Dict[str, Any]]:
    """Downloads a cohort of Meningioma cases."""
    target_dir.mkdir(parents=True, exist_ok=True)
    available = list_available_cases(max_cases=num_cases)
    print(f"\nTarget download cohort: {len(available)} Meningioma cases into: {target_dir.resolve()}\n")

    manifest = []
    t_start = time.time()

    for idx, case_id in enumerate(available, start=1):
        print(f"[{idx:>2}/{len(available)}] Downloading {case_id}...")
        case_dir = target_dir / case_id
        case_files = {}
        all_ok = True

        for mod in MODALITIES:
            file_name = f"{case_id}-{mod}.nii.gz"
            remote_url = f"{RAW_BASE}/BraTS-MEN-Train/{case_id}/{file_name}"
            local_path = case_dir / file_name

            ok = download_file(remote_url, local_path)
            if ok:
                case_files[mod] = str(local_path.resolve())
            else:
                all_ok = False
                break

        if all_ok:
            manifest.append({"case_id": case_id, "files": case_files})
            print(f"    [OK] {case_id} downloaded successfully.")
        else:
            print(f"    [SKIP] Incomplete files for {case_id}")

    elapsed = time.time() - t_start
    manifest_file = target_dir / "dataset_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=======================================================")
    print(f"  DOWNLOAD COMPLETE: {len(manifest)} cases in {elapsed:.1f}s")
    print(f"  Manifest: {manifest_file.resolve()}")
    print(f"=======================================================")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download BraTS Meningioma dataset from Hugging Face")
    parser.add_argument("--dest", type=Path, default=Path("Task_Meningioma/BraTS_MEN"), help="Destination directory")
    parser.add_argument("--num-cases", type=int, default=20, help="Number of cases to download (default: 20)")
    args = parser.parse_args()

    download_cohort(args.dest, args.num_cases)


if __name__ == "__main__":
    main()
