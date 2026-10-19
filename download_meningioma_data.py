#!/usr/bin/env python3
"""
download_meningioma_data.py
=============================
Downloads official BraTS 2023 Meningioma (BraTS-MEN) 3D MRI cases from Hugging Face.
Dataset: MedOtter/brats2023-men-dataset (953 cases available)
Each patient includes:
  - t1n.nii.gz (T1 native)
  - t1c.nii.gz (T1 contrast-enhanced)
  - t2f.nii.gz (T2-FLAIR)
  - t2w.nii.gz (T2-weighted)
  - seg.nii.gz (Ground-truth segmentation mask)
Fast concurrent multithreaded streaming.
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


def download_single_file(args: Tuple[str, Path, int]) -> bool:
    """Downloads a single file with streaming and retry logic."""
    remote_url, local_path, max_retries = args
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
                print(f"    [FAIL] {local_path.name}: {e}")
                if local_path.exists():
                    local_path.unlink()
                return False
            time.sleep(1.5)
    return False


def download_cohort(target_dir: Path, num_cases: int = 30, max_workers: int = 6) -> List[Dict[str, Any]]:
    """Downloads a cohort of Meningioma cases using concurrent threads."""
    target_dir.mkdir(parents=True, exist_ok=True)
    available = list_available_cases(max_cases=num_cases)
    print(f"\nTarget download cohort: {len(available)} Meningioma cases into: {target_dir.resolve()}\n")

    # Build work items for all files
    all_tasks = []
    case_file_map = {}
    for case_id in available:
        case_dir = target_dir / case_id
        case_file_map[case_id] = {}
        for mod in MODALITIES:
            file_name = f"{case_id}-{mod}.nii.gz"
            remote_url = f"{RAW_BASE}/BraTS-MEN-Train/{case_id}/{file_name}"
            local_path = case_dir / file_name
            case_file_map[case_id][mod] = local_path
            all_tasks.append((remote_url, local_path, 3))

    t_start = time.time()
    total_files = len(all_tasks)
    print(f"Downloading {total_files} files across {len(available)} cases ({max_workers} concurrent threads)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_single_file, all_tasks))

    success_count = sum(1 for r in results if r)
    print(f"Download results: {success_count}/{total_files} files ready.")

    manifest = []
    for case_id in available:
        files = case_file_map[case_id]
        if all(p.is_file() and p.stat().st_size > 1000 for p in files.values()):
            manifest.append({
                "case_id": case_id,
                "files": {mod: str(p.resolve()) for mod, p in files.items()}
            })

    elapsed = time.time() - t_start
    manifest_file = target_dir / "dataset_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=======================================================")
    print(f"  DOWNLOAD COMPLETE: {len(manifest)} / {len(available)} cases ready in {elapsed:.1f}s")
    print(f"  Manifest: {manifest_file.resolve()}")
    print(f"=======================================================")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download BraTS Meningioma dataset from Hugging Face")
    parser.add_argument("--dest", type=Path, default=Path("Task_Meningioma/BraTS_MEN"), help="Destination directory")
    parser.add_argument("--num-cases", type=int, default=30, help="Number of cases to download (default: 30)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent worker threads (default: 6)")
    args = parser.parse_args()

    download_cohort(args.dest, args.num_cases, max_workers=args.workers)


if __name__ == "__main__":
    main()
