#!/usr/bin/env python3
"""
download_and_expand_ms.py
=========================
NeuroScope AI — Multiple Sclerosis & Demyelinating Lesions Cohort Expansion.
Downloads the White Matter Hyperintensities (WMH) Challenge dataset (125 cases)
from Hugging Face (MedOtter/wmh-segmentation) with correct unique patient IDs,
merging with MSLesSeg (115 cases) to form a 240-case consolidated cohort.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Task_MultipleSclerosis"
WMH_DIR = DATA_DIR / "WMH_Cohort"
HF_BASE = "https://huggingface.co/datasets/MedOtter/wmh-segmentation/resolve/main"
PREFIX_STR = "data/nii/WMH_Segmentation_Challenge/"


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    if dest_path.is_file() and dest_path.stat().st_size > 500:
        # Check if valid gzip
        try:
            with gzip.open(dest_path, "rb") as f:
                while f.read(1024 * 512):
                    pass
            return True
        except Exception:
            dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
            # Verify gzip integrity
            with gzip.open(dest_path, "rb") as f:
                while f.read(1024 * 512):
                    pass
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[FAIL] {dest_path.name}: {e}", flush=True)
                if dest_path.is_file():
                    dest_path.unlink()
                return False
            time.sleep(1.0)
    return False


def fetch_wmh_case(case_dict: Dict) -> Dict | None:
    img_rel = case_dict["image"].replace(PREFIX_STR, "")
    mask_rel = case_dict["mask"].replace(PREFIX_STR, "")

    # URL encode for paths with spaces
    img_url = f"{HF_BASE}/{urllib.parse.quote(img_rel)}"
    mask_url = f"{HF_BASE}/{urllib.parse.quote(mask_rel)}"

    parts = img_rel.split("/")
    split_name = parts[1]
    center = parts[2]
    pid = parts[-3]
    scanner = parts[-4] if len(parts) > 6 else ""
    unique_pid = f"wmh_{center}_{scanner}_{pid}".replace("__", "_")

    dest_folder = WMH_DIR / unique_pid
    dest_img = dest_folder / "FLAIR.nii.gz"
    dest_mask = dest_folder / "wmh_mask.nii.gz"

    ok_img = download_file(img_url, dest_img)
    ok_mask = download_file(mask_url, dest_mask)

    if ok_img and ok_mask:
        return {
            "patient_id": unique_pid,
            "mri_path": str(dest_img.resolve()),
            "flair_path": str(dest_img.resolve()),
            "lesion_mask_path": str(dest_mask.resolve()),
            "center": center,
            "split": split_name,
        }
    return None


def main():
    print("=" * 80)
    print("  EXPANDING MULTIPLE SCLEROSIS & WHITE MATTER LESION COHORT TO 240 CASES")
    print("=" * 80, flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WMH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load MSLesSeg original cases
    manifest_file = DATA_DIR / "dataset_manifest.json"
    mslesseg_manifest = []
    if manifest_file.is_file():
        old_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        mslesseg_manifest = [m for m in old_manifest if not m.get("patient_id", "").startswith("wmh_")]
    print(f"[FOUND] Existing MSLesSeg scans: {len(mslesseg_manifest)}")

    needed_wmh = 240 - len(mslesseg_manifest)  # 125 cases
    print(f"[TARGET] Downloading {needed_wmh} White Matter Lesion cases from Hugging Face...")

    wmh_records = []
    for split_file in ["train", "test"]:
        url = f"https://huggingface.co/datasets/MedOtter/wmh-segmentation/raw/main/{split_file}.jsonl"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            for line in resp:
                wmh_records.append(json.loads(line.decode()))
                if len(wmh_records) >= needed_wmh:
                    break
        if len(wmh_records) >= needed_wmh:
            break

    print(f"[DISCOVERY] Selected {len(wmh_records)} WMH records. Starting download...")

    t0 = time.time()
    new_cases = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_wmh_case, rec) for rec in wmh_records]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                new_cases.append(res)
                if len(new_cases) % 20 == 0 or len(new_cases) == len(wmh_records):
                    print(f"  [Progress] Verified & Cached {len(new_cases)} / {len(wmh_records)} WMH cases...", flush=True)

    dur = time.time() - t0
    print(f"[OK] Downloaded/Verified {len(new_cases)} WMH cases in {dur:.1f}s!", flush=True)

    consolidated_manifest = mslesseg_manifest + new_cases
    manifest_file.write_text(json.dumps(consolidated_manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  SUCCESS! Consolidated Multiple Sclerosis Cohort: {len(consolidated_manifest)} cases")
    print(f"  - MSLesSeg Scans : {len(mslesseg_manifest)}")
    print(f"  - WMH Cohort     : {len(new_cases)}")
    print(f"  - Saved Manifest : {manifest_file.resolve()}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
