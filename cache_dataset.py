#!/usr/bin/env python3
"""
cache_dataset.py
================
Preprocesses and caches all 884 native T1 whole-brain scans:
- Glioma (MSD Task01 channel 1 T1w): 484 subjects
- Meningioma (BraTS-MEN t1n): 200 subjects
- Metastasis (Brain-Mets t1n): 200 subjects

Generates:
1. cache_preprocessed/<subject_id>.npy (float16, shape 128x128x128)
2. cache_preprocessed/splits.json (Patient-level 70/15/15 stratified partition)
3. cache_preprocessed/manifest.json (Metadata: shapes, spacings, nonzeros, mask volumes)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np

BASE_DIR = Path(r"C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data")
sys.path.insert(0, str(BASE_DIR))

from preprocess import preprocess_mri

CACHE_DIR = BASE_DIR / "cache_preprocessed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_all_records() -> List[Dict[str, Any]]:
    records = []

    # 1. Glioma (MSD Task01 imagesTr, channel 1 = T1w)
    g_dir = BASE_DIR / "Task01_BrainTumour" / "Task01_BrainTumour" / "imagesTr"
    g_files = sorted([p for p in g_dir.glob("*.nii.gz") if not p.name.startswith(".")])
    for p in g_files:
        sid = p.name.replace(".nii.gz", "")
        records.append({
            "subject_id": sid,
            "path": str(p),
            "class_name": "Glioma",
            "class_id": 1,
            "channel_idx": 1
        })

    # 2. Meningioma (BraTS-MEN *-t1n.nii.gz, channel 0)
    m_dir = BASE_DIR / "Task_Meningioma" / "BraTS_MEN"
    m_files = sorted([p for p in m_dir.glob("*/*-t1n.nii.gz") if not p.name.startswith(".")])
    for p in m_files:
        sid = p.parent.name
        records.append({
            "subject_id": sid,
            "path": str(p),
            "class_name": "Meningioma",
            "class_id": 2,
            "channel_idx": 0
        })

    # 3. Metastasis (Task_BrainMetastasis train/*/t1n.nii.gz, channel 0)
    met_dir = BASE_DIR / "Task_BrainMetastasis" / "train"
    met_files = sorted([p for p in met_dir.glob("*/t1n.nii.gz") if not p.name.startswith(".")])
    for p in met_files:
        sid = p.parent.name
        records.append({
            "subject_id": sid,
            "path": str(p),
            "class_name": "Metastasis",
            "class_id": 3,
            "channel_idx": 0
        })

    return records


def process_single_scan(rec: Dict[str, Any]) -> Dict[str, Any]:
    sid = rec["subject_id"]
    out_npy = CACHE_DIR / f"{sid}.npy"
    p = Path(rec["path"])

    # Read original metadata
    hdr = nib.load(str(p)).header
    orig_shape = [int(x) for x in hdr.get_data_shape()]
    orig_spacing = [float(round(float(z), 3)) for z in hdr.get_zooms()[:3]]

    if out_npy.exists():
        # Already processed
        vol = np.load(str(out_npy))
        nnz = int(np.count_nonzero(vol))
        return {
            "subject_id": sid,
            "class_name": rec["class_name"],
            "class_id": rec["class_id"],
            "orig_shape": orig_shape,
            "orig_spacing": orig_spacing,
            "nonzero_voxels": nnz,
            "padding_fraction": float(nnz / (128.0 ** 3)),
            "cache_file": str(out_npy),
            "status": "CACHED"
        }

    res = preprocess_mri(p, channel_index=rec["channel_idx"])
    vol = res["processed_volume"].astype(np.float16)  # 4.19 MB per file
    mask = res["brain_mask"]

    np.save(str(out_npy), vol)
    nnz = int(np.count_nonzero(vol))
    mask_vol = int(np.sum(mask))

    return {
        "subject_id": sid,
        "class_name": rec["class_name"],
        "class_id": rec["class_id"],
        "orig_shape": orig_shape,
        "orig_spacing": orig_spacing,
        "nonzero_voxels": nnz,
        "mask_voxels": mask_vol,
        "padding_fraction": float(nnz / (128.0 ** 3)),
        "cache_file": str(out_npy),
        "status": "PROCESSED"
    }


def main():
    records = get_all_records()
    print(f"Total native T1 records to cache: {len(records)}")
    counts = {}
    for r in records:
        counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
    for c, cnt in counts.items():
        print(f"  - {c}: {cnt} subjects")

    t0 = time.time()
    manifest = []
    # Use 6 parallel workers for stable CPU utilization
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_single_scan, rec): rec for rec in records}
        done_count = 0
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            manifest.append(res)
            done_count += 1
            if done_count % 100 == 0 or done_count == len(records):
                elapsed = time.time() - t0
                print(f"  Progress: [{done_count}/{len(records)}] scans ({elapsed:.1f}s, {done_count/elapsed:.1f} scans/s)", flush=True)

    with open(CACHE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Build stratified 70/15/15 patient-level split
    rng = np.random.default_rng(42)
    splits = {"train": [], "val": [], "test": []}

    for cname in ["Glioma", "Meningioma", "Metastasis"]:
        c_items = [m for m in manifest if m["class_name"] == cname]
        rng.shuffle(c_items)
        n = len(c_items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))

        splits["train"].extend(c_items[:n_train])
        splits["val"].extend(c_items[n_train:n_train + n_val])
        splits["test"].extend(c_items[n_train + n_val:])

    # Verify zero leakage
    train_ids = set(x["subject_id"] for x in splits["train"])
    val_ids = set(x["subject_id"] for x in splits["val"])
    test_ids = set(x["subject_id"] for x in splits["test"])

    assert len(train_ids & val_ids) == 0, "Patient leakage train-val"
    assert len(train_ids & test_ids) == 0, "Patient leakage train-test"
    assert len(val_ids & test_ids) == 0, "Patient leakage val-test"

    with open(CACHE_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("\nPatient-Level Stratified 70/15/15 Split:")
    for split_name in ["train", "val", "test"]:
        sub_list = splits[split_name]
        c_cnt = {}
        for s in sub_list:
            c_cnt[s["class_name"]] = c_cnt.get(s["class_name"], 0) + 1
        print(f"  - {split_name.upper()} (Total={len(sub_list)}): {c_cnt}")


if __name__ == "__main__":
    main()
