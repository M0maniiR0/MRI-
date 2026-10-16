#!/usr/bin/env python3
"""
convert_msd_to_nnunet_v2.py
===========================
Converts MSD Task01_BrainTumour 4D NIfTI format into nnU-Net v2 compliant dataset:
  - Destination structure: nnUNet_raw/Dataset001_BrainTumour/
  - Modality channel mapping:
      _0000: FLAIR
      _0001: T1w
      _0002: t1gd
      _0003: T2w
  - Preserves exact spatial headers, affines, and voxel zooms.
  - Supports converting a subset of cases (--max-cases or --split-file) to conserve disk space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MSD_ROOT = _SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"
DEFAULT_TARGET_DIR = _SCRIPT_DIR / "nnUNet_raw" / "Dataset001_BrainTumour"

CHANNEL_NAMES = {
    "0": "FLAIR",
    "1": "T1w",
    "2": "t1gd",
    "3": "T2w",
}

LABELS = {
    "background": 0,
    "edema": 1,
    "non_enhancing_tumor": 2,
    "enhancing_tumour": 3,
}


def create_nnunet_dataset_json(
    output_file: Path,
    num_training: int,
    description: str = "MSD Task01_BrainTumour converted for nnU-Net v2",
) -> None:
    meta = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "Dataset001_BrainTumour",
        "description": description,
        "reference": "BrainAgenticthon 2026 - Medical AI Workstream",
        "licence": "CC-BY-SA 4.0",
        "release": "2.0",
        "regions_class_order": [1, 2, 3],
        "derived_regions": {
            "Whole_Tumor_WT": [1, 2, 3],
            "Tumor_Core_TC": [2, 3],
            "Enhancing_Tumor_ET": [3],
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Created nnU-Net v2 dataset.json -> {output_file.name}")


def convert_case(
    case_id: str,
    img_4d_path: Path,
    lbl_path: Path | None,
    out_images_dir: Path,
    out_labels_dir: Path | None,
) -> None:
    """
    Splits a single 4D MRI [X, Y, Z, 4] into 4 separate 3D files:
      {case_id}_0000.nii.gz (FLAIR)
      {case_id}_0001.nii.gz (T1w)
      {case_id}_0002.nii.gz (t1gd)
      {case_id}_0003.nii.gz (T2w)
    And copies the label file to {case_id}.nii.gz in labelsTr.
    """
    img_obj = nib.load(str(img_4d_path))
    affine = img_obj.affine
    header = img_obj.header.copy()
    zooms_3d = header.get_zooms()[:3]
    header.set_zooms(zooms_3d)

    # Convert each channel
    for c in range(4):
        ch_data = np.asarray(img_4d_path if False else img_obj.dataobj[..., c])
        ch_nii = nib.Nifti1Image(ch_data, affine, header)
        out_ch_path = out_images_dir / f"{case_id}_{c:04d}.nii.gz"
        nib.save(ch_nii, str(out_ch_path))

    # Copy/Save label if provided
    if lbl_path is not None and out_labels_dir is not None:
        out_lbl_path = out_labels_dir / f"{case_id}.nii.gz"
        shutil.copyfile(str(lbl_path), str(out_lbl_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MSD Task01 to nnU-Net v2 format")
    ap.add_argument("--msd-dir", type=Path, default=DEFAULT_MSD_ROOT, help="MSD Task01 root")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Target nnU-Net raw dir")
    ap.add_argument("--split-file", type=Path, default=None, help="Optional CSV file of case IDs to convert")
    ap.add_argument("--max-cases", type=int, default=None, help="Convert only first N cases")
    ap.add_argument("--dry-run", action="store_true", help="Only generate dataset.json and print plan")
    args = ap.parse_args()

    msd_root: Path = args.msd_dir.resolve()
    target_root: Path = args.out_dir.resolve()

    manifest_path = msd_root / "dataset.json"
    if not manifest_path.is_file():
        sys.exit(f"[ERROR] dataset.json not found in {msd_root}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    training_entries = manifest["training"]

    # Filter by split file if provided
    selected_ids: List[str] | None = None
    if args.split_file is not None and args.split_file.is_file():
        import csv
        with open(args.split_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            selected_ids = [row[0].strip() for row in reader if row and row[0].strip() != "case_id"]
        print(f"[INFO] Filtered by split file: {len(selected_ids)} cases")

    cases_to_process = []
    for item in training_entries:
        rel_img = item["image"]
        rel_lbl = item["label"]
        cid = Path(rel_img).name.replace(".nii.gz", "").replace(".nii", "")
        if selected_ids is not None and cid not in selected_ids:
            continue
        cases_to_process.append((cid, msd_root / rel_img, msd_root / rel_lbl))

    if args.max_cases is not None:
        cases_to_process = cases_to_process[:args.max_cases]

    print("=" * 65)
    print("  MSD to nnU-Net v2 Dataset Converter")
    print("=" * 65)
    print(f"  Source MSD   : {msd_root}")
    print(f"  Target Dir   : {target_root}")
    print(f"  Cases to convert : {len(cases_to_process)} / {len(training_entries)}")
    print(f"  Dry run      : {args.dry_run}")
    print("=" * 65)

    imagesTr = target_root / "imagesTr"
    labelsTr = target_root / "labelsTr"
    imagesTs = target_root / "imagesTs"

    if not args.dry_run:
        imagesTr.mkdir(parents=True, exist_ok=True)
        labelsTr.mkdir(parents=True, exist_ok=True)
        imagesTs.mkdir(parents=True, exist_ok=True)

    # Always generate dataset.json
    create_nnunet_dataset_json(target_root / "dataset.json", num_training=len(training_entries))

    if args.dry_run:
        print("[OK] Dry run complete. No volume data written.")
        return

    t0 = time.perf_counter()
    n_total = len(cases_to_process)
    for i, (cid, img_path, lbl_path) in enumerate(cases_to_process, start=1):
        convert_case(cid, img_path, lbl_path, imagesTr, labelsTr)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i) * (n_total - i)
        print(f"  [{i:>3}/{n_total}] Converted {cid} (elapsed {elapsed:5.1f}s, ETA {eta:5.1f}s)")

    print("=" * 65)
    print(f"[SUCCESS] Converted {n_total} cases to nnU-Net v2 format in {time.perf_counter() - t0:.1f}s")
    print(f"Location: {target_root}")


if __name__ == "__main__":
    main()
