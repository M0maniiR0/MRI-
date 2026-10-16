#!/usr/bin/env python3
"""
validate_all_cases.py
=====================
Robust read-only validation for MSD Task01_BrainTumour dataset.

Checks performed per training case:
  1. Image and label file existence
  2. NIfTI readability (corrupt-file detection)
  3. MRI shape is 4D with exactly 4 channels [X, Y, Z, 4]
  4. Positive, finite voxel spacing
  5. Finite affine matrix
  6. Spatial shape consistency: image[:3] == label.shape
  7. Affine consistency between image and label (atol=1e-3)
  8. Segmentation labels ⊆ {0, 1, 2, 3}
  9. Non-integer mask values
 10. Non-finite (NaN / Inf) voxel intensities per channel (streamed)
 11. Non-finite mask voxels
 12. Duplicate case IDs in training list
 13. Train/test case-ID overlap

Usage (run from the MRI Data folder that contains Task01_BrainTumour/):
    python validate_all_cases.py

Or pointing to a different dataset root:
    python validate_all_cases.py --data "Task01_BrainTumour/Task01_BrainTumour" --out my_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

# ──────────────────────────────────────────────────────────────
# Default dataset root (relative to this script's location)
# ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = _SCRIPT_DIR / "Task01_BrainTumour" / "Task01_BrainTumour"

VALID_LABELS = {0, 1, 2, 3}


# ──────────────────────────────────────────────────────────────
# Per-case inspector
# ──────────────────────────────────────────────────────────────
def inspect_case(
    root: Path,
    case_id: str,
    image_rel: str,
    label_rel: str | None,
) -> dict[str, Any]:
    """
    Validate a single training case.

    Parameters
    ----------
    root      : Absolute dataset root (contains imagesTr/, labelsTr/).
    case_id   : Human-readable case identifier (e.g. 'BRATS_457').
    image_rel : Path relative to root for the MRI file.
    label_rel : Path relative to root for the label/mask file.

    Returns
    -------
    dict with keys: case, passed, errors, warnings, and optional
    metadata fields (image_shape, spacing, label_shape, labels_present,
    label_counts).
    """
    result: dict[str, Any] = {
        "case": case_id,
        "passed": False,
        "errors": [],
        "warnings": [],
    }

    # ── 1. Resolve & guard image path ────────────────────────
    img_path = (root / image_rel).resolve()
    try:
        img_path.relative_to(root.resolve())
    except ValueError:
        result["errors"].append("SECURITY: image path escapes dataset root")
        return result

    if not img_path.is_file():
        result["errors"].append(f"MISSING IMAGE: {image_rel}")
        return result

    # ── 2. Load image (catches corrupt / unreadable files) ───
    try:
        img = nib.load(str(img_path))
        sh = img.shape                       # tuple, lazy header read
    except Exception as exc:
        result["errors"].append(f"CORRUPT IMAGE ({type(exc).__name__}): {exc}")
        return result

    # ── 3. Shape: must be 4-D with 4 channels ───────────────
    result["image_shape"] = list(sh)
    if len(sh) != 4:
        result["errors"].append(f"SHAPE: expected 4-D, got {len(sh)}-D {sh}")
    elif sh[3] != 4:
        result["errors"].append(f"CHANNELS: expected 4, got {sh[3]}")

    # ── 4. Voxel spacing ────────────────────────────────────
    try:
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        result["spacing_mm"] = zooms
        if any(not np.isfinite(z) or z <= 0 for z in zooms):
            result["errors"].append(f"SPACING: invalid voxel size {zooms}")
    except Exception as exc:
        result["errors"].append(f"SPACING READ ERROR: {exc}")

    # ── 5. Affine matrix ────────────────────────────────────
    if not np.all(np.isfinite(img.affine)):
        result["errors"].append("AFFINE: non-finite values in MRI affine")

    # ── 6. Intensity check per channel (streaming, avoids full 4-D RAM) ─
    if len(sh) == 4:
        for c in range(sh[3]):
            try:
                channel = np.asarray(img.dataobj[..., c])
                if not np.isfinite(channel).all():
                    n_bad = int(np.sum(~np.isfinite(channel)))
                    result["errors"].append(
                        f"NON-FINITE INTENSITY: channel {c} has {n_bad} bad voxels"
                    )
            except Exception as exc:
                result["errors"].append(
                    f"INTENSITY READ ERROR channel {c} ({type(exc).__name__}): {exc}"
                )

    # ── 7–11. Label / mask checks ───────────────────────────
    if label_rel is not None:
        mask_path = (root / label_rel).resolve()
        try:
            mask_path.relative_to(root.resolve())
        except ValueError:
            result["errors"].append("SECURITY: label path escapes dataset root")
            return result

        if not mask_path.is_file():
            result["errors"].append(f"MISSING LABEL: {label_rel}")
        else:
            try:
                msk = nib.load(str(mask_path))
                msh = msk.shape
                result["label_shape"] = list(msh)

                # 7. Spatial shape consistency
                if tuple(sh[:3]) != tuple(msh[:3]):
                    result["errors"].append(
                        f"SHAPE MISMATCH: image {sh[:3]} vs mask {msh[:3]}"
                    )

                # 8. Affine consistency
                if not np.allclose(img.affine, msk.affine, atol=1e-3, rtol=1e-5):
                    diff = float(np.max(np.abs(img.affine - msk.affine)))
                    result["errors"].append(
                        f"AFFINE MISMATCH: max delta = {diff:.6f}"
                    )

                # 9–11. Mask voxel checks
                m = np.asarray(msk.dataobj)

                if not np.isfinite(m).all():
                    result["errors"].append("NON-FINITE MASK: mask contains NaN/Inf")
                else:
                    unique_vals = np.unique(m)
                    int_vals = np.round(unique_vals).astype(np.int32)
                    result["labels_present"] = [int(v) for v in int_vals]

                    counts = {
                        str(int(k)): int(v)
                        for k, v in zip(
                            int_vals,
                            [int(np.sum(m == v)) for v in unique_vals],
                        )
                    }
                    result["label_counts"] = counts

                    # 9. Non-integer values
                    if not np.all(unique_vals == np.floor(unique_vals)):
                        result["errors"].append("MASK VALUES: non-integer label values")

                    # 10. Unexpected label values
                    bad_labels = set(int_vals.tolist()) - VALID_LABELS
                    if bad_labels:
                        result["errors"].append(
                            f"LABELS OUT OF RANGE: unexpected labels {sorted(bad_labels)}"
                        )

                    # 11. Warn if only background
                    if set(int_vals.tolist()) == {0}:
                        result["warnings"].append(
                            "MASK: only background label (0) present — no tumour?"
                        )

            except Exception as exc:
                result["errors"].append(
                    f"CORRUPT LABEL ({type(exc).__name__}): {exc}"
                )

    result["passed"] = len(result["errors"]) == 0
    return result


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate all cases in an MSD-format NIfTI dataset."
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the dataset root containing dataset.json (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "validation_report.json",
        help="Output JSON report path (default: %(default)s)",
    )
    args = ap.parse_args()

    root: Path = args.data.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] Not a directory: {root}")

    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        sys.exit(f"[ERROR] dataset.json not found in: {root}")

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    dataset_name: str = manifest.get("name", "unknown")
    training: list[dict] = manifest.get("training", [])
    testing: list = manifest.get("test", [])

    print("=" * 65)
    print(f"  Dataset      : {dataset_name}")
    print(f"  Root         : {root}")
    print(f"  Training     : {manifest.get('numTraining')} expected / {len(training)} entries")
    print(f"  Test         : {manifest.get('numTest')} expected / {len(testing)} entries")
    print(f"  Modalities   : {manifest.get('modality', {})}")
    print(f"  Labels       : {manifest.get('labels', {})}")
    print("=" * 65)

    # ── 12. Duplicate case IDs ──────────────────────────────
    train_ids: list[str] = []
    for c in training:
        rel = c["image"]
        cid = Path(rel).name.replace(".nii.gz", "").replace(".nii", "")
        train_ids.append(cid)

    id_counts = Counter(train_ids)
    duplicates = sorted(k for k, n in id_counts.items() if n > 1)

    # ── 13. Train/test overlap ──────────────────────────────
    test_ids = set(
        Path(x).name.replace(".nii.gz", "").replace(".nii", "") for x in testing
    )
    train_id_set = set(train_ids)
    overlaps = sorted(train_id_set & test_ids)

    if duplicates:
        print(f"[WARN] Duplicate training IDs ({len(duplicates)}): {duplicates}")
    if overlaps:
        print(f"[WARN] Train/Test ID overlap ({len(overlaps)}): {overlaps}")

    # ── Per-case validation loop ────────────────────────────
    results: list[dict] = []
    t0 = time.perf_counter()
    n_total = len(training)

    for i, case in enumerate(training, start=1):
        rel_img = case["image"]
        rel_lbl = case.get("label")
        case_id = Path(rel_img).name.replace(".nii.gz", "").replace(".nii", "")

        res = inspect_case(root, case_id, rel_img, rel_lbl)
        results.append(res)

        status = "PASS" if res["passed"] else "FAIL"
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i) * (n_total - i) if i > 0 else 0
        errors_str = (
            ""
            if res["passed"]
            else "  ← " + " | ".join(res["errors"][:2])
        )
        print(
            f"  [{i:>3}/{n_total}] {status}  {case_id}"
            f"  (elapsed {elapsed:5.1f}s, ETA {eta:5.1f}s){errors_str}",
            flush=True,
        )

    # ── Summary ─────────────────────────────────────────────
    n_passed = sum(r["passed"] for r in results)
    n_failed = n_total - n_passed
    n_warn   = sum(1 for r in results if r["warnings"])

    print()
    print("=" * 65)
    print("  VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  Total cases        : {n_total}")
    print(f"  [PASS] PASSED      : {n_passed}")
    print(f"  [FAIL] FAILED      : {n_failed}")
    print(f"  [WARN] w/ warnings : {n_warn}")
    print(f"  Duplicate IDs      : {len(duplicates)}")
    print(f"  Train/Test overlap : {len(overlaps)}")
    print(f"  Total time         : {time.perf_counter() - t0:.1f}s")
    print("=" * 65)

    if n_failed:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  • {r['case']}")
                for e in r["errors"]:
                    print(f"      [{e}]")

    # ── Write JSON report ────────────────────────────────────
    summary = {
        "dataset": dataset_name,
        "expected_training": manifest.get("numTraining"),
        "actual_training_entries": n_total,
        "expected_test": manifest.get("numTest"),
        "actual_test_entries": len(testing),
        "passed": n_passed,
        "failed": n_failed,
        "cases_with_warnings": n_warn,
        "training_duplicates": duplicates,
        "train_test_id_overlap": overlaps,
    }
    report = {"summary": summary, "cases": results}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Report saved -> {args.out.resolve()}")

    # Exit with non-zero code when failures exist (useful in CI)
    if n_failed or duplicates or overlaps:
        sys.exit(1)


if __name__ == "__main__":
    main()
