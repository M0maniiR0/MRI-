#!/usr/bin/env python3
"""
create_splits_and_inventory.py
==============================
Generates:
  1. dataset_inventory.csv: metadata inventory for all 484 training cases.
  2. splits/train_ids.csv, splits/val_ids.csv, splits/test_ids.csv (70% / 15% / 15%)
     with deterministic seed for zero-leakage reproducibility.
"""

from pathlib import Path
import json
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Task01_BrainTumour" / "Task01_BrainTumour"
SPLITS_DIR = ROOT / "splits"

def main():
    manifest_file = DATA_DIR / "dataset.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    training = manifest["training"]
    num_total = len(training)

    # 1. Extract case IDs and build inventory
    inventory = []
    case_ids = []
    for item in training:
        img_rel = item["image"]
        lbl_rel = item["label"]
        cid = Path(img_rel).name.replace(".nii.gz", "").replace(".nii", "")
        case_ids.append(cid)
        inventory.append({
            "case_id": cid,
            "image_path": img_rel,
            "label_path": lbl_rel,
            "channels": "FLAIR,T1w,t1gd,T2w",
            "format": "NIfTI_4D"
        })

    # Save inventory CSV
    inventory_csv = ROOT / "dataset_inventory.csv"
    with open(inventory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "image_path", "label_path", "channels", "format"])
        writer.writeheader()
        writer.writerows(inventory)
    print(f"[OK] Generated dataset_inventory.csv ({len(inventory)} cases)")

    # 2. Deterministic Train/Val/Test Split (70% / 15% / 15%)
    SEED = 42
    rng = np.random.default_rng(SEED)
    shuffled_ids = np.array(sorted(case_ids))
    rng.shuffle(shuffled_ids)

    n_train = int(round(num_total * 0.70))
    n_val = int(round(num_total * 0.15))
    n_test = num_total - n_train - n_val

    train_ids = sorted(shuffled_ids[:n_train])
    val_ids = sorted(shuffled_ids[n_train:n_train + n_val])
    test_ids = sorted(shuffled_ids[n_train + n_val:])

    # Verification: no leakage
    assert len(set(train_ids) & set(val_ids)) == 0, "Leakage between Train and Val"
    assert len(set(train_ids) & set(test_ids)) == 0, "Leakage between Train and Test"
    assert len(set(val_ids) & set(test_ids)) == 0, "Leakage between Val and Test"
    assert len(train_ids) + len(val_ids) + len(test_ids) == num_total

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    def write_split(path, ids):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id"])
            for cid in ids:
                writer.writerow([cid])

    write_split(SPLITS_DIR / "train_ids.csv", train_ids)
    write_split(SPLITS_DIR / "val_ids.csv", val_ids)
    write_split(SPLITS_DIR / "test_ids.csv", test_ids)

    print(f"[OK] Generated splits (seed={SEED}):")
    print(f"   - Train : {len(train_ids)} ({len(train_ids)/num_total*100:.1f}%) -> splits/train_ids.csv")
    print(f"   - Val   : {len(val_ids)} ({len(val_ids)/num_total*100:.1f}%) -> splits/val_ids.csv")
    print(f"   - Test  : {len(test_ids)} ({len(test_ids)/num_total*100:.1f}%) -> splits/test_ids.csv")

if __name__ == "__main__":
    main()
