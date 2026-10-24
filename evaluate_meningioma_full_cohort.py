#!/usr/bin/env python3
"""
evaluate_meningioma_full_cohort.py
==================================
Evaluates all 40 validation cases in BraTS-MEN 2023 across multiple channel and threshold candidates.
Permanently determines the optimal output channel and threshold for Meningioma.
"""

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import Compose, NormalizeIntensityd, EnsureTyped

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Task_Meningioma" / "BraTS_MEN"
MODEL_PATH = SCRIPT_DIR / "models" / "segresnet_meningioma_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Evaluating Meningioma on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

# 1. Load model
model = SegResNet(
    spatial_dims=3, in_channels=4, out_channels=3,
    init_filters=8, blocks_down=[1, 2, 2], blocks_up=[1, 1], dropout_prob=0.2
).to(device)
ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Get 40 validation cases
case_dirs = sorted([d for d in DATA_DIR.glob("BraTS-MEN-*") if d.is_dir()])
val_split = max(1, int(len(case_dirs) * 0.2))  # exactly 40 cases
val_cases = case_dirs[-val_split:]
print(f"[DATA] Cohort total: {len(case_dirs)} | Validation cohort: {len(val_cases)} cases", flush=True)

norm_pipe = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys="image")
])

def compute_dice(pred, gt):
    p_b = pred > 0
    g_b = gt > 0
    inter = (p_b & g_b).sum()
    card = p_b.sum() + g_b.sum()
    if card == 0:
        return 1.0
    return float((2.0 * inter) / (card + 1e-8))

candidates = {
    "Ch0_th0.50": {"ch": 0, "th": 0.50},
    "Ch0_th0.80": {"ch": 0, "th": 0.80},
    "Ch0_th0.90": {"ch": 0, "th": 0.90},
    "Ch1_th0.20": {"ch": 1, "th": 0.20},
    "Ch1_th0.30": {"ch": 1, "th": 0.30},
    "Ch1_th0.40": {"ch": 1, "th": 0.40},
    "Ch1_th0.50": {"ch": 1, "th": 0.50},
    "Ch2_th0.30": {"ch": 2, "th": 0.30},
}

scores = {k: [] for k in candidates}
per_case_records = []

t_start = time.time()
print("\nRunning inference across all 40 validation cases with corrected channel order [t2f, t1n, t1c, t2w]...", flush=True)

for idx, c in enumerate(val_cases, 1):
    cid = c.name
    t1n_p = c / f"{cid}-t1n.nii.gz"
    t1c_p = c / f"{cid}-t1c.nii.gz"
    t2f_p = c / f"{cid}-t2f.nii.gz"
    t2w_p = c / f"{cid}-t2w.nii.gz"
    seg_p = c / f"{cid}-seg.nii.gz"

    if not (t1n_p.is_file() and t1c_p.is_file() and t2f_p.is_file() and t2w_p.is_file() and seg_p.is_file()):
        continue

    t1n = nib.load(str(t1n_p)).get_fdata(dtype=np.float32)
    t1c = nib.load(str(t1c_p)).get_fdata(dtype=np.float32)
    t2f = nib.load(str(t2f_p)).get_fdata(dtype=np.float32)
    t2w = nib.load(str(t2w_p)).get_fdata(dtype=np.float32)
    gt = (nib.load(str(seg_p)).get_fdata() > 0).astype(np.uint8)
    gt_voxels = int(gt.sum())

    # Permanent corrected order: [FLAIR, T1, T1-CE, T2]
    img_4d = np.stack([t2f, t1n, t1c, t2w], axis=0)

    inp = norm_pipe({"image": torch.from_numpy(img_4d)})["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = sliding_window_inference(inp, (96, 96, 96), 1, model, overlap=0.25)
        prob = torch.sigmoid(out).squeeze(0)
        bm = (inp.squeeze(0).abs().sum(dim=0) > 0)

    case_row = {"case_id": cid, "gt_voxels": gt_voxels}
    for c_name, c_cfg in candidates.items():
        ch_idx = c_cfg["ch"]
        th_val = c_cfg["th"]
        pred = ((prob[ch_idx] > th_val) & bm).cpu().numpy().astype(np.uint8)
        d = compute_dice(pred, gt)
        scores[c_name].append(d)
        case_row[c_name] = round(d, 4)
        case_row[f"{c_name}_voxels"] = int(pred.sum())

    per_case_records.append(case_row)
    if idx % 5 == 0 or idx == len(val_cases):
        print(f"  Processed [{idx:>2}/{len(val_cases)}] cases | {cid} GT={gt_voxels} | Ch1_th0.40 Dice={case_row['Ch1_th0.40']:.4f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Processed [{idx:>2}/{len(val_cases)}] cases | {cid} GT={gt_voxels} | Ch1_th0.40 Dice={case_row['Ch1_th0.40']:.4f}")

elapsed = time.time() - t_start
print(f"\n[DONE] 40 cases evaluated in {elapsed:.1f}s ({elapsed/len(val_cases):.2f}s per case).")

summary = {}
print("\n" + "=" * 70)
print(f"{'Candidate':<15} | {'Mean Dice':<10} | {'Median':<8} | {'Std Dev':<8} | {'Failures (<0.1)':<15}")
print("=" * 70)
for k, v in scores.items():
    arr = np.array(v)
    mean_val = float(np.mean(arr))
    med_val = float(np.median(arr))
    std_val = float(np.std(arr))
    fails = int(np.sum(arr < 0.1))
    summary[k] = {
        "mean_dice": round(mean_val, 4),
        "median_dice": round(med_val, 4),
        "std_dice": round(std_val, 4),
        "min_dice": round(float(np.min(arr)), 4),
        "max_dice": round(float(np.max(arr)), 4),
        "failures_count": fails,
        "failure_rate_percent": round((fails / len(arr)) * 100, 1),
    }
    print(f"{k:<15} | {mean_val:<10.4f} | {med_val:<8.4f} | {std_val:<8.4f} | {fails:>2}/{len(arr)} ({fails/len(arr)*100:.1f}%)")
print("=" * 70)

# Save results
out_data = {
    "experiment": "Meningioma Full 40-case Validation Cohort Parameter Sweep",
    "cohort_size": len(case_dirs),
    "val_cases_count": len(val_cases),
    "channel_order": ["t2f", "t1n", "t1c", "t2w"],
    "summary": summary,
    "per_case_results": per_case_records,
}

out_file = SCRIPT_DIR / "reports" / "meningioma_full_40cases_sweep.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
print(f"\n[EXPORT] Full 40-case sweep saved -> {out_file.name}")
