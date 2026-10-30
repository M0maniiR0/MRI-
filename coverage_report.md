# 📋 Task C Coverage Report (ACIT26 BrainAgenticthon)

This report establishes the verified data coverage across all **10 official diagnostic classes** defined for Track 2 (Task C).
All counts are strictly programmatically verified by directory audits and manifest parsing. No numbers are guessed or assumed.

---

## 1. Official 10-Class Coverage Summary

| # | Official Class Name | Local Source Dataset | Sequence / Channel | Number of Subjects | Status / Gap Flag | Notes & Contract Compliance |
|---|---------------------|----------------------|:------------------:|:------------------:|-------------------|-----------------------------|
| 1 | **Normal** | None (ADNI/OASIS/ABIDE/PPMI not local) | N/A | **0** | ⚠️ **Coverage Gap** | Rule strictly enforced: absence of label is NOT "healthy". Explicitly flagged. |
| 2 | **Glioma** | `Task01_BrainTumour/imagesTr` (MSD) | Native T1 (ch 1) | **484** | ✅ **Covered (Native T1)** | Whole-brain 3D native T1 scans. 4D multi-channel split, channel 1 extracted. |
| 3 | **Meningioma** | `Task_Meningioma/BraTS_MEN` (BraTS 2023) | Native T1 (`t1n`) | **200** | ✅ **Covered (Native T1)** | Whole-brain 3D native T1 (non-contrast `t1n`). $T_1\text{c}$ strictly removed. |
| 4 | **Metastasis** | `Task_BrainMetastasis/train` (BraTS-METS) | Native T1 (`t1n`) | **200** | ✅ **Covered (Native T1)** | Whole-brain 3D native T1 (non-contrast `t1n`). $T_1\text{c}$ strictly removed. |
| 5 | **Alzheimer's** | None (ADNI/OASIS whole-brain not found) | N/A | **0** | ⚠️ **Coverage Gap** | MSD Task04 Hippocampus removed (cropped $\sim 40^3$, no diagnosis). Documented as gap. |
| 6 | **FCD epilepsy** | None | N/A | **0** | ⚠️ **Expected Coverage Gap** | No local cohort available. Routing directly to `Uncertain/Other`. |
| 7 | **Post-stroke lesion** | None | N/A | **0** | ⚠️ **Expected Coverage Gap** | No local cohort available. Routing directly to `Uncertain/Other`. |
| 8 | **Parkinson's** | None (PPMI not found locally) | N/A | **0** | ⚠️ **Coverage Gap** | No local cohort available. Documented as coverage gap; cannot invent data or labels. |
| 9 | **ASD** | None (ABIDE not found locally) | N/A | **0** | ⚠️ **Coverage Gap** | No local cohort available. Documented as coverage gap; cannot invent data or labels. |
| 10 | **Uncertain/Other** | Internal Out-of-Distribution & Uncovered Classes | N/A | **N/A** | 🛡️ **System Default Fallback** | Handles coverage gaps, low confidence, and excluded classes (e.g. Multiple Sclerosis). |

**Total Available Valid Subjects for Task C Training**: $N = 884$ unique biological subjects.

---

## 2. Excluded / Unofficial Cohorts Audit

| Cohort | Local Source Path | Number of Subjects | Reason for Exclusion / Routing |
|---|---|:---:|---|
| **Multiple Sclerosis** | `Task_MultipleSclerosis` | **240** | Not in official 10 classes. Excluded from training; routed to `Uncertain/Other`. |
| **MSD Task04 Hippocampus** | `Task04_Hippocampus` / `DATA/Alzheimer_Hippocampus` | **390** | Cropped bounding boxes ($\sim 36 \times 48 \times 40$ voxels), lacks whole-brain anatomy and diagnostic labels. Removed to eliminate non-biological shortcut leakage. |

---

## 3. Strict Task C Rules Verification

1. **Zero Guessing & Honest Coverage:** Unsupported classes (`Normal`, `Alzheimer's`, `FCD epilepsy`, `Post-stroke lesion`, `Parkinson's`, `ASD`) are explicitly reported as coverage gaps. No synthetic or invented labels were created.
2. **Native T1 Uniformity:** Contrast-enhanced sequences ($T_1\text{c}$) have been completely removed across all cohorts. Training exclusively uses non-contrast Native T1 (`t1n` / channel 1).
3. **Subject-Level Accounting:** All figures represent unique biological subjects/patients, not repeated MRI scan files.
4. **Offline Compliance:** Every path is verified to reside locally on physical storage under `C:\Users\HP VICTUS\OneDrive\Desktop\MRI Data`.
