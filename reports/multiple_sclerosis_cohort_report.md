# Multiple Sclerosis (التصلب المتعدد / التصلب اللويحي) — Cohort Volumetric Report

**Date**: 2026-10-24 | **Model**: SegResNet 3D (Trained on RTX 2050 GPU) | **Dataset**: MSLesSeg-2024 & WMH Challenge (240 Cases Cohort)

| Scan / Patient ID | Total Lesion Load (mm³) | Ground Truth (mm³) | Plaque Count | Dice Similarity | Clinical Stage / Burden |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `P54` | 37,816.0 | 1,332.0 | 149 | 0.066 | 🔴 **Severe Burden (>15k mm³)** |
| `P12_T2` | 62,547.0 | 32,008.0 | 61 | 0.534 | 🔴 **Severe Burden (>15k mm³)** |
| `P26_T1` | 29,586.0 | 6,639.0 | 121 | 0.293 | 🔴 **Severe Burden (>15k mm³)** |
| `P46_T1` | 33,225.0 | 6,217.0 | 102 | 0.265 | 🔴 **Severe Burden (>15k mm³)** |
| `wmh_Singapore_51` | 58,248.0 | 27,849.0 | 25 | 0.466 | 🔴 **Severe Burden (>15k mm³)** |
| `wmh_Utrecht_29` | 63,824.4 | 46,579.6 | 21 | 0.622 | 🔴 **Severe Burden (>15k mm³)** |
| `wmh_Singapore_87` | 48,156.0 | 20,274.0 | 28 | 0.518 | 🔴 **Severe Burden (>15k mm³)** |
| `wmh_Utrecht_1` | 37,928.2 | 4,446.9 | 31 | 0.073 | 🔴 **Severe Burden (>15k mm³)** |

## Diagnostic Overlays
High-resolution 3-plane views (Coronal, Sagittal, Axial) saved to `outputs\multiple_sclerosis_analysis/`.