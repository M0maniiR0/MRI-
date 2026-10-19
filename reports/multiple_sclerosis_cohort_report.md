# Multiple Sclerosis (التصلب المتعدد / التصلب اللويحي) — Cohort Volumetric Report

**Date**: 2026-10-19 | **Model**: SegResNet 3D (Trained on RTX 2050 GPU) | **Dataset**: MSLesSeg-2024 Benchmark Dataset

| Scan / Patient ID | Total Lesion Load (mm³) | Ground Truth (mm³) | Plaque Count | Dice Similarity | Clinical Stage / Burden |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `P54` | 14,972.0 | 1,332.0 | 60 | 0.156 | 🟡 **Moderate Burden (5k-15k mm³)** |
| `P68` | 23,716.0 | 919.0 | 64 | 0.044 | 🔴 **Severe Burden (>15k mm³)** |
| `P11_T2` | 14,791.0 | 3,215.0 | 84 | 0.335 | 🟡 **Moderate Burden (5k-15k mm³)** |
| `P18_T1` | 20,612.0 | 770.0 | 79 | 0.056 | 🔴 **Severe Burden (>15k mm³)** |
| `P22_T2` | 18,350.0 | 2,857.0 | 64 | 0.209 | 🔴 **Severe Burden (>15k mm³)** |
| `P31_T1` | 15,832.0 | 4,293.0 | 69 | 0.377 | 🔴 **Severe Burden (>15k mm³)** |
| `P40_T1` | 19,343.0 | 6,806.0 | 55 | 0.473 | 🔴 **Severe Burden (>15k mm³)** |
| `P50_T2` | 47,064.0 | 36,435.0 | 45 | 0.787 | 🔴 **Severe Burden (>15k mm³)** |

## Diagnostic Overlays
High-resolution 3-plane views (Coronal, Sagittal, Axial) saved to `outputs\multiple_sclerosis_analysis/`.