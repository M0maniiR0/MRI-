# Brain Metastasis (الانتقالات الدماغية) — Cohort Volumetric Report

**Date**: 2026-10-19 | **Model**: SegResNet 3D (Trained on RTX 2050 GPU) | **Dataset**: Brain Metastases Dataset

| Patient ID | Metastatic Volume (mm³) | Ground Truth (mm³) | Foci Count | Dice Similarity | Clinical Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `P30` | 4,973.0 | 2,903.0 | 55 | 0.000 | **Multiple Foci (55)** |
| `P31` | 69,793.0 | 28,136.0 | 24 | 0.000 | **Multiple Foci (24)** |
| `P32` | 7,752.0 | 2,217.0 | 43 | 0.000 | **Multiple Foci (43)** |
| `P33` | 16,280.0 | 35,224.0 | 33 | 0.023 | **Multiple Foci (33)** |
| `P34` | 41,508.0 | 12,184.0 | 32 | 0.000 | **Multiple Foci (32)** |
| `P35` | 10,855.0 | 9,402.0 | 30 | 0.000 | **Multiple Foci (30)** |

## Diagnostic Overlays
High-resolution 3-plane views (Coronal, Sagittal, Axial) saved to `outputs\brain_metastasis_analysis/`.