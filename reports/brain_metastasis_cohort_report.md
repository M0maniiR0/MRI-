# Brain Metastasis (الانتقالات الدماغية) — Cohort Volumetric Report

**Date**: 2026-10-19 | **Model**: SegResNet 3D (Trained on RTX 2050 GPU) | **Dataset**: Brain Metastases Dataset

| Patient ID | Metastatic Volume (mm³) | Ground Truth (mm³) | Foci Count | Dice Similarity | Clinical Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sci_p001` | 15,337.0 | 3,927.0 | 16 | 0.110 | **Multiple Foci (16)** |
| `sci_p031` | 96,464.0 | 71,018.0 | 11 | 0.541 | **Multiple Foci (11)** |
| `sci_p061` | 126,320.0 | 143,894.0 | 24 | 0.362 | **Multiple Foci (24)** |
| `sci_p091` | 202,447.0 | 281,217.0 | 11 | 0.722 | **Multiple Foci (11)** |
| `sci_p121` | 68,984.0 | 24,409.0 | 16 | 0.369 | **Multiple Foci (16)** |
| `sci_p151` | 138,313.0 | 119,724.0 | 10 | 0.352 | **Multiple Foci (10)** |
| `sci_p181` | 46,427.0 | 389.0 | 19 | 0.000 | **Multiple Foci (19)** |
| `sci_p040` | 131,067.0 | 159,691.0 | 8 | 0.527 | **Multiple Foci (8)** |

## Diagnostic Overlays
High-resolution 3-plane views (Coronal, Sagittal, Axial) saved to `outputs\brain_metastasis_analysis/`.