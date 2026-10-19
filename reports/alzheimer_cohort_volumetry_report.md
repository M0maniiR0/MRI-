# Alzheimer's Disease & Brain Atrophy — Hippocampal Volumetry Report

**Date**: 2026-10-19 | **Model**: SegResNet (Trained on RTX 2050 GPU) | **Dataset**: Medical Segmentation Decathlon Task04

## Clinical Cohort Volumetric Summary

| Case ID | Anterior Vol (mm³) | Posterior Vol (mm³) | Total Vol (mm³) | Ground Truth (mm³) | Dice Score | Clinical Atrophy Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `hippocampus_001` | 1726.0 | 1889.0 | 3615.0 | 2948.0 | 0.837 | **NORMAL** |
| `hippocampus_003` | 1974.0 | 1985.0 | 3959.0 | 3353.0 | 0.880 | **NORMAL** |
| `hippocampus_004` | 2165.0 | 1919.0 | 4084.0 | 3698.0 | 0.891 | **NORMAL** |
| `hippocampus_006` | 2369.0 | 2045.0 | 4414.0 | 4263.0 | 0.917 | **NORMAL** |
| `hippocampus_007` | 1999.0 | 1811.0 | 3810.0 | 3372.0 | 0.845 | **NORMAL** |
| `hippocampus_008` | 1955.0 | 1832.0 | 3787.0 | 3248.0 | 0.895 | **NORMAL** |
| `hippocampus_011` | 2068.0 | 1863.0 | 3931.0 | 3456.0 | 0.806 | **NORMAL** |
| `hippocampus_014` | 2220.0 | 1915.0 | 4135.0 | 3622.0 | 0.864 | **NORMAL** |

## Clinical Diagnostic Criteria
- **Normal Brain Volume** ($\ge 3000\text{ mm}^3$): Healthy hippocampal structure, low risk of cognitive impairment.
- **Borderline / Mild Cognitive Impairment (MCI)** ($2400 - 3000\text{ mm}^3$): Early-stage neurodegeneration.
- **Severe Hippocampal Atrophy / Alzheimer's** ($< 2400\text{ mm}^3$): High radiological correlation with clinical Alzheimer's Disease.

## Visual Evidence
High-resolution 3-view overlays (Coronal, Sagittal, Axial) saved to `outputs\alzheimer_atrophy_analysis/`.