# Alzheimer's Disease & Brain Atrophy — Hippocampal Volumetry Report

**Date**: 2026-10-19 | **Model**: SegResNet (Trained on RTX 2050 GPU) | **Dataset**: Medical Segmentation Decathlon Task04

## Clinical Cohort Volumetric Summary

| Case ID | Anterior Vol (mm³) | Posterior Vol (mm³) | Total Vol (mm³) | Ground Truth (mm³) | Dice Score | Clinical Atrophy Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `hippocampus_001` | 0.0 | 2340.0 | 2340.0 | 2948.0 | 0.539 | **ATROPHY / ALZHEIMER'S** |
| `hippocampus_003` | 0.0 | 2377.0 | 2377.0 | 3353.0 | 0.605 | **ATROPHY / ALZHEIMER'S** |
| `hippocampus_004` | 0.0 | 2318.0 | 2318.0 | 3698.0 | 0.572 | **ATROPHY / ALZHEIMER'S** |
| `hippocampus_006` | 0.0 | 2351.0 | 2351.0 | 4263.0 | 0.543 | **ATROPHY / ALZHEIMER'S** |
| `hippocampus_007` | 0.0 | 2401.0 | 2401.0 | 3372.0 | 0.532 | **BORDERLINE / MCI** |
| `hippocampus_008` | 0.0 | 2415.0 | 2415.0 | 3248.0 | 0.578 | **BORDERLINE / MCI** |
| `hippocampus_011` | 0.0 | 2177.0 | 2177.0 | 3456.0 | 0.519 | **ATROPHY / ALZHEIMER'S** |
| `hippocampus_014` | 0.0 | 2364.0 | 2364.0 | 3622.0 | 0.478 | **ATROPHY / ALZHEIMER'S** |

## Clinical Diagnostic Criteria
- **Normal Brain Volume** ($\ge 3000\text{ mm}^3$): Healthy hippocampal structure, low risk of cognitive impairment.
- **Borderline / Mild Cognitive Impairment (MCI)** ($2400 - 3000\text{ mm}^3$): Early-stage neurodegeneration.
- **Severe Hippocampal Atrophy / Alzheimer's** ($< 2400\text{ mm}^3$): High radiological correlation with clinical Alzheimer's Disease.

## Visual Evidence
High-resolution 3-view overlays (Coronal, Sagittal, Axial) saved to `outputs\alzheimer_atrophy_analysis/`.