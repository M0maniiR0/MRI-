# 🧠 NeuroScope AI — Track 2: Task C
## Multi-Class 3D Brain MRI Classification Engine (Native T1 Non-Contrast)
**ACIT26 BrainAgenticthon 2026**  
**Track 2 Lead:** Roa'a (Task C Owner: Multi-Class Classification, Regional Analysis, Non-Lesional Diagnostics)  
**Teammates:** Abdulrahman (Tasks A & B: Anomaly Detection / Spatial Masking), Fadwa (Task D: Report Generation & Agentic Reasoning), Rand (Backend & Docker Deployment)

---

## 📌 Overview & Official Competition Contract

NeuroScope AI Track 2 (Task C) receives an unseen 3D brain MRI volume (`.nii` or `.nii.gz`) in an **offline, air-gapped Docker container** and performs calibrated, multi-class diagnostic probability estimation across all **10 official diagnostic classes**:

1. `Normal` *(Documented Coverage Gap)*
2. `Glioma` *(Covered: MSD Task01 Channel 1 T1w)*
3. `Meningioma` *(Covered: BraTS-MEN 2023 t1n)*
4. `Metastasis` *(Covered: BraTS-METS / SCI t1n)*
5. `Alzheimer's` *(Documented Coverage Gap)*
6. `FCD epilepsy` *(Documented Coverage Gap)*
7. `Post-stroke lesion` *(Documented Coverage Gap)*
8. `Parkinson's` *(Documented Coverage Gap)*
9. `ASD` *(Documented Coverage Gap)*
10. `Uncertain/Other` *(Safety Routing Buffer / Fallback)*

### Contract Guarantees:
* **Output Contract (`classification.json`):** Probabilities for all 10 official classes summing strictly to $1.0000 \pm 10^{-4}$.
* **100% Offline Execution:** Zero external network calls or web dependencies during inference.
* **MRI-Only Input:** No dependence on clinical metadata (age, sex, clinical scores) at inference.
* **Native T1 Uniformity:** To eliminate contrast-enhancement artifacts, only non-contrast Native T1 sequences (`t1n` / channel 1 $T_1\text{w}$) are utilized.

---

## 🏗️ Architecture & Preprocessing Pipeline

1. **Preprocessing (`preprocess.py`):**
   * Format verification & reorientation to **RAS+**.
   * Resampling to **$1.0\text{ mm}$ isotropic** spacing (bilinear/spline).
   * Brain extraction & skull-stripping via consistent intensity thresholding and morphological cleanup.
   * Intensity normalization: clipping between 0.5% and 99.5% within brain tissue, followed by non-zero z-score normalization.
   * Standard output shape: **$128 \times 128 \times 128$ voxels**.
   * Automatic Quality Control (QC): Detection of NaNs, infinities, empty brain masks, or non-3D dimensionality. Failures route to `Uncertain/Other` with `qc_status: FAIL`.

2. **Classification Architecture (`best_model.pt`):**
   * **3D DenseNet-121** (MONAI implementation, single-channel 3D input, 10 output logits).
   * Class-weighted Focal Loss ($\gamma = 2.0$) to counteract sample imbalance.
   * Cosine Annealing learning rate schedule ($10^{-4} \rightarrow 10^{-6}$).

3. **Probability Calibration & Abstain Policy (`calibration_params.json`):**
   * Temperature scaling parameter: $\hat{T} = 0.4000$ (fitted exclusively on validation split $n=133$).
   * Abstain rule: If $\max_c P(y=c|x) < 0.900$, case is routed to `Uncertain/Other`.

---

## 📊 Benchmark Test Metrics ($n=132$ Held-Out Patients)

Evaluated on held-out test split with $B=1,000$ bootstrap empirical 95% Confidence Intervals:

* **Balanced Accuracy:** **$97.78\%$** (95% CI: **$[94.44\%, 100.00\%]$**)
* **Macro-F1 Score:** **$98.39\%$** (95% CI: **$[95.74\%, 100.00\%]$**)
* **Expected Calibration Error (ECE):**
  * Before Calibration: $0.0883$
  * After Temperature Scaling ($T=0.4000$): **$0.0228$** ($74.2\%$ reduction)
* **Safety Abstain Rate:** $9/132$ scans ($6.8\%$) routed to `Uncertain/Other`.

### Per-Class Test Performance:
* **Glioma ($n=72$):** Recall = $100.00\%$, Precision = $97.30\%$, AUROC = $0.9993$ (95% CI: $[0.9966, 1.0000]$)
* **Meningioma ($n=30$):** Recall = $93.33\%$, Precision = $100.00\%$, AUROC = $0.9987$ (95% CI: $[0.9940, 1.0000]$)
* **Metastasis ($n=30$):** Recall = $100.00\%$, Precision = $100.00\%$, AUROC = $1.0000$ (95% CI: $[1.0000, 1.0000]$)

---

## 🔍 Leakage & Bias Audits

1. **Patient-Level Disjoint Split:**
   * $N=884$ unique biological patients partitioned into 70/15/15: Train ($n=619$), Val ($n=133$), Test ($n=132$).
   * Verified: $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$.
2. **Geometric Shortcut Audit:**
   * Ridge classifier trained solely on metadata (original dimensions, spacing, brain volume, padding fraction) achieved **$32.87\%$** Balanced Accuracy (vs chance $33.33\%$), verifying zero geometric leakage.
3. **Source Recognition Audit:**
   * Predictability from pure intensity-histogram and noise-texture features: **$83.89\%$** (95% CI: $[77.11\%, 90.22\%]$), documenting the acquisition protocol fingerprint (site confounding risk) that motivates our abstain policy.

---

## 📁 Repository Directory Structure

```text
MRI Data/
│
├── best_model.pt                       ← Trained 3D DenseNet-121 weights (Val Bal Acc: 96.67%)
├── calibration_params.json             ← Temperature T=0.4000 and abstain threshold 0.900
├── preprocessing_config.yaml           ← Preprocessing parameters (1.0mm isotropic, 128³)
├── label_map.json                      ← Official 10-class ID and source mapping
│
├── preprocess.py                       ← Unified 3D preprocessing & QC engine
├── inference.py                        ← Offline Task C inference entrypoint (classification.json)
├── train_native_t1_classifier.py       ← 3D DenseNet-121 training script
├── calibrate_native_t1.py              ← Temperature scaling calibration script
├── evaluate_native_t1_bootstrap.py     ← Bootstrap test evaluation script (B=1000)
├── evaluate_with_abstention.py         ← Abstention & selective evaluation script
├── run_shortcut_diagnosis_native_t1.py ← Geometric shortcut audit
├── run_source_recognition_check.py     ← Intensity/noise source recognition audit
├── cache_dataset.py                    ← Dataset preprocessing & caching script
│
├── tests/
│   └── test_task7_contract.py          ← Unit & contract test suite for Task 7
│
├── reports/
│   ├── task_c_evaluation_report.md     ← Detailed medical & statistical evaluation report
│   ├── calibration_curve.png           ← Reliability diagrams (Before vs After)
│   ├── qc_preproc_visualizations/      ← Visual inspection montages
│   ├── abstention_evaluation_results.json ← Abstention and selective classification results
│   ├── coverage_report.md              ← Cohort inventory & documented coverage gaps
│   ├── data_registry.csv               ← Comprehensive data registry & licensing
│   ├── test_evaluation_results.json    ← Raw test evaluation metrics
│   ├── source_recognition_results.json ← Raw source recognition check results
│   └── training_log.csv                ← Training & validation curve history
```

---

## 🚀 Usage & Execution Instructions

### 1. Running Offline Inference on an Unseen MRI:
```bash
python inference.py <path/to/scan.nii.gz> --output <path/to/classification.json>
```

### 2. Running Contract Tests:
```bash
python -m unittest tests/test_task7_contract.py
```

### 3. Re-evaluating on Held-Out Test Set with Bootstrap 95% CIs:
```bash
python evaluate_native_t1_bootstrap.py
```

### 4. Evaluating with Abstention & Deployment-View Metrics:
```bash
python evaluate_with_abstention.py
```
