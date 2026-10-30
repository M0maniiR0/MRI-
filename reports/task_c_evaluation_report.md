# 📊 NeuroScope AI — Task C Rigorous Evaluation Report
## Multi-Class 3D Brain MRI Classification (Native T1 Non-Contrast)
**ACIT26 BrainAgenticthon — Track 2 Baseline Verification**  
**Author:** Roa'a (Track 2 Lead) | **Reviewed by:** Senior Medical-Imaging ML Engineer  
**Date:** October 29, 2026 | **Environment:** 100% Offline PyTorch/MONAI Container  

---

## 1. Executive Summary & Verification Criteria

This report presents the objective evaluation of the **Task C 3D Multi-Class Classification Pipeline** trained across all available patient cohorts using strictly **Native T1 (non-contrast)** whole-brain MRI volumes.

### Strict Methodological Commitments:
1. **Sequence Standardization:** Contrast-enhanced scans ($T_1\text{c}$) have been completely eliminated. All classes utilize native $T_1\text{w}$ / $T_1\text{n}$ volumes to prevent contrast agent presence from acting as an artificial shortcut.
2. **Elimination of Geometric Shortcuts:** Cropped regions of interest (specifically MSD Task04 Hippocampus) were removed. All cohorts now share identical whole-brain geometry ($240 \times 240 \times 155$, $1.0\text{ mm}$ isotropic).
3. **Shortcut Diagnosis Verification:** A linear classifier trained solely on non-biological metadata (original shape, spacing, brain-mask volume, padding fraction) achieved **32.87% Balanced Accuracy** (95% CI: $[31.82\%, 33.33\%]$) against the theoretical chance baseline of $33.33\%$.
4. **Honest Coverage Accounting:** Six unsupported official classes (`Normal`, `Alzheimer's`, `FCD epilepsy`, `Post-stroke lesion`, `Parkinson's`, `ASD`) have zero valid whole-brain local data and are documented as coverage gaps routed to `Uncertain/Other`. No labels or synthetic data were invented.
5. **Statistical Rigor:** All test metrics are reported with empirical 95% Confidence Intervals derived from $B=1,000$ bootstrap resamples on the held-out test cohort ($n=132$).

---

## 2. Cohort Definition & Patient-Level Split

The dataset comprises $N=884$ unique biological subjects with whole-brain Native T1 scans, partitioned into a disjoint $70\% / 15\% / 15\%$ patient-level split. Scans from the same patient are strictly confined to a single split ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

| Official Class Name | Local Source Cohort | Native T1 Sequence | Train ($n=619$) | Val ($n=133$) | Held-Out Test ($n=132$) | Total Subjects ($N=884$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | MSD Task01_BrainTumour | Channel 1 ($T_1\text{w}$) | 339 | 73 | 72 | **484** |
| **Meningioma** | BraTS-MEN 2023 | `*-t1n.nii.gz` | 140 | 30 | 30 | **200** |
| **Metastasis** | BraTS-METS / SCI | `t1n.nii.gz` | 140 | 30 | 30 | **200** |
| **Normal** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **Alzheimer's** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **FCD epilepsy** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **Post-stroke lesion** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **Parkinson's** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **ASD** | N/A | N/A | 0 | 0 | 0 | **0 (Coverage Gap)** |
| **Uncertain/Other** | Safety Routing Fallback | N/A | 0 | 0 | 0 | **Fallback Buffer** |

---

## 3. Shortcut Diagnosis Audit (Geometry)

To verify whether geometric metadata predicts diagnosis:
* **Features Extracted ($n=751$ train+test):** `orig_shape_d`, `orig_shape_h`, `orig_shape_w`, `spacing_x`, `spacing_y`, `spacing_z`, `padding_fraction`, `mask_voxels`.
* **Standard Deviation across cohorts:** Dimensions and spacing variance were identically $0.0000$ across all 884 volumes.
* **Ridge Classifier Test Balanced Accuracy:** **$32.87\%$**
* **Bootstrap 95% Confidence Interval ($B=1,000$):** **$[31.82\%, 33.33\%]$**
* **Criterion:** Accuracy must be within $\pm 2.0\%$ of theoretical chance level ($33.33\%$).
* **Outcome:** Criterion met ($|32.87\% - 33.33\%| = 0.46\%$, consistent with theoretical chance baseline).

---

## 4. Source-Recognition Check (Intensity & Noise)

To evaluate whether non-biological intensity distributions or scanner characteristics differentiate the cohorts, a multinomial logistic regression model was trained using ONLY intensity-histogram and high-frequency noise texture features (40 features; zero geometric, shape, or volume information).
* **Target:** Predicting acquisition source dataset (`MSD Task01`, `BraTS-MEN`, `Brain-Mets`).
* **Evaluation:** Disjoint patient-level test split ($n=132$).
* **Balanced Accuracy:** **$83.89\%$** (Bootstrap 95% CI: **$[77.11\%, 90.22\%]$**) vs. theoretical chance baseline of **$33.33\%$**.
* **Limitation:** In the current available training data, each diagnostic class originates exclusively from a single consortium/source (Glioma from MSD, Meningioma from BraTS-MEN, Metastasis from Brain-Mets). Consequently, biological class and acquisition source cannot be separated, and the model may partially rely on scanner/protocol-specific intensity and noise signatures rather than purely lesion pathology.

---

## 5. Held-Out Test Evaluation Metrics ($n=132$)

The trained 3D DenseNet-121 model was evaluated on the independent test cohort ($n=132$ subjects, unseen during architecture tuning, training, or calibration).

### Overall Benchmark Metrics:
* **Balanced Accuracy:** **$97.78\%$** (Bootstrap 95% CI: **$[94.44\%, 100.00\%]$**)
* **Macro-F1 Score:** **$98.39\%$** (Bootstrap 95% CI: **$[95.74\%, 100.00\%]$**)
* **Expected Calibration Error (ECE):**
  * Before Temperature Scaling: **$0.0883$**
  * After Temperature Scaling ($T=0.4000$): **$0.0228$** (Relative ECE reduction: $74.2\%$)

### Per-Class Performance Breakdown:

| Diagnostic Class | Test Support ($n$) | Recall / Sensitivity (95% CI) | Precision (95% CI) | F1-Score | One-vs-Rest AUROC (95% CI) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 72 | **$100.00\%$** $[100.00\%, 100.00\%]$ | **$97.30\%$** $[93.05\%, 100.00\%]$ | 0.9863 | **$0.9993$** $[0.9966, 1.0000]$ |
| **Meningioma** | 30 | **$93.33\%$** $[83.33\%, 100.00\%]$ | **$100.00\%$** $[100.00\%, 100.00\%]$ | 0.9655 | **$0.9987$** $[0.9940, 1.0000]$ |
| **Metastasis** | 30 | **$100.00\%$** $[100.00\%, 100.00\%]$ | **$100.00\%$** $[100.00\%, 100.00\%]$ | 1.0000 | **$1.0000$** $[1.0000, 1.0000]$ |

---

## 6. Confusion Matrix & Error Analysis

Confusion matrix on the held-out test cohort ($n=132$):

```
                     Predicted:
                 Glioma    Meningioma    Metastasis
Actual:
Glioma (n=72)      72          0             0
Meningioma (n=30)   2         28             0
Metastasis (n=30)   0          0            30
```

* **True Positive Rate:**
  * Glioma: $72 / 72 = 100.0\%$
  * Meningioma: $28 / 30 = 93.3\%$
  * Metastasis: $30 / 30 = 100.0\%$
* **Observed Misclassifications ($n=2$):** Exactly 2 Meningioma cases were classified as Glioma. Review indicates these cases presented with diffuse intra-axial peritumoral edema patterns on non-contrast T1 mimicking high-grade infiltrative glial tissue.
* **Metastasis Separation:** $30/30$ Metastasis cases were correctly identified with zero cross-contamination into Glioma or Meningioma ($100\%$ precision, $100\%$ recall, AUROC $1.0000$).

---

## 7. Probability Calibration & Uncertainty Routing

* **Temperature Scaling:** Fitted exclusively on the validation split ($n=133$) via grid minimization of ECE:
  $$\hat{T} = 0.4000$$
  This reduced validation ECE from $0.0616 \rightarrow 0.0159$, and test ECE from $0.0883 \rightarrow 0.0228$.
* **Abstain Policy:** If the calibrated maximum class probability $\max_c P(y=c|x) < 0.900$, the case is flagged as high-uncertainty and probability mass is transferred to `Uncertain/Other`.
* **Abstain Rate on Test Set:** $9 / 132$ scans ($6.8\%$) triggered the safety abstain buffer, ensuring border cases are escalated for manual neuroradiological review.

---

## 8. Metrics with Abstention

### Argmax vs. Abstention Protocol
The headline metrics presented in Section 5 (Balanced Accuracy: **97.78%**, Macro-F1: **98.39%**) reflect standard forced-choice **argmax evaluation** on the 3 diagnostic classes, excluding the safety abstention mechanism. In actual clinical deployment, when calibrated maximum confidence $\max_c P(y=c|x) < 0.900$, the system refuses to guess and routes the case to `Uncertain/Other`.

### Abstention Breakdown & Cohort Coverage
Across the $n=132$ held-out test subjects:
* **Total Abstained:** $9 / 132$ scans (**$6.82\%$**)
* **Cohort Coverage (Decided Cases):** **$93.18\%$** ($123 / 132$ cases)
* **Per-Class Abstention Support:**
  * **Glioma:** $3 / 72$ abstained ($4.17\%$)
  * **Meningioma:** $4 / 30$ abstained ($13.33\%$)
  * **Metastasis:** $2 / 30$ abstained ($6.67\%$)

### Selective Performance (Decided Cases Only, $n=123$)
When evaluating only scans that pass the confidence threshold:
* **Selective Accuracy:** **$99.19\%$** ($122 / 123$)
* **Selective Balanced Accuracy:** **$98.72\%$**
* **Selective Macro-F1:** **$99.11\%$**
* **Per-Class Recall:** Glioma $100.00\%$ ($69/69$), Meningioma $96.15\%$ ($25/26$), Metastasis $100.00\%$ ($28/28$)

### Deployment-View Performance (Penalizing Abstentions as Misses)
In a strict safety deployment view, routing a true positive tumor scan to `Uncertain/Other` is accounted as a false negative for that pathology (3 true classes $\times$ 4 predicted classes):

#### 3x4 Confusion Matrix:
| True Class | Predicted Glioma | Predicted Meningioma | Predicted Metastasis | Predicted Uncertain/Other (Abstained) | Total Support |
|---|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 69 | 0 | 0 | 3 | 72 |
| **Meningioma** | 1 | 25 | 0 | 4 | 30 |
| **Metastasis** | 0 | 0 | 28 | 2 | 30 |

* **Deployment Balanced Accuracy:** **$90.83\%$** (Bootstrap 95% CI: $[84.75\%, 96.10\%]$)
* **Deployment Macro-F1:** **$94.88\%$** (Bootstrap 95% CI: $[90.88\%, 97.96\%]$)
* **Per-Class Penalized Recall:**
  * **Glioma:** $69 / 72 = 95.83\%$ (95% CI: $[91.18\%, 100.00\%]$), Precision: $98.57\%$, F1: $97.18\%$
  * **Meningioma:** $25 / 30 = 83.33\%$ (95% CI: $[69.22\%, 96.43\%]$), Precision: $100.00\%$, F1: $90.91\%$
  * **Metastasis:** $28 / 30 = 93.33\%$ (95% CI: $[82.76\%, 100.00\%]$), Precision: $100.00\%$, F1: $96.55\%$

### Side-by-Side Comparison: Argmax-Only vs. Deployment-View

| Metric | Argmax-Only (Forced Choice) | Deployment-View (With Abstention) |
|---|:---:|:---:|
| **Balanced Accuracy** | **$97.78\%$** $[94.44\%, 100.00\%]$ | **$90.83\%$** $[84.75\%, 96.10\%]$ |
| **Macro-F1 Score** | **$98.39\%$** $[95.74\%, 100.00\%]$ | **$94.88\%$** $[90.88\%, 97.96\%]$ |
| **Glioma Sensitivity / Recall** | **$100.00\%$** | **$95.83\%$** $[91.18\%, 100.00\%]$ |
| **Meningioma Sensitivity / Recall** | **$93.33\%$** | **$83.33\%$** $[69.22\%, 96.43\%]$ |
| **Metastasis Sensitivity / Recall** | **$100.00\%$** | **$93.33\%$** $[82.76\%, 100.00\%]$ |

### Clinical Trade-Off Analysis
Abstention trades test coverage ($93.18\%$) for significantly higher selective diagnostic certainty (**$99.19\%$** accuracy on decided cases vs. $97.78\%$ in forced argmax). However, if abstentions are conservatively penalized as clinical misses, raw sensitivity decreases (e.g., Meningioma drops from $93.33\% \rightarrow 83.33\%$). In clinical decision support, routing ambiguous non-contrast scans to neuroradiologist review prevents automated diagnostic overconfidence on difficult borderline lesions.

---

## 9. Known Limitations

1. **Class Coverage:** Only 3 of the 10 official classes (`Glioma`, `Meningioma`, `Metastasis`) have training data available. The remaining 7 official classes (`Normal`, `Alzheimer's`, `FCD epilepsy`, `Post-stroke lesion`, `Parkinson's`, `ASD`, and `Uncertain/Other`) have zero local whole-brain training data ($n=0$) and rely entirely on uncertainty routing to `Uncertain/Other`.
2. **Brain Tissue Truncation via Center Crop:** `preprocess.py` uses a fixed center crop to $128 \times 128 \times 128$ voxels, which removes approximately $8\text{--}16\%$ of peripheral brain voxels (measured on 15 cases). The extent of tumor tissue loss resulting from this crop was not measured.
3. **Evaluation Limited to In-Distribution Sources:** All reported metrics originate from cohorts seen during training (`MSD`, `BraTS-MEN`, `Brain-Mets`). No external validation on unseen hospital scanners or independent cohorts was performed.
4. **Site-Confounding:** As demonstrated in the source-recognition check, acquisition source is predictable with $83.89\%$ balanced accuracy from intensity and noise features alone, meaning class and source cannot be disentangled in the current dataset.

---

## 10. Official Contract & Pipeline Integration Verification

The inference engine in `inference.py` and the test suite in `tests/test_task7_contract.py` verify all competition contract rules:
1. **Contract Format:** Emits `classification.json` containing all 10 official class names with probabilities summing strictly to $1.0000 \pm 10^{-4}$.
2. **Zero In-Memory Leakage:** Input processing runs identical preprocessing (`preprocess.py`) in $< 0.8$ seconds per 3D scan.
3. **Graceful Handling of Corrupted / Missing Input:** Corrupted volumes, zero-filled arrays, or empty brain masks fail QC and are immediately routed to `Uncertain/Other` with `qc_status: FAIL`.
4. **Offline Isolation:** Runs 100% offline with zero external network calls or web dependencies.
