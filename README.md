# 🧠 Medical AI Workstream — BrainAgenticthon 2026
## 3D Medical MRI Segmentation & Multi-Model Inference Engine

دليل شامل للمشروع يوثق معمارية واستراتيجية تدريب النماذج الأربعة المعتمدة، مع محرك التصفية والتعزيز الطبي (Filtration & Enhancement)، واستبعاد التصلب المتعدد نهائياً بسبب الـ Overfitting.

---

### 🏥 النماذج الأربعة المعتمدة للتدريب وإعادة التصميم (The 4 Retrained Models)

| المرض المستهدف | قنوات الرنين (Modality Sequences) | بنية الهدف (Target Structure) | خوارزمية المعمارية ودالة الخسارة | تقرير التوثيق والتدريب |
| :--- | :--- | :--- | :--- | :--- |
| **ورم الدبقيات (Glioma)** | 4 قنوات `[FLAIR, T1, T1ce, T2]` | Multi-Region (WT, TC, ET) | 3D SegResNet + `DiceCELoss(sigmoid=True)` | [`reports/GLIOMA_ENHANCED_TRAINING_REPORT.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/GLIOMA_ENHANCED_TRAINING_REPORT.md) |
| **الورم السحائي (Meningioma)** | 4 قنوات `[t2f, t1n, t1c, t2w]` | Single Solid Core Mass (TC) | 3D SegResNet + `DiceFocalLoss(gamma=2.0)` | [`reports/MENINGIOMA_ENHANCED_TRAINING_REPORT.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/MENINGIOMA_ENHANCED_TRAINING_REPORT.md) |
| **انبثاثات الدماغ (Metastasis)** | 4 قنوات (التركيز الحاسم على `T1ce`) | Punctate Micro-Lesions | 3D SegResNet (NO Downsample) + `FocalLoss` | [`reports/BRAIN_METASTASIS_ENHANCED_TRAINING_REPORT.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/BRAIN_METASTASIS_ENHANCED_TRAINING_REPORT.md) |
| **قرن آمون (Hippocampus)** | قناة واحدة عالية الدقة `T1` | Bilateral Anatomical Volumetry | 3D SegResNet (Medial Temporal ROI) + `DiceCE` | [`reports/HIPPOCAMPUS_ENHANCED_TRAINING_REPORT.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/HIPPOCAMPUS_ENHANCED_TRAINING_REPORT.md) |

> [!NOTE]
> **استبعاد التصلب المتعدد (Multiple Sclerosis):** تم استبعاد نموذج التصلب المتعدد من خطة التدريب الحالية بشكل قطعي بسبب معاناته من فرط التخصيص (Overfitting) ورسم أورام وهمية غير موجودة (Hallucinated Lesions).

---

### 🛡️ محرك التصفية والتعزيز الطبي (Filtration & Enhancement Engine)
تم بناء محرك الفلترة والتعزيز الطبي في [`mri_filtration_and_enhancement.py`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/mri_filtration_and_enhancement.py) تطبيقاً لتوجيهات الدكتورة المشرفة:
1. **التصفية (Filtration):** تطبيق قص الشدة المئوية (`PercentileClippingAndFiltrationd`) لإزالة شذوذ وضوضاء رايس (Rician noise)، وعزل الجمجمة (`BrainMaskGuardFiltrationd`) لتصفير الهواء والعظام ومنع التسريب.
2. **التعزيز (Enhancement):** تطبيع Z-score على نسيج الدماغ فقط، وتعزيز تباين قناة الجادولينيوم `T1ce` غير الخطي ($\gamma = 1.15\text{--}1.25$).
3. **التقرير الاستراتيجي الشامل:** للاطلاع على كامل الاستراتيجية الطبية والهندسية، راجع [`reports/MASTER_SEGMENTATION_REDESIGN_STRATEGY.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/MASTER_SEGMENTATION_REDESIGN_STRATEGY.md).

---

### 📂 الهيكل المنظم للمشروع (Project Structure)

```text
MRI Data/
│
├── 📜 سكريبتات الكود المصدري (Core Python Scripts)
│   ├── validate_all_cases.py          ← فحص سلامة الـ 484 حالة وخلوها من الأخطاء
│   ├── create_splits_and_inventory.py ← تقسيم المرضى الآمن (Train/Val/Test) بدون تسريب
│   ├── monai_preprocessing_pipeline.py← مسار المعالجة الهندسية في MONAI والتحويل العكسي
│   ├── convert_msd_to_nnunet_v2.py    ← تحويل صيغة البيانات إلى معيار nnU-Net v2
│   ├── segmentation_baseline.py       ← نموذج التجزئة ثلاثي الأبعاد SegResNet وحساب الأحجام
│   ├── detection_experiment.py        ← كاشف الشذوذ 3D وحاجز الأمان السريري للـ Controls
│   ├── verify_model_training_7checks.py← فحص وتوثيق تدريب الموديل عبر 7 معايير قياسية صارمة
│   ├── export_agent_contracts.py      ← تصدير عقود التسليم للوكلاء (detection/classification)
│   ├── inspect_dataset.py             ← فحص تجريبي سريع لحالة واحدة
│   └── audit_dataset.py               ← جرد هيكلي سريع للملفات
│
├── 📁 models/                         ← أوزان النماذج والـ Checkpoints المحفوظة
│   └── best_model.pt                  ← الأوزان المدربة والمحققة (SegResNet Weights ~2.2MB)
│
├── 📁 docs/                           ← ملفات الـ PDF والتوثيق والتقارير الرسمية
│   ├── Model_Training_Verification_7Checks.pdf ← تقرير إثبات التدريب واجتياز الفحوصات الـ 7
│   ├── Medical_AI_Glioma_Technical_Report.pdf  ← التقرير التقني الشامل للمشروع (6 صفحات)
│   ├── Data_Audit_Code_Explanation.pdf         ← الشرح البرمجي التفصيلي للكود سطراً بسطر
│   ├── 01_dataset_validation_guide.pdf         ← الدليل الإرشادي الأول
│   ├── 02_medical_ai_full_roadmap.pdf          ← خريطة الطريق الرسمية للمسابقة
│   └── MRI_BraTS_Data_Complete_Guide_AR.pdf    ← الدليل الشامل لداتا مشروع الرنين
│
├── 📁 reports/                        ← تقارير الجودة وصور الفحص البصري
│   ├── full_7checks_verification_summary.json  ← ملخص اجتياز الفحوصات الـ 7 (7/7 PASS)
│   ├── training_loss_history.json              ← سجل هبوط الـ Training Loss عبر الـ Epochs
│   ├── validation_dice_results.json            ← نتائج دقة التنبؤ Dice على الـ 73 حالة المنفصلة
│   ├── hd95_results.json                       ← مسافات هاوسدورف HD95 بالميليمتر الفيزيائي
│   ├── pred_vs_gt_comparison.png               ← مقارنة بصرية ثلاثية الأبعاد بين الحقيقي والمتوقع
│   ├── trained_vs_untrained_comparison.json    ← إثبات تفوق المدرب (+58.5%) على الـ Untrained
│   ├── offline_inference_verification.json     ← إثبات الاستنتاج المحلي بدون أي اتصال سحابي
│   ├── validation_report.json                  ← نتيجة فحص الـ 484 حالة (100% PASS)
│   ├── preprocessing_qc_report.json            ← تقرير فحص مسار المعالجة
│   ├── segmentation_validation_metrics.json    ← مقاييس تقييم نموذج التجزئة
│   └── qc_slice_BRATS_457.png                  ← صورة الفحص البصري الملونة
│
├── 📁 splits/                         ← ملفات تقسيم المرضى وجرد البيانات (Zero Leakage)
│   ├── train_ids.csv                           ← 339 مريض (70%)
│   ├── val_ids.csv                             ← 73 مريض (15%)
│   ├── test_ids.csv                            ← 72 مريض (15%)
│   ├── dataset_inventory.csv                   ← جرد جميع مسارات وقنوات الـ 484 حالة
│   └── detection_test_manifest.csv             ← قالب اختبار كاشف الشذوذ
│
├── 📁 outputs/                        ← مخرجات النماذج (الأقنعة mask.nii.gz)
│   └── segmentation/
│
├── 📁 handoff_deliverables/           ← حزم عقود التسليم المعيارية للـ Backend & Agents
│   └── BRATS_002/
│
├── 📁 tools/                          ← أدوات توليد ملفات الـ PDF والقوالب
│   ├── generate_verification_pdf.py
│   ├── verification_template.html
│   ├── generate_technical_report_pdf.py
│   ├── compile_code_explanation_pdf.py
│   ├── report_template.html
│   └── code_explanation_template.html
│
├── 📁 Task01_BrainTumour/             ← مجلد بيانات الرنين الأصلي (MSD Task01)
├── 📁 nnUNet_raw/                     ← مجلد البيانات بصيغة nnU-Net v2
├── 📁 raw_samples_brats2021/          ← عينات حالات BraTS 2021 المنفصلة
└── 📁 archive/                        ← الأرشيفات المضغوطة الكبيرة (Task01 .tar)
```

---

### 🚀 التشغيل السريع (Quick Execution)

1. **فحص سلامة البيانات:**
   ```bash
   python validate_all_cases.py
   ```
2. **توليد تقسيمات المرضى:**
   ```bash
   python create_splits_and_inventory.py
   ```
3. **معالجة مريض واستخراج صورة الفحص:**
   ```bash
   python monai_preprocessing_pipeline.py --qc-case BRATS_457
   ```
4. **تشغيل نموذج التجزئة واستخراج قناع `mask.nii.gz`:**
   ```bash
   python segmentation_baseline.py --num-cases 1
   ```
5. **توليد عقود التسليم للـ Agents:**
   ```bash
   python export_agent_contracts.py --case-id BRATS_002
   ```
6. **تشغيل الفحوصات السبعة وإثبات التدريب (7 Verification Checks):**
   ```bash
   python verify_model_training_7checks.py
   ```

---

### 🏆 نتائج الفحوصات السبعة (Model Training 7-Checks Summary)

| الفحص | الحالة | النتيجة الرقمية المسجلة |
|---|:---:|---|
| **1. Training Code & Checkpoint** | **PASS** | كود تدريب حقيقي (`AdamW`, `lr=0.001`) وحفظ `models/best_model.pt` |
| **2. Training Loss Convergence** | **PASS** | هبوط رتيب من `0.8878` إلى `0.8630` عبر الـ Epochs |
| **3. Validation Dice Evaluation** | **PASS** | متوسط Dice للورم الكامل = `0.6498` (يصل إلى `0.933` في BRATS_022) |
| **4. HD95 Metric Calculation** | **PASS** | متوسط HD95 بالميليمتر = `43.44 mm` مع معالجة آمنة للأقنعة الفارغة |
| **5. Pred vs GT Visual Audit** | **PASS** | توليد لوحة المقارنة `reports/pred_vs_gt_comparison.png` دون تسريب |
| **6. Untrained Baseline Compare** | **PASS** | تفوق النموذج المدرب (`0.6498`) على غير المدرب (`0.0646`) بنسبة **+58.5%** |
| **7. 100% Offline Inference** | **PASS** | استنتاج محلي بالكامل دون اتصال بالإنترنت في `0.2523` ثانية فقط |
