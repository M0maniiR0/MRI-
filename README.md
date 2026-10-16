# 🧠 Medical AI Workstream — BrainAgenticthon 2026
## 3D Brain Tumour (Glioma) Segmentation & Detection

دليل منظم لملفات المشروع وسكريبتات التنفيذ لمهندسي وباحثي الذكاء الاصطناعي.

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
│   ├── export_agent_contracts.py      ← تصدير عقود التسليم للوكلاء (detection/classification)
│   ├── inspect_dataset.py             ← فحص تجريبي سريع لحالة واحدة
│   └── audit_dataset.py               ← جرد هيكلي سريع للملفات
│
├── 📁 docs/                           ← ملفات الـ PDF والتوثيق والتقارير الرسمية
│   ├── Medical_AI_Glioma_Technical_Report.pdf ← التقرير التقني الشامل للمشروع (6 صفحات)
│   ├── Data_Audit_Code_Explanation.pdf        ← الشرح البرمجي التفصيلي للكود سطراً بسطر
│   ├── 01_dataset_validation_guide.pdf        ← الدليل الإرشادي الأول
│   ├── 02_medical_ai_full_roadmap.pdf         ← خريطة الطريق الرسمية للمسابقة
│   └── MRI_BraTS_Data_Complete_Guide_AR.pdf   ← الدليل الشامل لداتا مشروع الرنين
│
├── 📁 reports/                        ← تقارير الجودة وصور الفحص البصري
│   ├── validation_report.json                 ← نتيجة فحص الـ 484 حالة (100% PASS)
│   ├── preprocessing_qc_report.json           ← تقرير فحص مسار المعالجة
│   ├── segmentation_validation_metrics.json   ← مقاييس تقييم نموذج التجزئة
│   └── qc_slice_BRATS_457.png                 ← صورة الفحص البصري الملونة
│
├── 📁 splits/                         ← ملفات تقسيم المرضى وجرد البيانات (Zero Leakage)
│   ├── train_ids.csv                          ← 339 مريض (70%)
│   ├── val_ids.csv                            ← 73 مريض (15%)
│   ├── test_ids.csv                           ← 72 مريض (15%)
│   ├── dataset_inventory.csv                  ← جرد جميع مسارات وقنوات الـ 484 حالة
│   └── detection_test_manifest.csv            ← قالب اختبار كاشف الشذوذ
│
├── 📁 outputs/                        ← مخرجات النماذج (الأقنعة mask.nii.gz)
│   └── segmentation/
│
├── 📁 handoff_deliverables/           ← حزم عقود التسليم المعيارية للـ Backend & Agents
│   └── BRATS_002/
│
├── 📁 tools/                          ← أدوات توليد ملفات الـ PDF والقوالب
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
