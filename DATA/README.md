# 🗂️ الدليل الشامل لمصادر وبيانات مشروع NeuroScope AI (DATA)

يحتوي هذا المجلد (`DATA/`) على جميع مجموعات البيانات الطبية ثلاثية الأبعاد (3D MRI Datasets) المعتمدة عالمياً والتي تم استخدامها لتدريب واختبار وتقييم نماذج الذكاء الاصطناعي في مشروع **BrainAgenticthon 2026**.

تم تنظيم البيانات بشكل منفصل تماماً لكل مرض لضمان سهولة الفهم والوصول السريع:

```
DATA/
├── 🧠 Alzheimer_Hippocampus/          # مرض الزهايمر وضمور الحُصين
│   ├── data_nii/                      # صور الرنين NIfTI (390 حالة: 260 تدريب + 130 اختبار)
│   └── DATA_SOURCE_INFO.md            # التفاصيل الطبية، الروابط، والمراجع العلمية
│
├── 🎯 BrainMetastasis_Dataset/        # الأورام المنتشرة إلى الدماغ (Metastases)
│   ├── data_nii/                      # صور الرنين NIfTI (40 مريضاً: 120 حجماً ثلاثي الأبعاد)
│   └── DATA_SOURCE_INFO.md            # التفاصيل الطبية، رابط Hugging Face، والمراجع
│
├── 🎗️ Meningioma_BraTS_MEN/           # أورام السحايا الدماغية
│   ├── data_nii/                      # صور الرنين NIfTI (25 مريضاً: 125 حجماً ثلاثي الأبعاد)
│   └── DATA_SOURCE_INFO.md            # التفاصيل الطبية، رابط Hugging Face، والمراجع
│
├── 🧬 Glioma_BraTS_Task01/            # الأورام الدبقية عالية ومنخفضة الدرجة
│   ├── data_nii/                      # صور الرنين NIfTI (484 مريضاً: 1,936 حجماً ثلاثي الأبعاد)
│   └── DATA_SOURCE_INFO.md            # التفاصيل الطبية، بروتوكول BraTS، والمراجع
│
└── 📖 README.md                       # هذا الفهرس الشامل
```

---

## 📊 جدول المقارنة الشامل لمصادر البيانات (Master Dataset Comparison)

| المرض / المهمة السريرية | المجموعة المعتمدة (Dataset) | الجهة المانحة / المرجع | عدد الحالات (Cases) | التسلسلات الإشعاعية (MRI Modalities) | أوزان النموذج المدرّب |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **1. الأورام المنتشرة (Brain Metastasis)** | **Brain Metastases MRI Dataset** | Pakawat-Phasook / Hugging Face | **40 مريضاً** | 3D High-Res T1-CE MRI ($240\times240\times155$) | `models/segresnet_brain_metastasis_best.pt` |
| **2. مرض الزهايمر (Alzheimer's)** | **MSD Task04_Hippocampus** | Vanderbilt Univ (VUMC) | **390 حالة** | T1-weighted 3D Isotropic ($1\text{mm}^3$) | `models/segresnet_alzheimer_hippocampus_best.pt` |
| **3. الأورام السحائية (Meningioma)** | **BraTS-MEN 2023** | RSNA / ASNR / MICCAI | **953 متاحة** (25 محلية) | 4 قنوات: T1n, T1c, T2f, T2w | `models/segresnet_meningioma_best.pt` |
| **4. الأورام الدبقية (Glioma)** | **MSD Task01_BrainTumour** | Penn Medicine (CBICA) | **484 حالة** | 4 قنوات: FLAIR, T1w, T1gd, T2w | `models/segresnet_fullvolume_best.pt` |

---

## 🔬 الأثر الطبي والتطبيقي لكل مجموعة بيانات:
1. **الأورام المنتشرة (Brain Metastasis):** رصد وتحديد البؤر السرطانية الثانوية المنبثة في الدماغ، وحساب عبء الورم الكلي ($mm^3$)، وعدّ البؤر (Foci Count) لدعم قرارات الجراحة الإشعاعية (SRS).
2. **الزهايمر (Alzheimer):** حساب الحجم الدقيق لقرن آمون بالمليمتر المكعب ($mm^3$) لتحديد الضمور المبكر وتصنيف الحالات سريرياً إلى (Normal vs. MCI vs. Severe Atrophy).
3. **الأورام السحائية (Meningioma):** كشف وتحديد حدود الأورام السطحية الخارجية التي تضغط على قشرة الدماغ وعزلها عن أنسجة المخ السليمة.
4. **الأورام الدبقية (Glioma):** تشريح الورم الداخلي متعدد المناطق (الوذمة الارتشاحية، المركز المتنخر، الحواف الماصة للصبغة).

---

## 💻 طريقة تشغيل التدريب لأي مرض بضغطة زر:
* **لتدريب الأورام المنتشرة:** تشغيل [`train_metastasis_gpu.bat`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/train_metastasis_gpu.bat)
* **لتدريب الزهايمر:** تشغيل [`train_alzheimer_gpu.bat`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/train_alzheimer_gpu.bat)
* **لتدريب الأورام السحائية:** تشغيل [`train_meningioma_gpu.bat`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/train_meningioma_gpu.bat)
* **لتقييم الزهايمر الحجمي:** `python quantify_alzheimer_atrophy.py`
* **لتقييم الأورام المنتشرة:** `python quantify_brain_metastasis.py`
* **لتقييم الأورام السحائية:** تشغيل [`run_meningioma_evaluation.bat`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/run_meningioma_evaluation.bat)
