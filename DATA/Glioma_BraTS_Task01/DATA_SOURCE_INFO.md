# 🧬 مصادر وتفاصيل بيانات الأورام الدبقية (Glioma Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** Medical Segmentation Decathlon (MSD) — **Task01_BrainTumour** (BraTS Gold Standard).
* **الجهة الأكاديمية والمنظمة:** Center for Biomedical Image Computing and Analytics (CBICA) — University of Pennsylvania & TCGA.
* **المجال الطبي والسريري:** الأورام الدبقية (Gliomas) بما في ذلك الورم الأرومي الدبقي عالي الخباثة (Glioblastoma - GBM) وأورام الدرجة الدنيا (LGG).

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الحالات المتاحة:** **484 مريضاً** (بمجموع **1,936 حجم ثلاثي الأبعاد 3D NIfTI**).
  - **حالات التدريب (Training Cases):** 388 حالة تشمل الأقنعة الخبيرة.
  - **حالات الاختبار (Test Cases):** 96 حالة للاختبار المحجوب.
* **التسلسلات الإشعاعية الأربعة المدمجة (4 Channels في ملف واحد):**
  1. `Channel 0`: **FLAIR** (لتحديد الوذمة والارتشاح المحيط بالورم).
  2. `Channel 1`: **T1w** (التشريح الأساسي لأنسجة الدماغ).
  3. `Channel 2`: **T1gd** (معزز بالصبغة لإظهار الحواف النشطة للورم).
  4. `Channel 3`: **T2w** (لتحديد كثافة السوائل والكتلة المركزية).
* **أبعاد الصورة القياسية:** $240 \times 240 \times 155$ بدقة $1.0\text{ mm}$ مكعبة.
* **صيغة الملفات:** NIfTI Compressed (`.nii.gz`).

---

## 3. المناطق الورمية المستهدفة (Tumor Subregions)
1. **Whole Tumor (WT):** كامل النسيج المتأثر بالورم (الوذمة + المركز + النسيج النشط).
2. **Tumor Core (TC):** نواة الورم (النسيج المتنخر + النسيج المعزز النشط).
3. **Enhancing Tumor (ET):** النسيج الورمي النشط الشديد التروية الدموية.

---

## 4. المرجع العلمي والترخيص (Citation & License)
* **المرجع العلمي:**
  > Bakas, S., et al. (2017). *"Advancing The Cancer Genome Atlas glioma analysis with pre-processed volumetric data and expert labels."* Scientific Data 4:170117.
* **الترخيص:** CC-BY-SA 4.0 International.

---

## 5. روابط التحميل والمسار المحلي (Download Source & Local Path)
* **الرابط الرسمي:** [Medical Segmentation Decathlon Challenge](http://medicaldecathlon.com/)
* **المسار المحلي للبيانات:** [`DATA/Glioma_BraTS_Task01/data_nii/`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/DATA/Glioma_BraTS_Task01/data_nii)
