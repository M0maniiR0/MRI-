# 🧠 مصادر وتفاصيل بيانات مرض الزهايمر (Alzheimer's Disease Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** Medical Segmentation Decathlon (MSD) — **Task04_Hippocampus**
* **الجهة الأكاديمية المصدرة:** Vanderbilt University Medical Center (VUMC)
* **المجال الطبي والسريري:** التقييم الحجمي لضمور قرن آمون (Hippocampus Volumetric Biomarker) المرتبط سريرياً بمرض الزهايمر والتدهور الإدراكي (Mild Cognitive Impairment - MCI).

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الحالات المتاحة:** **390 حالة رنين مغناطيسي ثلاثية الأبعاد (3D MRI)**.
  - **حالات التدريب (Training Cases):** 260 حالة (مع الأقنعة المرجعية المعتمدة).
  - **حالات الاختبار (Test Cases):** 130 حالة.
* **نوع التسلسل الإشعاعي (Modality):** T1-weighted High-Resolution 3D MRI.
* **أبعاد الـ Voxel Spacing:** متماثلة بدقة $1.0 \times 1.0 \times 1.0\text{ mm}$ (Isotropic).
* **صيغة الملفات:** NIfTI Compressed (`.nii.gz`).

---

## 3. التسميات التشريحية (Anatomical Labels)
تحتوي كل حالة على تسميات ثلاثية الأبعاد دقيقة مقسمة إلى:
* `0`: **الخلفية (Background)**
* `1`: **قرن آمون الأمامي (Anterior Hippocampus)** — المسؤول عن معالجة الذاكرة العاطفية والمكانية المبكرة.
* `2`: **قرن آمون الخلفي (Posterior Hippocampus)** — المسؤول عن الإدراك والترميز المعرفي المتقدم.

---

## 4. المرجع العلمي والترخيص (Citation & License)
* **الترخيص:** Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0).
* **المرجع العلمي:**
  > Simpson, A. L., et al. (2019). *"A large annotated medical image dataset for the development and evaluation of segmentation algorithms."* arXiv preprint arXiv:1902.09063.

---

## 5. روابط التحميل وطريقة التنزيل (Download Source & Automation)
* **الرابط المباشر:** [Medical Segmentation Decathlon Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
* **السكريبت الآلي للتنزيل:** يمكن تنزيل وفك ضغط البيانات بضغطة زر عبر:
  ```bash
  python download_alzheimer_hippocampus.py
  ```
* **المسار المحلي للبيانات:** [`DATA/Alzheimer_Hippocampus/data_nii/`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/DATA/Alzheimer_Hippocampus/data_nii)
