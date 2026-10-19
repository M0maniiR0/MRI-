# 🎗️ مصادر وتفاصيل بيانات الأورام السحائية (Meningioma Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** **BraTS-MEN 2023** (The Brain Tumor Segmentation Challenge 2023: Meningioma).
* **الجهة الأكاديمية والمنظمة:** تحالف المعاهد الدولية لسرطان الدماغ (RSNA, ASNR, MICCAI).
* **المستودع المباشر:** [Hugging Face — MedOtter/brats2023-men-dataset](https://huggingface.co/datasets/MedOtter/brats2023-men-dataset).
* **المجال الطبي والسريري:** الأورام السحائية هي أورام تنشأ في الأغشية المحيطة بالدماغ والحبل الشوكي. يتطلب تحديدها وفصلها قياسات دقيقة على صور الرنين ذات التعزيز الصبغي (T1c).

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الحالات المتاحة في المستودع العالمي:** **953 حالة مريض متكاملة**.
* **الحالات المجهزة والمدرّبة محلياً:** **200 حالة مريض متكاملة** (بمجموع **1,000 ملف رنين ثلاثي الأبعاد 3D NIfTI** مقسمة إلى 160 تدريب و 40 تحقق سريري).
* **التسلسلات الإشعاعية الأربعة (4 Multiparametric MRI Modalities لكل مريض):**
  1. `t1n.nii.gz`: T1 Native (التشريح الأساسي بدون صبغة).
  2. `t1c.nii.gz`: T1 Contrast-Enhanced (مع صبغة الغادولينيوم لتحديد كتلة الورم بدقة فائقة).
  3. `t2f.nii.gz`: T2-FLAIR (لتقييم الارتشاح المائي حول الورم Peritumoral Edema).
  4. `t2w.nii.gz`: T2-weighted (لإظهار الفروق النسيجية وسوائل الدماغ).
  5. `seg.nii.gz`: قناع الورم المعتمد يدوياً من أطباء الأشعة المعتمدين (Ground Truth).
* **صيغة الملفات:** NIfTI Compressed (`.nii.gz`).

---

## 3. التسميات التشريحية (Anatomical Labels)
* **Whole Tumor (WT):** كامل الكتلة الورمية السحائية ومحيطها.
* **Tumor Core (TC):** النواة الصلبة للورم.
* **Enhancing Tumor (ET):** النسيج الورمي النشط الماص للصبغة.

---

## 4. المرجع العلمي والترخيص (Citation & License)
* **المرجع العلمي:**
  > Karargyris, A., et al. (2023). *"The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on Meningioma."* Radiological Society of North America & MICCAI.
* **الترخيص:** Open Research Academic Access (CC-BY 4.0).

---

## 5. روابط التحميل وطريقة التنزيل (Download Source & Automation)
* **المستودع السحابي:** `https://huggingface.co/datasets/MedOtter/brats2023-men-dataset`
* **السكريبت الآلي للتنزيل المسرّع:** يمكن تحميل أي عدد من الحالات عبر التحميل المتوازي:
  ```bash
  python download_meningioma_data.py --num-cases 25 --workers 6
  ```
* **المسار المحلي للبيانات:** [`DATA/Meningioma_BraTS_MEN/data_nii/`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/DATA/Meningioma_BraTS_MEN/data_nii)
