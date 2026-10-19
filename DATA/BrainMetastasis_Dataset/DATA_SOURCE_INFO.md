# 🎯 مصادر وتفاصيل بيانات الأورام المنتشرة إلى الدماغ (Brain Metastasis Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** **Consolidated Multi-Modal Brain Metastases Cohort (240 Cases)**.
* **المستودعات المباشرة المعتمدة:**
  1. [Hugging Face — Emulated-Inc/sci-brain-metastases-new-centre](https://huggingface.co/datasets/Emulated-Inc/sci-brain-metastases-new-centre) (200 مريض متعدد الأنماط: T1c, T1n, T2f, T2w, Seg).
  2. [Hugging Face — Pakawat-Phasook/brain_metastases_dataset](https://huggingface.co/datasets/Pakawat-Phasook/brain_metastases_dataset) (40 مريضاً: T1c, Lesion Mask, Brain Mask).
* **المجال الطبي والسريري:** الأورام الثانوية المنتشرة إلى المخ (Metastases) الناتجة غالباً عن سرطانات أولية أخرى (مثل الرئة، الثدي، أو الميلانوما). تتميز بوجود بؤر متعددة وصغيرة الحجم تتطلب رصداً دقيقاً لعددها ومجموع حجمها لتحديد خطة العلاج الإشعاعي (Stereotactic Radiosurgery - SRS) أو الجراحي.

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الحالات المتاحة والمدرّبة:** **240 مريضاً متكاملاً** (بمجموع يتجاوز **1,000 ملف رنين وقناع ثلاثي الأبعاد 3D NIfTI**).
  - **حالات التدريب (Training Cohort):** 192 مريضاً.
  - **حالات التحقق (Validation Cohort):** 48 مريضاً.
* **التسلسلات الإشعاعية المعتمدة لكل مريض:**
  1. `t1c.nii.gz / mri.nii.gz`: رنين مغناطيسي ثلاثي الأبعاد معزز بالصبغة بدقة كاملة $1.0\text{ mm}$ مكعبة (Isotropic).
  2. `seg.nii.gz / lesion_mask.nii.gz`: أقنعة التحديد ثلاثية الأبعاد للبؤر الورمية المنتشرة معتمدة شعاعياً (Ground Truth).
  3. `t1n`, `t2f`, `t2w`: تسلسلات الرنين المساعدة لتأكيد الوذمة المحيطة والتشريح النسيجي.
* **صيغة الملفات:** NIfTI Compressed (`.nii.gz`).

---

## 3. الأهداف والمخرجات التشخيصية (Clinical Diagnostic Outputs)
1. **حساب العبء الحجمي للورم (Metastatic Burden):** قياس الحجم الإجمالي للبؤر بالمليمتر المكعب ($mm^3$).
2. **عدّ البؤر الانتقالية (Foci Count):** تمييز ما إذا كانت الإصابة بؤرة منفردة (Solitary Metastasis) أو بؤر متعددة (Multiple Foci) لتحديد ملاءمة الجراحة الإشعاعية.
3. **التصوير متعدد المحاور (3-View Visual Overlays):** تحديد موقع كل بؤرة بدقة على مقاطع Axial و Coronal و Sagittal.

---

## 4. المرجع العلمي والترخيص (Citation & License)
* **الترخيص:** Open Access Academic Research (Apache 2.0 / CC-BY).
* **المصدر السحابي:** Hugging Face Datasets API.

---

## 5. روابط التحميل وطريقة التنزيل (Download Source & Automation)
* **السكريبت الآلي للتنزيل المسرّع:**
  ```bash
  python download_brain_metastasis.py --num-patients 40 --workers 6
  ```
* **المسار المحلي للبيانات:** [`DATA/BrainMetastasis_Dataset/data_nii/`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/DATA/BrainMetastasis_Dataset/data_nii)
