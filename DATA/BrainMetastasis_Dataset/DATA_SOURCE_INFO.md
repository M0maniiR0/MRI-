# 🎯 مصادر وتفاصيل بيانات الأورام المنتشرة إلى الدماغ (Brain Metastasis Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** **Brain Metastases MRI Dataset** (Pakawat-Phasook / Hugging Face).
* **المستودع المباشر:** [Hugging Face — Pakawat-Phasook/brain_metastases_dataset](https://huggingface.co/datasets/Pakawat-Phasook/brain_metastases_dataset).
* **المجال الطبي والسريري:** الأورام الثانوية المنتشرة إلى المخ (Metastases) الناتجة غالباً عن سرطانات أولية أخرى (مثل الرئة، الثدي، أو الميلانوما). تتميز بوجود بؤر متعددة وصغيرة الحجم تتطلب رصداً دقيقاً لعددها ومجموع حجمها لتحديد خطة العلاج الإشعاعي (Stereotactic Radiosurgery - SRS) أو الجراحي.

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الحالات المتاحة والمدرّبة:** **40 مريضاً متكاملاً** (بمجموع **120 ملف رنين وقناع ثلاثي الأبعاد 3D NIfTI**).
  - **حالات التدريب (Training Cohort):** 32 مريضاً.
  - **حالات التحقق (Validation Cohort):** 8 مرضى.
* **التسلسلات الإشعاعية المعتمدة لكل مريض:**
  1. `mri.nii.gz`: رنين مغناطيسي ثلاثي الأبعاد معزز بالصبغة بدقة كاملة $240 \times 240 \times 155$ و $1.0\text{ mm}$ مكعبة (Isotropic).
  2. `lesion_mask.nii.gz`: أقنعة التحديد ثلاثية الأبعاد للبؤر الورمية المنتشرة معتمدة شعاعياً (Ground Truth).
  3. `brain_mask.nii.gz`: قناع عزل نسيج المخ لتفادي أي تشويش خارجي.
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
