# 🎗️ مصادر وتفاصيل بيانات التصلب المتعدد (Multiple Sclerosis Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** **MSLesSeg-2024: Multiple Sclerosis Lesion Segmentation Benchmark**.
* **المستودع المباشر والمصدر العلمي:** [Nature Scientific Data / Figshare — MSLesSeg Dataset](https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209) (DOI: 10.6084/m9.figshare.27919209).
* **الكود الرسمي:** [GitHub — alessiarondinella/MSLesSeg-2024](https://github.com/alessiarondinella/MSLesSeg-2024).
* **المجال الطبي والسريري:** مرض التصلب المتعدد / التصلب اللويحي (Multiple Sclerosis - MS) هو مرض مناعي ذاتي مزمن يهاجم فيه الجهاز المناعي غمد المايلين (Myelin Sheath) المحيط بالألياف العصبية في الدماغ والحبل الشوكي، مما يسبب بؤراً التهابية وتندبات (Demyelinating Plaques / Lesions). تتطلب المتابعة السريرية قياس العبء الحجمي الإجمالي للآفات (Total Lesion Load - TLL بالمليمتر المكعب) ورصد ظهور آفات جديدة أو تضخمها وفق معايير ماكدونالد (McDonald Criteria).

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الفحوصات والحالات:** **115 فحصاً ثلاثي الأبعاد لـ 75 مريضاً متكاملاً** (بمجموع **460 ملف رنين وقناع ثلاثي الأبعاد 3D NIfTI** بحجم غير مضغوط يتجاوز 3.8 جيجابايت).
  - **حالات التدريب (Training Cohort):** 92 فحصاً ثلاثي الأبعاد.
  - **حالات التحقق والتقييم (Validation Cohort):** 23 فحصاً ثلاثي الأبعاد.
* **المعالجة المعيارية المسبقة (Standardized Preprocessing):**
  1. التسجيل والمطابقة الفراغية المعيارية مع قالب MNI152 بدقة متساوية الأبعاد $1.0\text{ mm}^3$ (Isotropic Spacing).
  2. عزل الجمجمة والأنسجة غير الدماغية (Brain Extraction / BET Skull-stripping).
* **التسلسلات الإشعاعية المعتمدة لكل فحص:**
  1. `*_FLAIR.nii.gz`: تسلسل T2-FLAIR ثلاثي الأبعاد (التسلسل الحاسم لرصد اللويحات البيضاء Periventricular Plaques).
  2. `*_MASK.nii.gz`: أقنعة التحديد ثلاثية الأبعاد المعتمدة سريرياً من قبل استشاريي الأشعة العصبية (Ground Truth).
  3. `*_T1.nii.gz` و `*_T2.nii.gz`: تسلسلات الرنين المغناطيسي التشريحية المصاحبة.
* **صيغة الملفات:** NIfTI Compressed (`.nii.gz`).

---

## 3. الأهداف والمخرجات التشخيصية (Clinical Diagnostic Outputs)
1. **حساب العبء الحجمي للآفات (Total Lesion Load - TLL):** قياس مجموع أحجام اللويحات بالمليمتر المكعب ($mm^3$).
2. **عدّ اللويحات الالتهابية (Plaque Count):** استخراج عدد البؤر التندبية المستقلة في المادة البيضاء عبر تحليل المكونات المتصلة (Connected Components).
3. **التصنيف السريري لشدة المرض (Clinical Burden Staging):**
   - عبء خفيف: $< 5,000\text{ mm}^3$
   - عبء متوسط: $5,000 - 15,000\text{ mm}^3$
   - عبء شديد / متقدم: $> 15,000\text{ mm}^3$
4. **التصوير ثلاثي المحاور (3-View Visual Overlays):** تحديد موقع كل لويحة في مقاطع Coronal و Sagittal و Axial بلون سيان كهربائي بارز.

---

## 4. نموذج الذكاء الاصطناعي والأوزان المدربة (Model Checkpoint)
* **المعمارية:** 3D SegResNet مع Automatic Mixed Precision (AMP FP16).
* **الأوزان المعتمدة:** [`models/segresnet_multiple_sclerosis_best.pt`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/models/segresnet_multiple_sclerosis_best.pt).
* **أفضل دقة تحقق (Validation Dice):** **`0.4525`** (مع وصول الدقة في الحالات عالية العبء إلى **`0.787` Dice**).
* **ملف التقييم:** [`reports/multiple_sclerosis_cohort_report.md`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/reports/multiple_sclerosis_cohort_report.md).
