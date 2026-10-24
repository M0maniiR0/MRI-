# 🎗️ مصادر وتفاصيل بيانات التصلب المتعدد (Multiple Sclerosis Dataset)

## 1. التعريف بالبيانات (Dataset Overview)
* **اسم المجموعة المعتمد:** **MSLesSeg-2024 & WMH Challenge (Consolidated 240 Cohort)**.
* **المستودع المباشر والمصادر العلمية:** 
  1. [Nature Scientific Data / Figshare — MSLesSeg Dataset](https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209) (DOI: 10.6084/m9.figshare.27919209) لـ 115 فحصاً معيارياً.
  2. [MICCAI WMH Challenge (White Matter Hyperintensities / MS Lesions)](https://huggingface.co/datasets/MedOtter/wmh-segmentation) لـ 125 فحصاً متعدد المراكز (Utrecht, Singapore, Amsterdam).
* **المجال الطبي والسريري:** مرض التصلب المتعدد / التصلب اللويحي (Multiple Sclerosis - MS) هو مرض مناعي ذاتي مزمن يهاجم فيه الجهاز المناعي غمد المايلين (Myelin Sheath) المحيط بالألياف العصبية في الدماغ والحبل الشوكي، مما يسبب بؤراً التهابية وتندبات (Demyelinating Plaques / Lesions). تتطلب المتابعة السريرية قياس العبء الحجمي الإجمالي للآفات (Total Lesion Load - TLL بالمليمتر المكعب) ورصد ظهور آفات جديدة أو تضخمها وفق معايير ماكدونالد (McDonald Criteria).

---

## 2. الإحصائيات الفنية للبيانات (Technical Specifications)
* **إجمالي الفحوصات والحالات:** **240 فحصاً وحالة ثلاثية الأبعاد 3D NIfTI متكاملة ومحققة 100%**.
  - **حالات التدريب (Training Cohort):** 192 فحصاً ثلاثي الأبعاد (80%).
  - **حالات التحقق والتقييم (Validation Cohort):** 48 فحصاً ثلاثي الأبعاد (20%).
* **المعالجة المعيارية المسبقة (Standardized Preprocessing):**
  1. مطابقة فراغية وإعادة ضبط الكثافة مع وسادات فراغية ذكية `SpatialPadd` لاستيعاب كافة أحجام المسح.
  2. عزل الأنسجة الدماغية ورصد التباين على صور FLAIR و T1.
* **التسلسلات الإشعاعية المعتمدة لكل فحص:**
  1. صور FLAIR ثلاثية الأبعاد (3D T2-FLAIR / 2D Multi-slice FLAIR): التسلسل الأكثر حساسية لرصد لويحات المادة البيضاء (White Matter Hyperintensities & Demyelinating Plaques).
  2. أقنعة التحديد ثلاثية الأبعاد المعتمدة سريرياً (Binary Ground Truth Masks).
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
