# 🧠 NeuroScope AI — Master Medical Segmentation Redesign & Retraining Strategy
**Project:** BrainAgenticthon 2026 — Medical AI Workstream  
**Author:** Senior Medical Imaging AI Engineer  
**Date:** October 25, 2026  
**Status:** **ACTIVE SPECIFICATION & ALGORITHM BLUEPRINT**

---

## 📋 1. Executive Summary & Clinical Directive

Following rigorous evaluation of the initial baseline models and direct medical directives from the supervising consultant/radiologist:
1. **Scope of Retraining (4 Diseases):**
   - **Glioma (ورم الدبقيات):** 3D multi-region segmentation across 4 MRI sequences.
   - **Meningioma (الورم السحائي):** Extra-axial solid tumor core and dural mass segmentation.
   - **Brain Metastasis (انبثاثات الدماغ):** Multi-focal punctate lesion detection without spatial downsampling.
   - **Hippocampus (قرن آمون - الزهايمر):** High-precision medial temporal anatomical volumetry.
2. **Permanent Exclusion of Multiple Sclerosis (التصلب المتعدد):**
   - **Clinical Justification:** The MS model exhibited severe overfitting on white matter hyperintensities (WMH) and hallucinated non-existent lesions in healthy brain tissue. It is strictly removed from the competitive retraining track.
3. **Mandatory Preprocessing Upgrades (Filtration & Enhancement):**
   - **Filtration (التصفية والتنقية):** Percentile intensity clipping ([0.5%, 99.5%]) to eliminate scanner Rician noise and spike artifacts, combined with an intracranial Brain Mask Guard to eliminate skull and air leakage.
   - **Enhancement (تحسين التباين والمعايرة):** Non-zero brain tissue Z-score standardization and targeted contrast stretching on Gadolinium-enhanced T1ce sequences.

```
خاصية المرض السريرية → شكلها بصور الرنين (MRI) → التحدي الهندسي للموديل → قرار التصميم المعتمد
```

---

## 🔬 2. Anatomy & Disease Characteristics: How Clinical Features Dictate Training Design

### أ) الوظائف الدماغية وموقع الإصابة وتأثيرها السريري:
| المنطقة الدماغية | الوظيفة الطبيعية | التأثير والأعراض عند انضغاطها أو تدميرها بالورم |
| :--- | :--- | :--- |
| **الفص الجبهي (Frontal Lobe)** | اتخاذ القرار، الشخصية، التحكم بالسلوك، الحركة الإرادية (Motor Cortex)، مركز الكلام (Broca). | تغيّر الشخصية، اللامبالاة، الاندفاع، ضعف أو شلل في جانب من الجسم، صعوبة في التعبير بالكلام. |
| **الفص الجداري (Parietal Lobe)** | الإحساس، إدراك المكان، تنسيق اليد والعين، القراءة والحساب. | إهمال جهة من الجسم (Spatial Neglect)، ضياع مكاني، صعوبة بالقراءة والحساب. |
| **الفص الصدغي (Temporal Lobe)** | الذاكرة، السمع، فهم اللغة (Wernicke)، المشاعر والعواطف. | نسيان، نوبات صرع فص صدغي، صعوبة في فهم كلام الآخرين. |
| **الفص القفوي (Occipital Lobe)** | معالجة الرؤية البصرية. | فقدان جزء من المجال البصري، هلوسات بصرية. |
| **المخيخ (Cerebellum)** | التوازن وتنسيق وتناغم الحركة. | ترنح في المشي (Ataxia)، رجفة بالحركة. |
| **جذع الدماغ (Brainstem)** | التنفس، النبض، ضغط الدم، الوعي، مسارات الأعصاب القحفية. | خطر مباشر وفوري على الحياة، اختلال العلامات الحيوية. |

---

### ب) قرن آمون (Hippocampus) والتمييز بين أنواع الخرف (الألزهايمر):
* **الزهايمر النموذجي (Typical Alzheimer's):** يبدأ الضمور من **قرن آمون (Hippocampus)** في الفص الصدغي الإنسي (Medial Temporal Lobe) ثم ينتشر إلى الفص الجداري. العَرَض الأول هو **فقدان الذاكرة الحديثة**؛ لذا فإن قياس حجمه (Volumetry) بدقة المليمتر هو المؤشر السريري الحاسم.
* **الخرف الجبهي الصدغي (FTD):** يبدأ بضمور الفص الأمامي (Frontal)؛ والعَرَض الأول هو تغيّر السلوك والشخصية وليس الذاكرة.
* **ضمور القشرة الخلفية (PCA):** يبدأ بالفص القفوي والجداري الخلفي بمشاكل في الرؤية وإدراك المسافات.

---

## ⚙️ 3. Comparative Matrix: The 4 Target Models

| المعيار | 1. ورم الدبقيات (Glioma) | 2. الورم السحائي (Meningioma) | 3. انبثاثات الدماغ (Brain Metastasis) | 4. قرن آمون (Hippocampus) |
| :--- | :--- | :--- | :--- | :--- |
| **طبيعة المرض** | ورم ارتشاحي متداخل، حدود مبهمة | كتلة صلبة ملساء متصلة بالسحايا | بؤر دقيقة متعددة مبعثرة | بنية تشريحية سليمة ثنائية الجانب |
| **قنوات الرنين (Inputs)** | 4 قنوات `[FLAIR, T1, T1ce, T2]` | 4 قنوات `[FLAIR, T1, T1ce, T2]` | 4 قنوات (التركيز الحاسم على `T1ce`) | قناة واحدة عالية الدقة `T1` |
| **بنية الأقنعة (Labels)** | Multi-Region (WT, TC, ET) | Single Binary Class (Solid Core) | Single Binary Class (Disjoint Lesions) | Single Anatomical Class (Bilateral) |
| **خوارزمية المعمارية** | 3D SegResNet with Skip Connections | 3D SegResNet with Skip Connections | 3D SegResNet (Full Native Resolution) | 3D SegResNet with Medial ROI Crop |
| **دالة الخسارة (Loss)** | `DiceCELoss(sigmoid=True)` | `DiceFocalLoss(gamma=2.0)` / `DiceCE` | `FocalTverskyLoss` / `DiceFocalLoss` | `DiceCELoss(sigmoid=True)` |
| **معالجة الصور والفلترة** | تصفية الضوضاء + Brain Mask Guard | تعزيز صبغة T1ce + حماية عظام الجمجمة | **ممنوع الـ Downsampling** + تعزيز الحواف | توحيد الدقة المكانية $1.0\text{ mm}^3$ |
| **استراتيجية الـ Patches** | $96 \times 96 \times 96$ (`pos=2.0, neg=1.0`) | $96 \times 96 \times 96$ (`pos=3.0, neg=1.0`) | $96 \times 96 \times 96$ (`pos=4.0, neg=1.0`) | $48 \times 64 \times 48$ حول الفص الصدغي |
| **مقياس التقييم الأساسي** | Dice لكل منطقة + HD95 | Dice + HD95 للحدود الجراحية | Lesion-wise Recall + False Positives | Dice + خطأ قياس الحجم (Volume Error) |

---

## 🛡️ 4. Medical MRI Filtration & Enhancement Protocol

تطبيقاً لتوجيهات الدكتورة الاستشارية، تم بناء موديول المعالجة الطبية [`mri_filtration_and_enhancement.py`](file:///c:/Users/HP%20VICTUS/OneDrive/Desktop/MRI%20Data/mri_filtration_and_enhancement.py):

### 1) خطوة التصفية (Filtration & Denoising):
* **Percentile Intensity Clipping:** قص الشدة بين الشريحة المئوية 0.5% و 99.5% لكل قناة داخل نسيج الدماغ لإزالة الشذوذ الناتج عن أجهزة الرنين (Rician noise tails & scanner spikes).
* **Brain Mask Guard:** إنشاء قناع تشريحي للدماغ وعزل الفوكسلات خارج الدماغ تماماً (Zero out air, skull, and orbit voxels) لمنع أي تسريب للتنبؤ خارج الجمجمة.

### 2) خطوة التحسين (Enhancement & Contrast Calibration):
* **Non-Zero Z-Score Standardization:** معايرة الشدة لكل مريض بشكل مستقل بحيث يكون المتوسط 0 والانحراف المعياري 1 داخل خلايا الدماغ فقط.
* **T1ce Contrast Stretching:** تطبيق دالة تحويل لا خطية ($\gamma = 1.15\text{--}1.25$) على قناة T1ce لزيادة التباين البصري والرياضي بين الورم النشط والأنسجة السليمة.
* **Isotropic Resampling:** توحيد دقة الفوكسل لتكون $1.0 \times 1.0 \times 1.0\text{ mm}^3$ وتوجيه الصورة للنمط العصبي القياسي `RAS`.

---

## 🚫 5. Multiple Sclerosis (التصلب المتعدد): توثيق أسباب الاستبعاد

تم استبعاد نموذج التصلب المتعدد نهائياً من خطة التدريب الحالية استناداً إلى الأدلة التجريبية التالية:
1. **ظاهرة الـ Overfitting الحاد:** النموذج يعاني من التباس شديد بين آفات التصلب المتعدد الحقيقية وبين التغيرات الإقفارية الدقيقة للوعاء المجهري (Microvascular ischemic changes / WMH) الشائعة مع التقدم بالسن.
2. **توليد أورام وآفات وهمية (Hallucinating Lesions):** في الفحص الشامل الأخير عبر `unified_inference.py`، قام النموذج برسم آفات ورمية في مناطق المادة البيضاء الطبيعية في مرضى سليمين بنسب خطأ تجاوزت 70% في بعض المقاطع.
3. **المحاذاة التحكيمية للمسابقة:** التصلب المتعدد ليس ضمن الأمراض الإلزامية للمسابقة، وتدريبه دون بيانات سريرية محكمة يؤدي إلى الإضرار بمصداقية النظام ككل.

---

## 🚀 6. خطة التنفيذ المعتمدة (Execution Roadmap)

1. **الترتيب والتنظيف:** أرشفة وحذف الملفات المؤقتة والزائدة في مستودع GitHub ليبقى المشروع نظيفاً ومهنياً.
2. **تنفيذ التدريب المحسن لكل مرض:**
   - Glioma: تشغيل `finetune_segresnet.py` المدعوم بـ AMP و $96^3$ crops و `DiceCELoss`.
   - Meningioma: تشغيل `train_meningioma_enhanced.py` بالهدف الثنائي الصلب وقنوات `[t2f, t1n, t1c, t2w]`.
   - Brain Metastasis: تشغيل `train_brain_metastasis_enhanced.py` بدقة كاملة و `FocalLoss`.
   - Hippocampus: تشغيل `train_hippocampus_enhanced.py` باقتصاص الفص الصدغي وقياس الحجم.
3. **تحديث تقارير الأداء والميتاداتا:** توليد التقارير النهائية وتحديث واجهة الاستدلال الموحدة `unified_inference.py`.
