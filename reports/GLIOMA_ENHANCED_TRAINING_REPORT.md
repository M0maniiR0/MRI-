# 🔬 NeuroScope AI — Glioma (ورم الدبقيات) Enhanced Training & Algorithm Report
**Project:** BrainAgenticthon 2026 — Medical AI Workstream  
**Disease Target:** Glioma Multi-Region 3D Segmentation (BraTS / MSD Task01)  
**Date:** October 25, 2026  
**Status:** **ENHANCED SPECIFICATION & ALGORITHM AUDIT**

---

## 1. الطبيعة السريرية والفيزيائية للمرض (Clinical & Pathological Profile)
* **الوصف السريري:** أورام الدبقيات (Gliomas) هي أورام ارتشاحية تنشأ داخل نسيج الدماغ نفسه (Intra-axial). تتغلغل خلايا الورم بين خلايا المادة البيضاء الطبيعية، مما يجعل حواف الورم غير واضحة المعالم وتحيط بها وذمة وعائية ضخمة (Vasogenic Edema).
* **المناطق المرضية الثلاث المتداخلة (Sub-regions):**
  1. **الورم بأكمله (Whole Tumor - WT):** يشمل كافة التغيرات النسيجية (الوذمة + النواة المتنخرة + الجزء المعزز).
  2. **نواة الورم (Tumor Core - TC):** الكتلة الجراحية المركزية (النسيج غير المعزز + النخر + النسيج النشط).
  3. **الورم النشط المعزز (Enhancing Tumor - ET):** الجزء عالي التروية ذو الحاجز الدموي الدماغي المكسور الذي يمتص الصبغة بشدة.

---

## 2. قنوات الرنين المغناطيسي المستخدمة (Input Modalities & Sequences)
تم اعتماد التكديس الرباعي القياسي (4-Channel 3D Tensor):
1. **Channel 0 — FLAIR:** إبراز الوذمة المحيطة بالورم بدقة فائقة مع إخماد إشارة السائل الدماغي الشوكي (CSF).
2. **Channel 1 — T1 Native:** إظهار البنية التشريحية الأساسية للأنسجة الرمادية والبيضاء.
3. **Channel 2 — T1ce (Gadolinium Contrast):** إظهار النواة النشطة للورم وحدوده الجراحية بدقة عالية.
4. **Channel 3 — T2-weighted:** إظهار المحتوى المائي وكتلة الورم الكلية.

---

## 3. المعالجة المسبقة والتصفية والتعزيز (Filtration & Enhancement Pipeline)
تطبيقاً لتوجيهات الدكتورة، يمر كل فحص بالخطوات التالية:
1. **التصفية وإزالة الضوضاء (Percentile Noise Filtration):**
   - قص الشدة الشاذة بين `0.5%` و `99.5%` لإزالة ضوضاء رايس (Rician noise) والشذوذات الناتجة عن حركة المريض أو أجهزة التصوير.
2. **درع عزل الدماغ (Brain Mask Guard):**
   - تقييد جميع التنبؤات داخل حجم الدماغ المستمر؛ وتصفير الفوكسلات الخارجية (عظام الجمجمة، الهواء، الجيوب الأنفية) لمنع أي تسريب تنبؤي.
3. **التعزيز ومعايرة الشدة (Enhancement & Z-score):**
   - معايرة Z-Score لكل قناة بشكل مستقل على نسيج الدماغ الحي فقط (`nonzero=True, channel_wise=True`).
   - تعزيز التباين غير الخطي لقناة T1ce بقيمة $\gamma = 1.15$ لإبراز الحدود النشطة.
4. **توحيد الدقة الهندسية (Spatial Normalization):**
   - دقة مكانية متساوية الأبعاد $1.0 \times 1.0 \times 1.0\text{ mm}^3$ وتوجيه قياسي `RAS`.

---

## 4. الخوارزمية والمعمارية (Model Architecture & Algorithm)
* **المعمارية:** **3D SegResNet** (Encoder-Decoder مع وصلات تخطي Skip Connections وكتل Residual ثلاثية الأبعاد).
* **دالة الخسارة (Loss Function):**
  $$\mathcal{L} = \text{DiceLoss}(\text{pred}, \text{target}) + 1.0 \times \text{CrossEntropyLoss}(\text{pred}, \text{target})$$
  - يتم استخدام `DiceCELoss(sigmoid=True)` مع تصنيف Multi-Label للمناطق المتداخلة (WT, TC, ET).
  - دمج الـ Cross-Entropy يعاقب التنبؤات الخاطئة في الخلفية والمادة البيضاء السليمة ويقضي على التسريب الورمي السابق.
* **استراتيجية القطع (Receptive Field):**
  - استخدام قطع ثلاثية الأبعاد بحجم $96 \times 96 \times 96$ بدلاً من $64^3$ لمنح النموذج سياقاً تشريحياً كافياً للجمجمة والورم.
  - أخذ العينات بانحياز للمناطق الورمية بنسبة (`pos=2.0, neg=1.0`).

---

## 5. مقاييس التقييم السريرية (Clinical Metrics)
* **Dice Similarity Coefficient (DSC):** يقاس لكل منطقة على حدة (WT, TC, ET) لمعرفة دقة استئصال كل مكون ورمي.
* **Hausdorff Distance 95 (HD95):** لقياس أقصى خطأ عند حدود الورم، وهو المقياس الأكثر أهمية لجراح الأعصاب لتفادي ترك بقايا ورمية أو إيذاء أنسجة سليمة.
