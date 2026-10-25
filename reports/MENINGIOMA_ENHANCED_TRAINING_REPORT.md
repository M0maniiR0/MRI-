# 🔬 NeuroScope AI — Meningioma (الورم السحائي) Enhanced Training & Algorithm Report
**Project:** BrainAgenticthon 2026 — Medical AI Workstream  
**Disease Target:** Meningioma Solid Tumor Core Segmentation (BraTS-MEN 2023)  
**Date:** October 25, 2026  
**Status:** **ENHANCED SPECIFICATION & ROOT CAUSE RESOLUTION AUDIT**

---

## 1. الطبيعة السريرية والفيزيائية للمرض (Clinical & Pathological Profile)
* **الوصف السريري:** الأورام السحائية (Meningiomas) هي أورام تنشأ من الأغشية المحيطة بالدماغ (السحايا - Dura Mater) وتكون خارج نسيج الدماغ (Extra-axial). تتسم بأنها كتل ملساء ومحددة بدقة ولها ذيل سحائي (Dural Tail) يمتص صبغة الجادولينيوم بشدة وتجانس كامل.
* **التحديات السريرية والتشخيصية:**
  1. قد تلتبس حدود الورم مع عظام الجمجمة الملاصقة لها أو الجيوب الوريدية الجافية (Dural Venous Sinuses) إذا لم تكن المعالجة المسبقة دقيقة.
  2. وجود أورام سحائية صفيحية مسطحة ومتناهية الصغر (Sub-centimeter en-plaque lesions) يقل حجمها عن $1,000\text{ mm}^3$، مما يسبب عدم توازن طبقي شديد (Class Imbalance).

---

## 2. قنوات الرنين المغناطيسي وتثبيت الترتيب (Input Modalities & Channel Order)
تم تثبيت الترتيب الصحيح عالمياً في ملفات التدريب والاستدلال ليطابق ترتيب أوزان النموذج المدرب:
$$\text{Input Tensor} = [\text{t2f (FLAIR)}, \text{t1n (T1 native)}, \text{t1c (T1-CE)}, \text{t2w (T2)}]$$
* **قناة T1ce (Gadolinium Contrast):** تمثل القناة التشخيصية الأهم؛ لأن الورم يتشرب الصبغة بقوة وتظهر حدوده بوضوح ناصع.

---

## 3. المعالجة المسبقة والتصفية والتعزيز (Filtration & Enhancement Pipeline)
1. **التصفية وعزل عظام الجمجمة (Skull & Bone Filtration):**
   - تطبيق `PercentileClippingAndFiltrationd([0.5%, 99.5%])` لإخماد الشذوذ الضوئي في السحايا وعظام الجمجمة.
   - تفعيل `BrainMaskGuardFiltrationd` لمنع ارتشاح التنبؤ إلى تجويف الأنف أو عظام الجمجمة الخارجية.
2. **تعزيز صبغة T1ce (Targeted Contrast Stretching):**
   - تطبيق دالة تعزيز لا خطية ($\gamma = 1.25$) على القناة رقم 2 (T1ce) لرفع نسبة الإشارة إلى الضوضاء (SNR) للجسم الصلب للورم.
3. **المعايرة المكانية (Isotropic Spacing):**
   - توحيد الأبعاد إلى $1.0\text{ mm}^3$ والتوجيه القياسي `RAS`.

---

## 4. الخوارزمية وصياغة الهدف (Model Architecture & Target Formulation)
* **المعمارية:** **3D SegResNet** مع اتصالات تخطي ثلاثية الأبعاد.
* **صياغة الهدف كـ فئة ثنائية صلبة (Single Binary Solid Target):**
  - تم إلغاء التشتت متعدد القنوات، والتركيز على هدف نقي واحد: $\text{Meningioma Mass} = (\text{seg} > 0)$.
* **دالة الخسارة المعتمدة:**
  $$\mathcal{L} = \text{DiceLoss} + \text{FocalLoss}(\gamma=2.0)$$
  - دمج `FocalLoss` ذو أهمية قصوى للأورام السحائية الصغيرة والمسطحة (en-plaque) لأنه يخفف وزن البكسلات السهلة (خلايا الدماغ الطبيعية) ويركز تدرجات التعلم على البكسلات الصعبة للورم الصغير.
* **استراتيجية أخذ العينات:**
  - أخذ العينات بانحياز كبير للبؤر المعززة بالصبغة (`pos=3.0, neg=1.0`) في قطع بحجم $96 \times 96 \times 96$.

---

## 5. مقاييس التقييم السريرية (Clinical Metrics)
* **Dice Similarity Coefficient (DSC):** لقياس حجم التطابق للكتلة الصلبة.
* **Hausdorff Distance 95 (HD95):** للتأكد من ابتعاد حدود الاستئصال عن الشرايين الرئيسية والأعصاب القحفية المجاورة لقاعدة الجمجمة.
