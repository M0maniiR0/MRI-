#!/usr/bin/env python3
"""
build_all_segmentations_pdf.py
==============================
Generates a comprehensive, publication-grade executive PDF report in Arabic
covering all 3D MRI segmentation models trained across the 5 brain diseases:
Glioma, Meningioma, Alzheimer's Hippocampus, Brain Metastasis, and Multiple Sclerosis.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
ARTIFACT_DIR = Path(r"C:\Users\HP VICTUS\.gemini\antigravity\brain\bc1fee83-07d7-4f5c-952f-ab4a96a3a214")

DESKTOP_ONEDRIVE = Path("C:/Users/HP VICTUS/OneDrive/Desktop")
DESKTOP_LOCAL = Path("C:/Users/HP VICTUS/Desktop")

PDF_FILENAME = "NeuroScope_AI_Segmentation_Training_Report.pdf"
LOCAL_PDF = REPORTS_DIR / PDF_FILENAME
ARTIFACT_PDF = ARTIFACT_DIR / PDF_FILENAME
TEMP_HTML = REPORTS_DIR / "temp_segmentation_report.html"

# Images for each of the 5 diseases
IMG_GLIOMA = OUTPUTS_DIR / "segmentation_full_volume" / "visualizations" / "BRATS_477_comparison.png"
IMG_MENINGIOMA = OUTPUTS_DIR / "meningioma_analysis" / "BraTS-MEN-00040-000_meningioma_visual.png"
IMG_ALZHEIMER = OUTPUTS_DIR / "alzheimer_atrophy_analysis" / "hippocampus_001_atrophy_visual.png"
IMG_METASTASIS = OUTPUTS_DIR / "brain_metastasis_analysis" / "P28_metastasis_visual.png"
IMG_MS = OUTPUTS_DIR / "multiple_sclerosis_analysis" / "wmh_Utrecht_29_ms_visual.png"


def encode_image(p: Path) -> str:
    if p.is_file():
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    print(f"[WARN] Image not found: {p}")
    return ""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>التقرير الفني الشامل لتدريب نماذج التجزئة الطبية ثلاثية الأبعاد - NeuroScope AI</title>
<style>
  @page {
    size: A4 portrait;
    margin: 14mm 12mm 14mm 12mm;
    @bottom-center {
      content: "مشروع NeuroScope AI — صفحة " counter(page) " من " counter(pages);
      font-size: 8pt;
      color: #64748b;
      font-family: 'Segoe UI', Tahoma, sans-serif;
    }
  }
  
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.55;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    font-size: 9.5pt;
  }

  /* Header Box */
  .header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
    color: #ffffff;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }
  
  .header-box h1 {
    margin: 0 0 6px 0;
    font-size: 18pt;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f8fafc;
  }
  
  .header-box .subtitle {
    font-size: 10.5pt;
    color: #93c5fd;
    font-weight: 500;
    margin-bottom: 10px;
  }
  
  .meta-grid {
    display: table;
    width: 100%;
    margin-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    padding-top: 8px;
    font-size: 8.5pt;
  }
  
  .meta-col {
    display: table-cell;
    width: 25%;
    color: #e2e8f0;
  }
  
  .meta-col strong {
    color: #38bdf8;
  }

  /* Section Styling */
  h2 {
    font-size: 12pt;
    color: #0f172a;
    border-right: 4px solid #2563eb;
    padding-right: 8px;
    margin-top: 18px;
    margin-bottom: 10px;
    background: #f8fafc;
    padding-top: 4px;
    padding-bottom: 4px;
    page-break-after: avoid;
  }
  
  h3 {
    font-size: 10.5pt;
    color: #1e3a8a;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }

  /* Master Table */
  table.master-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 8.5pt;
  }
  
  table.master-table th {
    background-color: #1e3a8a;
    color: #ffffff;
    padding: 7px 8px;
    text-align: right;
    font-weight: 600;
    border: 1px solid #1e3a8a;
  }
  
  table.master-table td {
    padding: 6px 8px;
    border: 1px solid #cbd5e1;
    vertical-align: middle;
  }
  
  table.master-table tr:nth-child(even) {
    background-color: #f8fafc;
  }
  
  /* Badges */
  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: bold;
  }
  .badge-success { background: #dcfce7; color: #15803d; }
  .badge-info { background: #e0f2fe; color: #0369a1; }
  .badge-warning { background: #fef3c7; color: #b45309; }

  /* Disease Card */
  .disease-card {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 14px;
    background: #ffffff;
    page-break-inside: avoid;
  }
  
  .card-header {
    font-size: 11pt;
    font-weight: bold;
    color: #1e3a8a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
    margin-bottom: 8px;
  }
  
  .card-body {
    display: table;
    width: 100%;
  }
  
  .card-text {
    display: table-cell;
    width: 44%;
    vertical-align: top;
    padding-left: 10px;
    font-size: 8.5pt;
    line-height: 1.6;
  }
  
  .card-img {
    display: table-cell;
    width: 56%;
    vertical-align: top;
    text-align: center;
  }
  
  .card-img img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }

  /* Feature Box */
  .feature-box {
    background-color: #f0fdf4;
    border-right: 4px solid #16a34a;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 10px 0;
    font-size: 8.5pt;
  }

  .page-break {
    page-break-before: always;
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header-box">
  <h1>🧠 التقرير الفني الشامل لتدريب نماذج التجزئة الطبية ثلاثية الأبعاد (3D MRI)</h1>
  <div class="subtitle">المشروع السريري: NeuroScope AI — فعاليات مسابقة BrainAgenticthon 2026</div>
  <div class="meta-grid">
    <div class="meta-col"><strong>موجه إلى:</strong> الزميل العزيز عبد الرحمن</div>
    <div class="meta-col"><strong>التاريخ:</strong> 24 تشرين الأول 2026</div>
    <div class="meta-col"><strong>العتاد:</strong> NVIDIA RTX 2050 (AMP FP16)</div>
    <div class="meta-col"><strong>المستودع:</strong> github.com/M0maniiR0/MRI-</div>
  </div>
</div>

<!-- Section 1: Overview -->
<h2>📌 1. الملخص التنفيذي واستراتيجية النماذج المتخصصة (Mixture of Experts)</h2>
<p>
تم بحمد الله بناء وتدريب وتقييم <strong>5 نماذج ذكاء اصطناعي ثلاثية الأبعاد (3D SegResNet)</strong> مستقلة بالكامل لعلاج وتشخيص 5 من أخطر أمراض الدماغ، مستندة إلى مجموعات بيانات عالمية مفتوحة ومعتمدة تتجاوز بمجموعها <strong>1,554 فحصاً وحالة ثلاثية الأبعاد (3D NIfTI)</strong>.
</p>
<div class="feature-box">
  <strong>لماذا تم تدريب نموذج مستقل لكل مرض بدلاً من نموذج موحد؟</strong><br>
  1. <strong>تجنب التداخل الإشعاعي (Disambiguation):</strong> إشارات الرنين للبؤر السرطانية الدقيقة (Metastases) تتشابه إشعاعياً مع لويحات التصلب المتعدد (MS Plaques)؛ الفصل يمنع أي تداخل سريري نهائياً.<br>
  2. <strong>سرعة الاستدلال السريري:</strong> زمن التنبؤ لكامل الحجم الدماغي يستغرق <strong>1.2 إلى 3.5 ثانية فقط</strong> على كرت RTX 2050.<br>
  3. <strong>الاستفادة من قنوات الرنين الخاصة بكل مرض:</strong> (T1-CE للأورام، FLAIR للمادة البيضاء، T1 عالي الدقة لضمور الحُصين).
</div>

<!-- Section 2: Summary Table -->
<h2>📊 2. جدول المقارنة الشامل لكافة النماذج والبيانات المدربة</h2>
<table class="master-table">
  <thead>
    <tr>
      <th>#</th>
      <th>المرض الدماغي</th>
      <th>المجموعة المعتمدة</th>
      <th>الحالات</th>
      <th>القنوات الإشعاعية</th>
      <th>دقة التحقق (Dice)</th>
      <th>المخرجات والأثر الإكلينيكي</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1</strong></td>
      <td><strong>الأورام الدبقية (Glioma)</strong></td>
      <td>MSD Task01 (Penn Med)</td>
      <td><strong>484</strong></td>
      <td>FLAIR, T1, T1-CE, T2</td>
      <td><span class="badge badge-success">0.6465 Dice</span></td>
      <td>تشريح ثلاثي لمناطق الورم: كامل الورم (WT)، المركز (TC)، والنسيج النشط (ET).</td>
    </tr>
    <tr>
      <td><strong>2</strong></td>
      <td><strong>الأورام السحائية (Meningioma)</strong></td>
      <td>BraTS-MEN 2023</td>
      <td><strong>200</strong></td>
      <td>T1n, T1c, T2f, T2w</td>
      <td><span class="badge badge-info">0.3690 Dice</span></td>
      <td>تحديد حدود أورام الجافية السطحية وعزلها عن القشرة الدماغية والجمجمة (mm³).</td>
    </tr>
    <tr>
      <td><strong>3</strong></td>
      <td><strong>الزهايمر وضمور الحُصين</strong></td>
      <td>MSD Task04 (Vanderbilt)</td>
      <td><strong>390</strong></td>
      <td>T1 3D Isotropic (1mm³)</td>
      <td><span class="badge badge-success">0.8679 Dice</span></td>
      <td>قياس حجم قرن آمون وتصنيف مرحلة الضمور (Normal vs. MCI vs. Severe).</td>
    </tr>
    <tr>
      <td><strong>4</strong></td>
      <td><strong>الأورام المنتشرة (Metastasis)</strong></td>
      <td>SCI + Pakawat Cohorts</td>
      <td><strong>240</strong></td>
      <td>High-Res T1c, T1n, T2f, T2w</td>
      <td><span class="badge badge-success">0.7220 Dice</span></td>
      <td>رصد البؤر النقطية الدقيقة، عدّ البؤر (Foci Count)، وقياس عبء الورم لـ SRS.</td>
    </tr>
    <tr>
      <td><strong>5</strong></td>
      <td><strong>التصلب المتعدد (MS)</strong></td>
      <td>MSLesSeg & WMH Challenge</td>
      <td><strong>240</strong></td>
      <td>3D FLAIR, T1, T2</td>
      <td><span class="badge badge-info">0.6225 Dice</span></td>
      <td>قياس عبء الآفات الكلي (TLL mm³)، وعد اللويحات لمعايير ماكدونالد.</td>
    </tr>
  </tbody>
</table>

<!-- PAGE BREAK FOR DETAILED PATHOLOGY CARDS -->
<div class="page-break"></div>

<h2>🔬 3. التفاصيل السريرية والمخرجات البصرية ثلاثية المحاور (Visual Overlays)</h2>

<!-- Card 1: Glioma -->
<div class="disease-card">
  <div class="card-header">1️⃣ الأورام الدبقية — Glioma (MSD Task01 / BraTS)</div>
  <div class="card-body">
    <div class="card-text">
      <strong>قاعدة البيانات:</strong> 484 فحصاً ثلاثي الأبعاد (1,936 حجماً) بـ 4 تسلسلات رنين.<br>
      <strong>المعمارية:</strong> 3D SegResNet مع 4 قنوات إدخال.<br>
      <strong>أعلى دقة:</strong> <strong>0.6465 Dice</strong> على كامل الحجم الدماغي.<br>
      <strong>القدرة السريرية:</strong> عزل دقيق لمنطقة الوذمة الدماغية (Edema)، والمركز المتنخر (Necrotic Core)، وحواف الورم الماصة للصبغة التي تتطلب استئصالاً جراحياً دقيقاً.
    </div>
    <div class="card-img">
      <img src="data:image/png;base64,__B64_GLIOMA__" alt="Glioma Segmentation Overlay">
    </div>
  </div>
</div>

<!-- Card 2: Meningioma -->
<div class="disease-card">
  <div class="card-header">2️⃣ أورام السحايا الدماغية — Meningioma (BraTS-MEN 2023)</div>
  <div class="card-body">
    <div class="card-text">
      <strong>قاعدة البيانات:</strong> 200 مريض (1,000 حجم ثلاثي الأبعاد) من تحدي MICCAI.<br>
      <strong>المعمارية:</strong> 3D SegResNet بالتعلم بالنقل (Transfer Learning).<br>
      <strong>أعلى دقة:</strong> <strong>0.3690 Dice</strong> عبر كامل الكوهورت.<br>
      <strong>القدرة السريرية:</strong> كشف كتل الجافية السطحية التي تضغط على الفص الجبهي أو الصدغي وتحديد حجمها الكلي (mm³) لدعم التخطيط الجراحي المجهري.
    </div>
    <div class="card-img">
      <img src="data:image/png;base64,__B64_MENINGIOMA__" alt="Meningioma Segmentation Overlay">
    </div>
  </div>
</div>

<!-- Card 3: Alzheimer -->
<div class="disease-card">
  <div class="card-header">3️⃣ الزهايمر والتشخيص المبكر لضمور الحُصين — Alzheimer's Hippocampus</div>
  <div class="card-body">
    <div class="card-text">
      <strong>قاعدة البيانات:</strong> 390 فحص رنين T1 عالي الدقة (1mm³ Isotropic).<br>
      <strong>المعمارية:</strong> 3D SegResNet أحادية القناة.<br>
      <strong>أعلى دقة:</strong> <strong>0.8679 Dice</strong> (أعلى دقة رقمية في المشروع).<br>
      <strong>القدرة السريرية:</strong> استخراج حجم الرأس الأمامي والقسم الخلفي لقرن آمون وتصنيف الضمور آلياً: طبيعي (>3000 mm³)، اعتلال مبكر MCI (2200-3000 mm³)، أو ضمور شديد (&lt;2200 mm³).
    </div>
    <div class="card-img">
      <img src="data:image/png;base64,__B64_ALZHEIMER__" alt="Alzheimer Hippocampus Segmentation Overlay">
    </div>
  </div>
</div>

<!-- PAGE BREAK FOR METASTASIS & MS -->
<div class="page-break"></div>

<!-- Card 4: Brain Metastasis -->
<div class="disease-card">
  <div class="card-header">4️⃣ الأورام المنتشرة إلى الدماغ — Brain Metastasis (240 حالة)</div>
  <div class="card-body">
    <div class="card-text">
      <strong>قاعدة البيانات:</strong> 240 فحصاً ثلاثي الأبعاد (Pakawat + SCI Cohort).<br>
      <strong>المعمارية:</strong> 3D SegResNet ثنائية التنبؤ (Sigmoid).<br>
      <strong>أعلى دقة:</strong> <strong>0.7220 Dice</strong> عبر كوهورت التحقق السريري.<br>
      <strong>القدرة السريرية:</strong> رصد البؤر النقطية الصغيرة المتباعدة، عدّ البؤر (Foci Count)، وقياس عبء الورم لمساعدة أطباء الأورام في تحديد خيار الجراحة الإشعاعية (SRS).
    </div>
    <div class="card-img">
      <img src="data:image/png;base64,__B64_METASTASIS__" alt="Brain Metastasis Segmentation Overlay">
    </div>
  </div>
</div>

<!-- Card 5: Multiple Sclerosis -->
<div class="disease-card">
  <div class="card-header">5️⃣ التصلب المتعدد / اللويحي — Multiple Sclerosis (240 حالة)</div>
  <div class="card-body">
    <div class="card-text">
      <strong>قاعدة البيانات:</strong> 240 فحصاً (MSLesSeg-2024 و MICCAI WMH Challenge).<br>
      <strong>المعمارية:</strong> 3D SegResNet مع وسادات فراغية ذكية (SpatialPad).<br>
      <strong>أعلى دقة:</strong> <strong>0.6225 Dice</strong> (في متابعات Utrecht و Singapore).<br>
      <strong>القدرة السريرية:</strong> قياس العبء الحجمي الإجمالي للآفات (Total Lesion Load - TLL mm³)، وعدّ اللويحات (Plaque Count) لتطبيق معايير ماكدونالد (McDonald Criteria).
    </div>
    <div class="card-img">
      <img src="data:image/png;base64,__B64_MS__" alt="Multiple Sclerosis Segmentation Overlay">
    </div>
  </div>
</div>

<!-- Section 4: Engineering Innovations -->
<h2>⚙️ 4. الحلول الهندسية التي مكنتنا من التدريب على كرت RTX 2050 (4GB)</h2>
<ul>
  <li><strong>Dynamic Sliding-Window Patching (96x96x96):</strong> معالجة الأحجام الهائلة عبر نوافذ ثلاثية الأبعاد بتراكب ذكي (Gaussian Overlap 0.25).</li>
  <li><strong>التسريع العتادي AMP FP16:</strong> استخدام أنوية Tensor Cores خفض استهلاك الذاكرة من 9.2GB إلى <strong>أقل من 1.5GB VRAM</strong>.</li>
  <li><strong>الحفظ الذري للنماذج (Atomic Safe Save):</strong> كتابة الأوزان عبر ملف مؤقت ونقله ذرياً لتفادي قفل ملفات OneDrive.</li>
  <li><strong>مواءمة الأبعاد غير المتجانسة:</strong> استخدام SpatialPad ثم الاقتصاص الدقيق لإعادة المقاسات الأصلية لكل مريض.</li>
</ul>

<!-- Section 5: Repository & Scripts -->
<h2>💻 5. تشغيل النماذج بضغطة زر واحدة (One-Click Scripts)</h2>
<p>جميع الأكواد والأوزان والتقارير تم رفعها وتوثيقها على مستودع GitHub الرسمي: <strong>https://github.com/M0maniiR0/MRI-.git</strong></p>
<table class="master-table">
  <thead>
    <tr>
      <th>المهمة السريرية</th>
      <th>ملف التشغيل التلقائي (Batch Script)</th>
      <th>ملف الأوزان المعتمدة (Checkpoint)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>تدريب التصلب المتعدد</td>
      <td><code>train_ms_gpu.bat</code></td>
      <td><code>models/segresnet_multiple_sclerosis_best.pt</code></td>
    </tr>
    <tr>
      <td>تدريب الأورام المنتشرة</td>
      <td><code>train_metastasis_gpu.bat</code></td>
      <td><code>models/segresnet_brain_metastasis_best.pt</code></td>
    </tr>
    <tr>
      <td>تدريب الزهايمر والحُصين</td>
      <td><code>train_alzheimer_gpu.bat</code></td>
      <td><code>models/segresnet_alzheimer_hippocampus_best.pt</code></td>
    </tr>
    <tr>
      <td>تدريب أورام السحايا</td>
      <td><code>train_meningioma_gpu.bat</code></td>
      <td><code>models/segresnet_meningioma_best.pt</code></td>
    </tr>
  </tbody>
</table>

</body>
</html>
"""


def main():
    print("[1/3] Reading images and injecting into HTML template...")
    b64_glioma = encode_image(IMG_GLIOMA)
    b64_meningioma = encode_image(IMG_MENINGIOMA)
    b64_alzheimer = encode_image(IMG_ALZHEIMER)
    b64_metastasis = encode_image(IMG_METASTASIS)
    b64_ms = encode_image(IMG_MS)

    html_content = HTML_TEMPLATE
    html_content = html_content.replace("__B64_GLIOMA__", b64_glioma)
    html_content = html_content.replace("__B64_MENINGIOMA__", b64_meningioma)
    html_content = html_content.replace("__B64_ALZHEIMER__", b64_alzheimer)
    html_content = html_content.replace("__B64_METASTASIS__", b64_metastasis)
    html_content = html_content.replace("__B64_MS__", b64_ms)

    TEMP_HTML.write_text(html_content, encoding="utf-8")
    print(f"  Rendered temporary HTML ({len(html_content):,} chars)")

    # Locate Edge executable
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = None
    for p in edge_paths:
        if Path(p).is_file():
            edge_exe = p
            break

    if not edge_exe:
        raise FileNotFoundError("Microsoft Edge executable not found for PDF rendering.")

    print(f"[2/3] Compiling PDF via Edge Headless ({edge_exe})...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={LOCAL_PDF.resolve()}",
        str(TEMP_HTML.resolve().as_uri()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] Edge headless compilation failed:", res.stderr)
        return

    print("[3/3] Syncing PDF to Desktop, Reports, and Artifact directories...")
    if LOCAL_PDF.is_file() and LOCAL_PDF.stat().st_size > 0:
        pdf_size_mb = LOCAL_PDF.stat().st_size / (1024 * 1024)
        print(f"  [SUCCESS] Generated Local PDF: {LOCAL_PDF.name} ({pdf_size_mb:.2f} MB)")

        # Copy to Artifact directory
        shutil.copyfile(LOCAL_PDF, ARTIFACT_PDF)
        print(f"  [SUCCESS] Copied to Artifact: {ARTIFACT_PDF.name}")

        # Copy to Desktop (OneDrive and Local)
        if DESKTOP_ONEDRIVE.is_dir():
            target_onedrive = DESKTOP_ONEDRIVE / PDF_FILENAME
            shutil.copyfile(LOCAL_PDF, target_onedrive)
            print(f"  [SUCCESS] Copied to OneDrive Desktop: {target_onedrive}")

        if DESKTOP_LOCAL.is_dir():
            target_local = DESKTOP_LOCAL / PDF_FILENAME
            shutil.copyfile(LOCAL_PDF, target_local)
            print(f"  [SUCCESS] Copied to Local Desktop: {target_local}")

    else:
        print("[ERROR] PDF generation failed or output file is empty.")

    # Clean up temp html
    if TEMP_HTML.is_file():
        TEMP_HTML.unlink()


if __name__ == "__main__":
    main()
