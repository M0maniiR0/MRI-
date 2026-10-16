#!/usr/bin/env python3
"""
generate_technical_report_pdf.py
================================
Compiles report_template.html into Medical_AI_Glioma_Technical_Report.pdf
using Edge headless.
"""

import base64
import subprocess
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
PDF_OUTPUT_PATH = PROJECT_ROOT / "docs" / "Medical_AI_Glioma_Technical_Report.pdf"
HTML_TEMPLATE_PATH = _SCRIPT_DIR / "report_template.html"
HTML_TEMP_PATH = _SCRIPT_DIR / "report_temp.html"
QC_IMAGE_PATH = _SCRIPT_DIR / "qc_slice_BRATS_457.png"

def main():
    if not HTML_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Template not found: {HTML_TEMPLATE_PATH}")

    # 1. Base64 encode the QC preview image
    img_b64 = ""
    if QC_IMAGE_PATH.is_file():
        with open(QC_IMAGE_PATH, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"[OK] Encoded QC image ({len(img_b64)} chars)")

    # 2. Read template and inject image
    raw_html = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    final_html = raw_html.replace("__IMG_B64__", img_b64)

    HTML_TEMP_PATH.write_text(final_html, encoding="utf-8")
    print(f"[OK] Rendered temporary HTML -> {HTML_TEMP_PATH.name}")

    # 3. Locate Edge executable
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
        raise FileNotFoundError("Microsoft Edge executable not found for headless PDF generation.")

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUTPUT_PATH.resolve()}",
        str(HTML_TEMP_PATH.resolve().as_uri()),
    ]

    print(f"[INFO] Generating PDF via {edge_exe}...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if PDF_OUTPUT_PATH.is_file() and PDF_OUTPUT_PATH.stat().st_size > 1000:
        print("=" * 65)
        print(f"[SUCCESS] PDF Generated Successfully!")
        print(f"File Path : {PDF_OUTPUT_PATH.resolve()}")
        print(f"File Size : {PDF_OUTPUT_PATH.stat().st_size:,} bytes")
        print("=" * 65)
        if HTML_TEMP_PATH.is_file():
            HTML_TEMP_PATH.unlink()
    else:
        print(f"[ERROR] PDF generation failed with code {res.returncode}")
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    main()
