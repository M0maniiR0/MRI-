#!/usr/bin/env python3
"""
generate_verification_pdf.py
============================
Compiles verification_template.html into docs/Model_Training_Verification_7Checks.pdf
via Edge headless.
"""

import base64
import subprocess
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
PDF_OUT = PROJECT_ROOT / "docs" / "Model_Training_Verification_7Checks.pdf"
HTML_TEMPLATE = _SCRIPT_DIR / "verification_template.html"
HTML_TEMP = _SCRIPT_DIR / "verification_report_temp.html"
COMP_IMG_PATH = PROJECT_ROOT / "reports" / "pred_vs_gt_comparison.png"

def main():
    if not HTML_TEMPLATE.is_file():
        raise FileNotFoundError(f"Template not found: {HTML_TEMPLATE}")

    # 1. Base64 encode the comparison image
    img_b64 = ""
    if COMP_IMG_PATH.is_file():
        with open(COMP_IMG_PATH, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"[OK] Encoded comparison image ({len(img_b64)} chars)")

    # 2. Inject into HTML
    raw_html = HTML_TEMPLATE.read_text(encoding="utf-8")
    final_html = raw_html.replace("__IMG_B64__", img_b64)
    HTML_TEMP.write_text(final_html, encoding="utf-8")
    print(f"[OK] Wrote temporary HTML -> {HTML_TEMP.name}")

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
        raise FileNotFoundError("Edge executable not found for PDF generation.")

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT.resolve()}",
        str(HTML_TEMP.resolve().as_uri()),
    ]

    print(f"[INFO] Compiling {PDF_OUT.name} via Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if PDF_OUT.is_file() and PDF_OUT.stat().st_size > 1000:
        print("=" * 65)
        print(f"[SUCCESS] Verification PDF Generated Successfully!")
        print(f"File Path : {PDF_OUT.resolve()}")
        print(f"File Size : {PDF_OUT.stat().st_size:,} bytes")
        print("=" * 65)
        if HTML_TEMP.is_file():
            HTML_TEMP.unlink()
    else:
        print(f"[ERROR] PDF generation failed with code {res.returncode}")
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    main()
