#!/usr/bin/env python3
"""
compile_code_explanation_pdf.py
===============================
Compiles code_explanation_template.html into Data_Audit_Code_Explanation.pdf
using Edge headless.
"""

import subprocess
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
PDF_OUTPUT_PATH = PROJECT_ROOT / "docs" / "Data_Audit_Code_Explanation.pdf"
HTML_TEMPLATE_PATH = _SCRIPT_DIR / "code_explanation_template.html"

def main():
    if not HTML_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Template not found: {HTML_TEMPLATE_PATH}")

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
        str(HTML_TEMPLATE_PATH.resolve().as_uri()),
    ]

    print(f"[INFO] Compiling {PDF_OUTPUT_PATH.name} via Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if PDF_OUTPUT_PATH.is_file() and PDF_OUTPUT_PATH.stat().st_size > 1000:
        print("=" * 65)
        print(f"[SUCCESS] Code Explanation PDF Generated Successfully!")
        print(f"File Path : {PDF_OUTPUT_PATH.resolve()}")
        print(f"File Size : {PDF_OUTPUT_PATH.stat().st_size:,} bytes")
        print("=" * 65)
    else:
        print(f"[ERROR] PDF generation failed with code {res.returncode}")
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    main()
