#!/usr/bin/env python3
"""
build_master_pdf.py
===================
Compiles reports/master_guide_template.html into:
1. reports/Medical_AI_Segmentation_Master_Guide.pdf
2. <ArtifactDir>/Medical_AI_Segmentation_Master_Guide.pdf
using Microsoft Edge headless.
"""

import base64
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "reports" / "master_guide_template.html"
HTML_TEMP_PATH = SCRIPT_DIR / "reports" / "master_guide_temp.html"
PDF_LOCAL_PATH = SCRIPT_DIR / "reports" / "Medical_AI_Segmentation_Master_Guide.pdf"
ARTIFACT_DIR = Path(r"C:\Users\HP VICTUS\.gemini\antigravity\brain\bc1fee83-07d7-4f5c-952f-ab4a96a3a214")
PDF_ARTIFACT_PATH = ARTIFACT_DIR / "Medical_AI_Segmentation_Master_Guide.pdf"

# Images
VIS_DIR = SCRIPT_DIR / "outputs" / "segmentation_full_volume" / "visualizations"
IMG_477 = VIS_DIR / "BRATS_477_comparison.png"
IMG_100 = VIS_DIR / "BRATS_100_comparison.png"
IMG_002 = VIS_DIR / "BRATS_002_comparison.png"


def encode_image(p: Path) -> str:
    if p.is_file():
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    return ""


def main():
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Template missing: {TEMPLATE_PATH}")

    print("[1/4] Reading images and encoding to Base64...")
    b64_477 = encode_image(IMG_477)
    b64_100 = encode_image(IMG_100)
    b64_002 = encode_image(IMG_002)
    print(f"  Encoded 477: {len(b64_477)} chars | 100: {len(b64_100)} chars | 002: {len(b64_002)} chars")

    print("[2/4] Injecting Base64 images into HTML template...")
    html_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    html_text = html_text.replace("__B64_477__", b64_477)
    html_text = html_text.replace("__B64_100__", b64_100)
    html_text = html_text.replace("__B64_002__", b64_002)

    HTML_TEMP_PATH.write_text(html_text, encoding="utf-8")
    print(f"  Rendered temp HTML -> {HTML_TEMP_PATH.name} ({len(html_text):,} chars)")

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

    print(f"[3/4] Compiling PDF via Edge Headless ({edge_exe})...")
    PDF_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_LOCAL_PATH.resolve()}",
        str(HTML_TEMP_PATH.resolve().as_uri()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] Edge headless failed:", res.stderr)
        return

    print("[4/4] Verifying generated PDF and syncing to Artifact Directory...")
    if PDF_LOCAL_PATH.is_file() and PDF_LOCAL_PATH.stat().st_size > 0:
        shutil.copyfile(PDF_LOCAL_PATH, PDF_ARTIFACT_PATH)
        print(f"[SUCCESS] PDF generated and verified successfully!")
        print(f"  Local PDF Path    : {PDF_LOCAL_PATH} ({PDF_LOCAL_PATH.stat().st_size:,} bytes)")
        print(f"  Artifact PDF Path : {PDF_ARTIFACT_PATH} ({PDF_ARTIFACT_PATH.stat().st_size:,} bytes)")
    else:
        print("[ERROR] PDF generation failed or output file is empty.")


if __name__ == "__main__":
    main()
