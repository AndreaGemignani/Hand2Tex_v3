from __future__ import annotations

import shutil
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, compile_pdf

root = Path(tempfile.mkdtemp(prefix="hand2tex_smoke_"))
try:
    img = root / "figure.png"
    canvas = Image.new("RGB", (400, 220), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle((30, 40, 170, 160), outline="black", width=4)
    d.line((170, 100, 350, 100), fill="black", width=4)
    canvas.save(img)

    page = PageLayout(index=0, width_px=1000, height_px=1414, image_path=str(img))
    page.units = [
        ProcessingUnit("T1", 0, "text", BBox(70, 100, 580, 240), ["b1"], .95, decoded="Newton's second law", decoder="test", validation_score=1),
        ProcessingUnit("M1", 0, "math", BBox(70, 300, 580, 430), ["b2"], .95, decoded=r"F = ma", decoder="test", validation_score=1),
        ProcessingUnit("F1", 0, "figure", BBox(620, 100, 930, 480), ["b3"], .95, crop_path=str(img), decoder="test", validation_score=1),
    ]
    tex = build_tex(DocumentLayout([page]), root / "result", "Smoke Test")
    result = compile_pdf(tex)
    print(result)
    if shutil.which("pdflatex") and not result["ok"]:
        raise SystemExit(1)
finally:
    print(root)
