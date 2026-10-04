"""Render a synthetic, public sample for visual verification of document flow.

Run with python -m scripts.preview_document. No OCR service or API key is used.
The optional PNG log export contains this synthetic sample only.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, compile_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".test-tmp-document-preview"))
    parser.add_argument("--emit-preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    drawing = args.output / "diagram.png"
    image = Image.new("RGB", (640, 200), "white")
    pen = ImageDraw.Draw(image)
    pen.rounded_rectangle((20, 40, 210, 160), radius=14, outline=(35, 75, 125), width=4)
    pen.rounded_rectangle((430, 40, 620, 160), radius=14, outline=(35, 75, 125), width=4)
    pen.line((235, 100, 395, 100), fill=(35, 75, 125), width=4)
    pen.polygon([(395, 100), (379, 90), (379, 110)], fill=(35, 75, 125))
    pen.text((95, 90), "INPUT", fill=(35, 75, 125))
    pen.text((500, 90), "OUTPUT", fill=(35, 75, 125))
    image.save(drawing)

    def unit(identifier: str, category: str, content: str, top: int, height: int = 40) -> ProcessingUnit:
        return ProcessingUnit(identifier, 0, category, BBox(100, top, 900, top + height),
                              [identifier], .95, decoded=content, decoder="synthetic", validation_score=1)

    figure = unit("diagram", "figure", "", 800, 160)
    figure.bbox = BBox(200, 800, 800, 960)
    figure.crop_path = str(drawing.resolve())
    page = PageLayout(0, 1000, 1400, "unused.png", units=[
        unit("heading", "text", "# Motion and forces", 20),
        unit("intro", "text", "The input scan supplies content and reading order.\n"
             "The result is a regular document with readable text, consistent margins and natural line wrapping.\n\n"
             "For a constant mass, $F=ma$. The expression $\\sqrt{3}$ remains an inline formula.", 100, 120),
        unit("math", "math", r"\begin{align*}F&=ma\\a&=\frac{F}{m}\end{align*}", 300, 80),
        unit("list", "text", "## Working notes\n- Interpret the text in reading order.\n"
             "- Keep formulas editable in LaTeX.\n- Preserve the original drawing as a local image.", 400, 100),
        unit("table", "table", "| Quantity | Value |\n|---|---|\n| Mass | 2 kg |\n| Force | 6 N |", 600, 100),
        figure,
        unit("end", "text", "The next paragraph follows the illustration. Extra content continues on later pages "
             "without being scaled to match a handwritten bounding box.", 1100, 80),
    ])
    page.units[2].quality_review = {"status": "uncertain", "issues": ["Synthetic ambiguous symbol"]}
    page.units[-1].quality_review = {"status": "not_reviewed", "mark_pdf": True}
    tex = build_tex(DocumentLayout([page]), args.output / "document", "Clean notes")
    compiled = compile_pdf(tex)
    if not compiled["ok"]:
        raise RuntimeError((tex.parent / "compile.log").read_text(encoding="utf-8") if (tex.parent / "compile.log").exists() else compiled["reason"])
    with fitz.open(tex.with_suffix(".pdf")) as document:
        for index, page in enumerate(document):
            png = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
            (tex.parent / f"page-{index + 1}.png").write_bytes(png)
            if args.emit_preview:
                print(f"HAND2TEX_PREVIEW:{index + 1}:" + base64.b64encode(png).decode("ascii"))
    print(f"Synthetic document compiled: {tex}")


if __name__ == "__main__":
    main()
