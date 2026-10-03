import shutil
from pathlib import Path

from PIL import Image

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, compile_pdf
from app.services.table_latex import to_latex


def test_markdown_table_to_latex():
    raw = "| A | B |\n|---|---|\n| 1 | 2 |"
    latex = to_latex(raw)
    assert "\\begin{tabular}" in latex
    assert "1 & 2" in latex


def test_renderer_writes_standard_document_with_original_figure(tmp_path):
    fig = tmp_path / "figure.png"
    Image.new("RGB", (100, 100), "white").save(fig)
    page = PageLayout(0, 1000, 1400, str(fig))
    page.units = [
        ProcessingUnit("T", 0, "text", BBox(100,100,600,220), ["b"], .9, decoded="Hello & world", decoder="test", validation_score=1),
        ProcessingUnit("M", 0, "math", BBox(100,260,600,380), ["m"], .9, decoded=r"\\frac{a}{b}", decoder="test", validation_score=1),
        ProcessingUnit("F", 0, "figure", BBox(650,100,900,400), ["f"], .9, crop_path=str(fig), decoder="preserve", validation_score=1),
    ]
    tex = build_tex(DocumentLayout([page]), tmp_path / "out", "Test")
    content = tex.read_text()
    assert "\\put(" not in content
    assert "\\begin{picture}" not in content
    assert "margin=25mm" in content
    assert "Hello \\& world" in content
    assert "\\frac{a}{b}" in content
    assert "assets/p0000_F.png" in content


def test_pdf_compile_if_available(tmp_path):
    if not shutil.which("pdflatex"):
        return
    fig = tmp_path / "f.png"
    Image.new("RGB", (20,20), "white").save(fig)
    p = PageLayout(0,1000,1400,str(fig))
    p.units = [ProcessingUnit("T",0,"text",BBox(100,100,500,200),["b"],.9,decoded="Smoke",decoder="test",validation_score=1)]
    tex = build_tex(DocumentLayout([p]), tmp_path / "out", "Smoke")
    assert compile_pdf(tex)["ok"] is True
