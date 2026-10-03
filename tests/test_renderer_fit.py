import re
import shutil

import fitz
import pytest
from PIL import Image

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import A4_H_MM, A4_W_MM, _text_box, build_tex, compile_pdf


def unit(identifier, text, bbox, category="text"):
    return ProcessingUnit(
        identifier, 0, category, bbox, [identifier], .95,
        decoded=text, decoder="test", validation_score=1,
    )


@pytest.mark.parametrize("formula", [
    r"$\sqrt{3}$",
    r"$$\sqrt{3}$$",
    r"\(\sqrt{3}\)",
    r"\[\sqrt{3}\]",
])
def test_text_keeps_explicit_math_and_escapes_only_prose(formula):
    box = _text_box(unit("mixed", f"Valore {formula} e testo & altro.", BBox(20, 40, 380, 180)), 400, 600)
    assert r"\sqrt{3}" in box
    assert r"\textbackslash{}sqrt" not in box
    assert "Valore " in box
    assert r"e testo \& altro." in box
    assert "{$" in box


@pytest.mark.parametrize("text,escaped", [
    ("Costo $5 e 20%", r"Costo \$5 e 20\%"),
    ("Saldo $", r"Saldo \$"),
    ("Prezzi $5 e $10", r"Prezzi \$5 e \$10"),
    ("Segno $$ incompleto", r"Segno \$\$ incompleto"),
    (r"Percorso C:\temp e parentesi \(", r"Percorso C:\textbackslash{}temp e parentesi \textbackslash{}("),
])
def test_currency_and_unmatched_math_delimiters_remain_literal_text(text, escaped):
    box = _text_box(unit("literal", text, BBox(20, 40, 380, 180)), 400, 600)
    assert escaped in box
    assert "{$" not in box


def test_escaped_dollar_inside_math_and_multiline_alignment_are_kept():
    text = (
        "Costo $\\text{cost \\$5}=x$ e formula "
        "\\[\\begin{align*}\nx&=1\\\\\ny&=2\n\\end{align*}\\]."
    )
    box = _text_box(unit("mixed", text, BBox(20, 40, 380, 180)), 400, 600)
    assert r"\text{cost \$5}=x" in box
    assert r"\begin{aligned}" in box and r"\end{aligned}" in box
    assert r"\textbackslash{}text" not in box
    assert r"\begin{align*}" not in box


def test_body_height_stops_before_next_overlapping_row_without_moving_box(tmp_path):
    first = unit("first", "First row", BBox(100, 100, 900, 500))
    second = unit("second", "Second row", BBox(120, 200, 880, 550))
    page = PageLayout(0, 1000, 1400, "unused.png", units=[first, second])
    document = build_tex(DocumentLayout([page]), tmp_path / "result", "Height limits").read_text(encoding="utf-8")
    first_box = next(line for line in document.splitlines() if "First row" in line)
    fit = re.search(r"\\HandFitText\{([\d.]+)mm\}\{([\d.]+)mm\}", first_box)
    height = float(fit.group(2))
    assert height < (second.bbox.y1 - first.bbox.y1) / 1400 * A4_H_MM
    assert first_box.startswith(_text_box(first, 1000, 1400).split(r"\HandFitText", 1)[0])
    assert r"\fontsize{15" not in document


def test_other_column_does_not_reduce_available_body_height(tmp_path):
    first = unit("first", "Left row", BBox(20, 100, 400, 500))
    right = unit("right", "Right row", BBox(600, 200, 980, 550))
    page = PageLayout(0, 1000, 1400, "unused.png", units=[first, right])
    document = build_tex(DocumentLayout([page]), tmp_path / "result", "Columns").read_text(encoding="utf-8")
    line = next(line for line in document.splitlines() if "Left row" in line)
    match = re.search(r"\\HandFitText\{([\d.]+)mm\}\{([\d.]+)mm\}", line)
    assert float(match.group(2)) == pytest.approx(400 / 1400 * A4_H_MM - .15, abs=.001)


def test_figure_keeps_original_dimensions_and_asset_bytes(tmp_path):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (20, 30), "orange").save(image_path)
    figure = unit("figure", "", BBox(100, 200, 600, 400), "figure")
    figure.crop_path = str(image_path)
    page = PageLayout(0, 1000, 1000, str(image_path), units=[figure])
    result_dir = tmp_path / "result"
    document = build_tex(DocumentLayout([page]), result_dir, "Preserved figure").read_text(encoding="utf-8")
    assert r"\includegraphics[width=105.000mm,height=59.400mm,keepaspectratio]" in document
    assert (result_dir / "assets" / image_path.name).read_bytes() == image_path.read_bytes()


def compile_page(tmp_path, units, page_size=(1000, 1400)):
    page = PageLayout(0, *page_size, "unused.png", units=units)
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Fitted text")
    result = compile_pdf(tex_path)
    log = (tex_path.parent / "compile.log").read_text(encoding="utf-8")
    assert result["ok"], log
    assert "Overfull" not in log, log
    assert "size substitutions" not in log.lower(), log
    return tex_path.with_suffix(".pdf")


def expected_pdf_box(bbox, page_size):
    width, height = page_size
    return fitz.Rect(
        bbox.x1 / width * A4_W_MM * 72 / 25.4,
        bbox.y1 / height * A4_H_MM * 72 / 25.4,
        bbox.x2 / width * A4_W_MM * 72 / 25.4,
        min(A4_H_MM, bbox.y2 / height * A4_H_MM) * 72 / 25.4,
    )


def assert_ink_inside(page, box, clip=None, tolerance=1):
    # Extracted word boxes describe font metrics, not visible glyph outlines.
    # In particular, TeX's radical font has a displaced PDF text origin. Check
    # the rendered ink at 4x resolution while testing extracted content below.
    scale = 4
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, clip=clip)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    bounds = image.point(lambda value: 255 if value < 224 else 0).getbbox()
    assert bounds, "No visible text in the compiled PDF"
    ink = fitz.Rect(
        (pixmap.x + bounds[0]) / scale, (pixmap.y + bounds[1]) / scale,
        (pixmap.x + bounds[2]) / scale, (pixmap.y + bounds[3]) / scale,
    )
    assert ink.x0 >= box.x0 - tolerance, (ink, box)
    assert ink.y0 >= box.y0 - tolerance, (ink, box)
    assert ink.x1 <= box.x1 + tolerance, (ink, box)
    assert ink.y1 <= box.y1 + tolerance, (ink, box)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
@pytest.mark.parametrize("bbox,text", [
    (BBox(100, 100, 220, 500), "ALPHA " + "measured words occupy this narrow column " * 12 + "OMEGA"),
    (BBox(100, 100, 700, 150), "ALPHA " + "many words need vertical fitting " * 12 + "OMEGA"),
    (BBox(100, 1330, 700, 1398), "ALPHA " + "bottom page words " * 10 + "OMEGA"),
])
def test_real_pdf_long_or_narrow_text_stays_inside_original_box(tmp_path, bbox, text):
    pdf_path = compile_page(tmp_path, [unit("paragraph", text, bbox)])
    with fitz.open(pdf_path) as document:
        assert len(document) == 1
        words = document[0].get_text("words")
        assert any(word[4] == "ALPHA" for word in words)
        assert any(word[4] == "OMEGA" for word in words)
        assert_ink_inside(document[0], expected_pdf_box(bbox, (1000, 1400)))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_overlapping_envelopes_do_not_overlap_rendered_words(tmp_path):
    page_size = (1663, 2420)
    first = unit("first", "ALPHA " * 35, BBox(65, 0, 1511, 148))
    second = unit("second", "BETA " * 35, BBox(78, 50, 1511, 202))
    pdf_path = compile_page(tmp_path, [first, second], page_size)
    with fitz.open(pdf_path) as document:
        words = document[0].get_text("words")
        first_words = [word for word in words if word[4] == "ALPHA"]
        second_words = [word for word in words if word[4] == "BETA"]
        assert len(first_words) == 35 and len(second_words) == 35
        first_box = expected_pdf_box(first.bbox, page_size)
        first_box.y1 = second.bbox.y1 / page_size[1] * A4_H_MM * 72 / 25.4
        assert max(word[3] for word in first_words) < min(word[1] for word in second_words)
        separation = (max(word[3] for word in first_words) + min(word[1] for word in second_words)) / 2
        page = document[0]
        assert_ink_inside(page, first_box, fitz.Rect(0, 0, page.rect.width, separation))
        assert_ink_inside(page, expected_pdf_box(second.bbox, page_size),
                          fitz.Rect(0, separation, page.rect.width, page.rect.height))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_narrow_wrapped_radical_renders_without_literal_latex_or_overflow(tmp_path):
    bbox = BBox(1012, 735, 1077, 786)
    pdf_path = compile_page(tmp_path, [unit("radical", r"$\sqrt{3}$", bbox)], (1663, 2420))
    with fitz.open(pdf_path) as document:
        text = document[0].get_text()
        assert "sqrt" not in text and "$" not in text
        assert "3" in text
        assert_ink_inside(document[0], expected_pdf_box(bbox, (1663, 2420)))
