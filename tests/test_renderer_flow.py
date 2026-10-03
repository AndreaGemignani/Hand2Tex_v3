import shutil
from pathlib import Path

import fitz
import pytest
from PIL import Image

from app.models import BBox, DocumentLayout, LayoutBlock, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, compile_pdf, render_text


def unit(identifier, text, bbox=None, category="text", page=0, members=None):
    return ProcessingUnit(identifier, page, category, bbox or BBox(100, 100, 900, 140),
                          members or [identifier], .95, decoded=text, decoder="test", validation_score=1)


@pytest.mark.parametrize("formula", [r"$\sqrt{3}$", r"$$\sqrt{3}$$", r"\(\sqrt{3}\)", r"\[\sqrt{3}\]"])
def test_text_keeps_explicit_math_and_escapes_only_prose(formula):
    content = render_text(f"Valore {formula} e testo & altro.")
    assert r"\sqrt{3}" in content
    assert r"\textbackslash{}sqrt" not in content
    assert "Valore " in content and r"e testo \& altro." in content


@pytest.mark.parametrize("text,escaped", [
    ("Costo $5 e 20%", r"Costo \$5 e 20\%"),
    ("Saldo $", r"Saldo \$"),
    ("Prezzi $5 e $10", r"Prezzi \$5 e \$10"),
    ("Segno $$ incompleto", r"Segno \$\$ incompleto"),
    (r"Percorso C:\temp e parentesi \(", r"Percorso C:\textbackslash{}temp e parentesi \textbackslash{}("),
])
def test_currency_and_unmatched_math_delimiters_remain_literal_text(text, escaped):
    assert render_text(text) == escaped


def test_multiline_math_is_not_rewrapped_as_prose():
    content = render_text("Costo $\\text{cost \\$5}=x$ e formula "
                          "\\[\\begin{align*}\nx&=1\\\\\ny&=2\n\\end{align*}\\].")
    assert r"\text{cost \$5}=x" in content
    assert r"\begin{aligned}" in content and r"\end{aligned}" in content
    assert "x&=1\\\\\ny&=2" in content


@pytest.mark.parametrize("formula", [r"$x=1$", r"$\frac{x}{2}=1$", r"$5 + x=1$"])
def test_currency_before_a_formula_does_not_consume_formula_delimiters(formula):
    content = render_text("Costo $5 e formula " + formula + ".")
    assert content.startswith(r"Costo \$5 e formula \(")
    assert content.endswith(r"\).")
    assert "formula" not in content.split(r"\(")[1]


def test_explicit_display_environment_inside_text_becomes_a_formula():
    content = render_text(r"Prima \begin{equation}x=1\end{equation} dopo.")
    assert content == "Prima \\[\nx=1\n\\] dopo."
    assert r"\textbackslash" not in content


def document(tmp_path, units, title=""):
    return build_tex(DocumentLayout([PageLayout(0, 1000, 1400, "unused.png", units=units)]),
                     tmp_path / "result", title)


def test_small_and_overlapping_source_boxes_do_not_constrain_typography(tmp_path):
    first = unit("first", "First paragraph is a normal sentence.", BBox(10, 0, 45, 5))
    second = unit("second", "Next paragraph follows in document flow.", BBox(10, 3, 45, 8))
    source = document(tmp_path, [second, first]).read_text(encoding="utf-8")
    assert source.index("First paragraph") < source.index("Next paragraph")
    assert "11pt,a4paper" in source and "margin=25mm" in source
    for obsolete in (r"\put", r"\parbox", r"\begin{picture}", r"\resizebox", r"\fontsize", r"\newpage", "HandFit"):
        assert obsolete not in source


def test_request_chunks_join_into_one_paragraph(tmp_path):
    first = unit("first", "The opening part\ncontinues with", BBox(100, 100, 900, 340), members=list("abcdef"))
    second = unit("second", "the remaining sentence.", BBox(100, 350, 850, 390))
    source = document(tmp_path, [first, second]).read_text(encoding="utf-8")
    assert "The opening part continues with the remaining sentence." in source


def test_reading_order_keeps_complete_columns(tmp_path):
    units = [unit("right1", "RIGHTSTART", BBox(600, 100, 950, 140)),
             unit("left2", "LEFTEND", BBox(50, 160, 450, 200)),
             unit("left1", "LEFTSTART", BBox(50, 100, 450, 140)),
             unit("right2", "RIGHTEND", BBox(600, 160, 950, 200))]
    source = document(tmp_path, units).read_text(encoding="utf-8")
    assert source.index("LEFTSTART") < source.index("LEFTEND") < source.index("RIGHTSTART") < source.index("RIGHTEND")


def test_source_pages_do_not_force_document_pages_or_break_a_sentence(tmp_path):
    first = PageLayout(0, 1000, 1400, "unused.png", units=[unit("first", "A sentence continued")])
    second = PageLayout(1, 1000, 1400, "unused.png", units=[unit("second", "on the next source page.", page=1)])
    source = build_tex(DocumentLayout([second, first]), tmp_path / "result", "").read_text(encoding="utf-8")
    assert "A sentence continued on the next source page." in source
    assert r"\newpage" not in source and r"\clearpage" not in source


def test_explicit_headings_lists_and_numbering_become_standard_latex():
    source = render_text("# Introduction\nText with **emphasis** and $x=1$.\n\n"
                         "- First item\n  wrapped continuation\n- Second item\n\n3. Third\n4. Fourth")
    assert r"\section*{Introduction}" in source
    assert r"\textbf{emphasis}" in source and r"\(x=1\)" in source
    assert r"\begin{itemize}" in source
    assert r"\item First item wrapped continuation" in source
    assert r"\item[3.] Third" in source and r"\item[4.] Fourth" in source


def test_figure_uses_original_asset_and_typographic_limits(tmp_path):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (20, 30), "orange").save(image_path)
    figure = unit("figure", "", BBox(100, 200, 600, 400), "figure")
    figure.crop_path = str(image_path)
    tex = document(tmp_path, [figure])
    assert r"width=0.500\linewidth,height=0.55\textheight,keepaspectratio" in tex.read_text()
    assert (tex.parent / figure.crop_path).read_bytes() == image_path.read_bytes()


def test_same_named_figures_from_different_pages_do_not_overwrite(tmp_path):
    pages = []
    for index, color in enumerate(("red", "blue")):
        folder = tmp_path / str(index)
        folder.mkdir()
        image_path = folder / "figure.png"
        Image.new("RGB", (20, 30), color).save(image_path)
        figure = unit("figure", "", category="figure", page=index)
        figure.crop_path = str(image_path)
        pages.append(PageLayout(index, 1000, 1400, "unused.png", units=[figure]))
    tex = build_tex(DocumentLayout(pages), tmp_path / "result", "")
    assert len(list((tex.parent / "assets").glob("*.png"))) == 2
    assert pages[0].units[0].crop_path != pages[1].units[0].crop_path


@pytest.mark.parametrize("label", ["residual_background", "whole_page_no_text"])
def test_legacy_background_is_not_copied_or_inserted(tmp_path, label):
    figure = unit("background", "", BBox(0, 0, 1000, 1400), "figure")
    figure.crop_path = str(tmp_path / "nonexistent-background.png")
    block = LayoutBlock("background", 0, "figure", label, 1, figure.bbox)
    page = PageLayout(0, 1000, 1400, "unused.png", blocks=[block], units=[figure, unit("text", "Normal prose")])
    source = build_tex(DocumentLayout([page]), tmp_path / "result", "").read_text()
    assert "Normal prose" in source and r"\includegraphics" not in source
    assert not (tmp_path / "result" / "assets").exists()


def test_table_and_formula_remain_editable_content_in_sequence(tmp_path):
    units = [unit("intro", "Before the equation", BBox(100, 100, 900, 140)),
             unit("equation", r"\begin{equation}F=ma\end{equation}", BBox(100, 200, 900, 240), "math"),
             unit("table", "| Quantity | Value |\n|---|---|\n| mass | 2 |", BBox(100, 300, 900, 400), "table"),
             unit("end", "After the table", BBox(100, 500, 900, 540))]
    source = document(tmp_path, units).read_text()
    assert source.index("Before the equation") < source.index("F=ma") < source.index("mass & 2") < source.index("After the table")
    assert r"\begin{tabular}" in source and "\\[\nF=ma\n\\]" in source


def compile_document(tex):
    result = compile_pdf(tex)
    log = (tex.parent / "compile.log").read_text()
    assert result["ok"], log
    assert "Overfull" not in log, log
    return tex.with_suffix(".pdf")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_wraps_normal_size_text_and_paginates_instead_of_shrinking(tmp_path):
    text = "ALPHA " + ("Normal document paragraphs wrap at the page margin and stay readable. " * 230) + "OMEGA"
    pdf = compile_document(document(tmp_path, [unit("long", text, BBox(1012, 735, 1077, 786))]))
    with fitz.open(pdf) as doc:
        assert len(doc) >= 3
        extracted = "".join(page.get_text() for page in doc)
        assert "ALPHA" in extracted and "OMEGA" in extracted
        for page in doc:
            prose = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"] if len(span["text"]) > 3]
            assert prose and all(span["size"] >= 10 for span in prose)
            assert all(span["bbox"][0] >= 69 and span["bbox"][2] <= page.rect.width - 69 for span in prose)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_overlapping_scanned_boxes_have_sequential_nonoverlapping_text(tmp_path):
    pdf = compile_document(document(tmp_path, [unit("first", "ALPHA " * 35, BBox(65, 0, 900, 148)),
                                                unit("second", "BETA " * 35, BBox(78, 50, 900, 202))]))
    with fitz.open(pdf) as doc:
        words = doc[0].get_text("words")
        first, second = [w for w in words if w[4] == "ALPHA"], [w for w in words if w[4] == "BETA"]
        assert len(first) == 35 and len(second) == 35
        # They may share a normal text line; their visible word rectangles must
        # not intersect even though the source envelopes did.
        assert all((fitz.Rect(a[:4]) & fitz.Rect(b[:4])).is_empty for a in first for b in second)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_inline_radical_and_structured_content_compile_normally(tmp_path):
    content = "# Basic example\nThe value is $\\sqrt{3}$ and the text continues.\n\n- First item\n- Second item"
    pdf = compile_document(document(tmp_path, [unit("mixed", content, BBox(1012, 735, 1077, 786))]))
    with fitz.open(pdf) as doc:
        text = doc[0].get_text()
        assert "sqrt" not in text and "$" not in text
        assert "Basic example" in text and "3" in text and "First item" in text and "Second item" in text
