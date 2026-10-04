import shutil

import fitz
import pytest

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, compile_pdf


NOTE = "Trascrizione da verificare sull’originale."


def unit(identifier, content, category="text", *, review=None, top=100, members=1):
    return ProcessingUnit(
        identifier, 0, category, BBox(100, top, 900, top + 35 * members),
        [f"{identifier}-{index}" for index in range(members)], .95,
        decoded=content, quality_review=review or {},
    )


def document(tmp_path, units):
    page = PageLayout(0, 1000, 1400, "unused.png", units=units)
    return build_tex(DocumentLayout([page]), tmp_path / "result", "")


@pytest.mark.parametrize("category,content", [
    ("text", "A second OCR draft must not be printed"),
    ("math", r"\sqrt{3}"),
    ("table", "| Lost | Draft |\n|---|---|\n| 1 | 2 |"),
])
def test_merged_secondary_units_do_not_render_content_notes_or_copy_assets(tmp_path, category, content):
    primary = unit("primary", r"The complete expression is $1/\sqrt{3}$.",
                   review={"status": "corrected"})
    secondary = unit("secondary", content, category, top=200,
                     review={"status": "uncertain", "merged_into": "primary", "mark_pdf": True})
    secondary.crop_path = str(tmp_path / "nonexistent-secondary.png")
    source = document(tmp_path, [primary, secondary]).read_text(encoding="utf-8")
    assert r"\(1/\sqrt{3}\)" in source
    if category == "math":
        assert source.count(r"\sqrt{3}") == 1
    else:
        assert content not in source
    assert NOTE not in source
    assert r"\begin{tabular}" not in source
    assert r"\includegraphics" not in source
    assert not (tmp_path / "result" / "assets").exists()
    assert secondary.decoded == content


def test_uncertain_prose_keeps_inline_math_escaping_and_paragraph_continuity(tmp_path):
    first = unit("first", "Before the reading", review={"status": "verified"}, members=2)
    draft = "contains $\\sqrt{3}$ & 50% of A_B"
    uncertain = unit("uncertain", draft, top=180, review={
        "status": "uncertain", "issues": [r"Untrusted issue $\badcommand{50%}"],
    })
    last = unit("last", "and the sentence continues.", top=225, review={"status": "verified"})
    source = document(tmp_path, [first, uncertain, last]).read_text(encoding="utf-8")
    assert source.count(NOTE) == 1
    assert r"\(\sqrt{3}\) \& 50\% of A\_B" in source
    assert "Before the reading contains" in source
    assert "]} and the sentence continues." in source
    assert r"\badcommand" not in source
    assert r"\textbackslash{}sqrt" not in source
    assert uncertain.decoded == draft


@pytest.mark.parametrize("content", [
    r"\frac{1}{\sqrt{3}}",
    r"\begin{equation}x=1\end{equation}",
])
def test_uncertain_formula_note_is_outside_the_math_environment(tmp_path, content):
    source = document(tmp_path, [unit("formula", content, "math", review={"status": "uncertain"})]).read_text(encoding="utf-8")
    assert source.count(NOTE) == 1
    assert "\\]\n\n{\\small\\itshape [" + NOTE in source
    assert r"\textbackslash{}frac" not in source
    assert source.count(r"\[") == source.count(r"\]") == 1


def test_uncertain_table_note_does_not_become_a_cell(tmp_path):
    table = "| Item | Amount |\n|---|---|\n| A_B | 50% |"
    source = document(tmp_path, [unit("table", table, "table", review={"status": "uncertain"})]).read_text(encoding="utf-8")
    assert r"A\_B & 50\%" in source
    assert source.index(r"\end{tabular}") < source.index(NOTE)
    assert source.count(NOTE) == 1


@pytest.mark.parametrize("review", [
    {}, {"status": "disabled"}, {"status": "not_reviewed"},
    {"status": "verified"}, {"status": "corrected"},
    {"status": "not_reviewed", "mark_pdf": False},
])
def test_unmarked_review_states_do_not_break_normal_inline_prose(tmp_path, review):
    first = unit("first", "The value is $x=1$ and", review=review, members=2)
    second = unit("second", "the prose continues.", top=180, review=review)
    source = document(tmp_path, [first, second]).read_text(encoding="utf-8")
    assert r"The value is \(x=1\) and the prose continues." in source
    assert NOTE not in source


def test_explicit_pdf_mark_can_flag_an_unreviewed_region(tmp_path):
    source = document(tmp_path, [unit("draft", "A draft", review={
        "status": "not_reviewed", "mark_pdf": True,
    })]).read_text(encoding="utf-8")
    assert source.count(NOTE) == 1


@pytest.mark.parametrize("category", ["text", "math", "table"])
def test_empty_uncertain_primary_regions_remain_visibly_flagged(tmp_path, category):
    source = document(tmp_path, [unit("empty", "", category, review={"status": "uncertain"})]).read_text(encoding="utf-8")
    assert source.count(NOTE) == 1
    assert r"\includegraphics" not in source


def test_uncertain_code_fenced_markdown_keeps_its_semantics(tmp_path):
    draft = "```markdown\n# A heading\nThe value is $x=1$.\n```"
    source = document(tmp_path, [unit("draft", draft, review={"status": "uncertain"})]).read_text(encoding="utf-8")
    assert r"\section*{A heading}" in source
    assert r"The value is \(x=1\)." in source
    assert "```" not in source
    assert source.count(NOTE) == 1


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_pdf_review_notes_compile_beside_prose_math_and_table(tmp_path):
    tex = document(tmp_path, [
        unit("text", "Text & $x=1$.", review={"status": "uncertain"}),
        unit("math", r"\frac{1}{\sqrt{3}}", "math", top=250, review={"status": "uncertain"}),
        unit("table", "| X | Y |\n|---|---|\n| 1 | 2 |", "table", top=400, review={"status": "uncertain"}),
    ])
    compilation = compile_pdf(tex)
    assert compilation["ok"], (tex.parent / "compile.log").read_text(encoding="utf-8")
    with fitz.open(tex.with_suffix(".pdf")) as doc:
        extracted = "\n".join(page.get_text() for page in doc)
        assert extracted.count("Trascrizione da veri") == 3
        assert "sqrt" not in extracted and "$" not in extracted
