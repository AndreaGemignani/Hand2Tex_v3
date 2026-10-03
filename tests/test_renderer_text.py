import re
import shutil

import pytest

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import _text_box, build_tex, clean_math, compile_pdf


def text_unit(decoded):
    return ProcessingUnit(
        "text-box", 0, "text", BBox(100, 200, 600, 400), ["text-block"], .95,
        decoded=decoded, decoder="test", validation_score=1,
    )


def test_multiline_ocr_text_preserves_characters_without_empty_latex_linebreaks(tmp_path):
    decoded = (
        "\r\n  \r\n   È già un caffè & tè: 50%  \r\n"
        "Seconda riga: $ # _ { } ~ ^ \\  \r\n \t\r\n\r\n"
        "  Ultimo paragrafo   \n\n   "
    )
    unit = text_unit(decoded)
    page = PageLayout(0, 1000, 1000, "unused.png", units=[unit])
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Text rendering")
    document = tex_path.read_text(encoding="utf-8")
    box = _text_box(unit, 1000, 1000)

    assert "È già un caffè \\& tè: 50\\%" in document
    assert r"Seconda riga: \$ \# \_ \{ \} \textasciitilde{} \textasciicircum{} \textbackslash{}" in document
    assert "Ultimo paragrafo" in document
    assert document.index("È già un caffè") < document.index("Seconda riga") < document.index("Ultimo paragrafo")
    assert len(re.findall(r"\\par\b", box)) >= 2
    assert r"\\ \\" not in document
    assert not re.search(r"\\\\\s*\\\\", document)
    # Inspect the box to exclude legitimate document macros such as \noindent.
    assert r"\n" not in box
    assert "\r" not in box
    assert box.startswith(r"\put(21.000,178.200){\parbox[b][59.400mm][t]{105.000mm}{")


def test_line_and_paragraph_separators_leave_box_position_and_size_unchanged():
    plain = _text_box(text_unit("First line"), 1000, 1000)
    multiline = _text_box(text_unit("\n  First line\n\n\nSecond line\r\n  \r\nThird line\n  "), 1000, 1000)
    geometry_pattern = r"^\\put\(([^)]+)\)\{\\parbox\[b\]\[([^]]+)\]\[t\]\{([^}]+)\}"
    assert re.match(geometry_pattern, multiline).groups() == re.match(geometry_pattern, plain).groups()
    assert "First line" in multiline and "Second line" in multiline and "Third line" in multiline
    assert re.search(r"\\par\b", multiline)
    assert not re.search(r"\\\\\s*\\\\", multiline)
    assert r"\n" not in multiline


@pytest.mark.parametrize("decoded", ["", "\n\n", "\r\n   \r\n\t\r\n"])
def test_empty_or_whitespace_only_text_has_no_forced_linebreaks(decoded):
    box = _text_box(text_unit(decoded), 1000, 1000)
    assert r"\\" not in box
    assert r"\n" not in box
    assert not re.search(r"\\par\b", box)
    assert box.startswith(r"\put(21.000,178.200){\parbox[b][59.400mm][t]{105.000mm}{")


@pytest.mark.parametrize("wrapped", [
    r"$\frac{F}{m}=a$",
    r"$$\frac{F}{m}=a$$",
    r"\[\frac{F}{m}=a\]",
    r"\(\frac{F}{m}=a\)",
    "```latex\n\\frac{F}{m}=a\n```",
    "```tex\n$\\frac{F}{m}=a$\n```",
])
def test_clean_math_removes_outer_display_and_inline_wrappers(wrapped):
    assert clean_math(wrapped) == r"\frac{F}{m}=a"


@pytest.mark.parametrize("wrapped,segments", [
    (r"$x=1$ $y=2$", ["x=1", "y=2"]),
    ("$$x=1$$\n$$y=2$$", ["x=1", "y=2"]),
    (r"\(x=1\) \(y=2\)", ["x=1", "y=2"]),
    (r"\[x=1\] \[y=2\]", ["x=1", "y=2"]),
    (r"$x=1$ \(y=2\)", ["x=1", "y=2"]),
    (r"$\text{cost \$5}=x$ $y=2$", [r"\text{cost \$5}=x", "y=2"]),
    (r"$$\text{cost \$5}=x$$ $$y=2$$", [r"\text{cost \$5}=x", "y=2"]),
])
def test_adjacent_wrapped_formulas_are_joined_without_nested_math(wrapped, segments):
    cleaned = clean_math(wrapped)
    assert re.split(r"\s*\\quad\s*", cleaned) == segments
    assert cleaned.count(r"\quad") == len(segments) - 1


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
@pytest.mark.parametrize("math_text", [
    r"$\frac{F}{m}=a$",
    r"\(\frac{F}{m}=a\)",
    r"\begin{aligned}F&=ma\\a&=\frac{F}{m}\end{aligned}",
    r"$x=1$ $y=2$",
    r"$\text{cost \$5}=x$ $y=2$",
])
def test_real_tex_compiler_accepts_blank_ocr_lines_accents_and_wrapped_math(tmp_path, math_text):
    text = (
        "\r\n \r\nÈ già una temperatura di 20° & umidità del 50%.\r\n\r\n"
        "Costo $5, codice #A_1, {nota}, ~, ^ e \\.\r\n \t\r\n"
        "Ultima riga di testo sintetico.\r\n\r\n"
    )
    page = PageLayout(0, 1000, 1400, "unused.png")
    page.units = [
        ProcessingUnit(
            "text", 0, "text", BBox(50, 100, 600, 500), ["t"], .95,
            decoded=text, decoder="test", validation_score=1,
        ),
        ProcessingUnit(
            "math", 0, "math", BBox(50, 700, 800, 850), ["m"], .95,
            decoded=math_text, decoder="test", validation_score=1,
        ),
    ]
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Real compilation")
    compilation = compile_pdf(tex_path)
    compile_log = (tex_path.parent / "compile.log").read_text(encoding="utf-8")
    assert compilation["ok"], compile_log
    assert tex_path.with_suffix(".pdf").exists()
    assert "There's no line here to end" not in compile_log
