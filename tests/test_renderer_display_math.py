import re
import shutil

import pytest

from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.services.renderer import build_tex, clean_math, compile_pdf


@pytest.mark.parametrize("environment", ["equation", "equation*", "displaymath"])
def test_display_equation_wrappers_are_removed_before_inline_rendering(environment):
    formula = r"\frac{u}{v}=w"
    source = rf"\begin{{{environment}}}{formula}\end{{{environment}}}"
    assert clean_math(source) == formula


@pytest.mark.parametrize("source_environment,target_environment,argument", [
    ("align", "aligned", ""),
    ("align*", "aligned", ""),
    ("gather", "gathered", ""),
    ("gather*", "gathered", ""),
    ("multline", "gathered", ""),
    ("multline*", "gathered", ""),
    ("eqnarray", "aligned", ""),
    ("eqnarray*", "aligned", ""),
    ("alignat", "alignedat", "{2}"),
    ("alignat*", "alignedat", "{2}"),
    ("split", "aligned", ""),
])
def test_display_environments_become_compatible_inner_math(
    source_environment, target_environment, argument,
):
    body = r"x=1\\y=2"
    source = rf"\begin{{{source_environment}}}{argument}{body}\end{{{source_environment}}}"
    expected = rf"\begin{{{target_environment}}}{argument}{body}\end{{{target_environment}}}"
    assert clean_math(source) == expected


@pytest.mark.parametrize("wrapper", [
    "${}$",
    "$${}$$",
    r"\({}\)",
    r"\[{}\]",
    "```latex\n{}\n```",
    "```tex\n$${}$$\n```",
])
def test_external_delimiters_and_fences_are_removed_before_environment_mapping(wrapper):
    source = r"\begin{align*}u&=v+w\\x&=y+z\end{align*}"
    expected = r"\begin{aligned}u&=v+w\\x&=y+z\end{aligned}"
    assert clean_math(wrapper.format(source)) == expected


@pytest.mark.parametrize("formula", [
    r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
    r"\begin{aligned}u&=v+w\\x&=y+z\end{aligned}",
    r"\begin{array}{cc}1&2\\3&4\end{array}",
])
def test_existing_inner_math_environments_and_arguments_are_preserved(formula):
    assert clean_math(formula) == formula


def test_nested_equation_and_split_keep_alignment_and_formula_content():
    source = r"\begin{equation}\begin{split}x&=y+z\\&=1+2\end{split}\end{equation}"
    assert clean_math(source) == r"\begin{aligned}x&=y+z\\&=1+2\end{aligned}"


def math_unit(identifier, formula, bbox):
    return ProcessingUnit(
        identifier, 0, "math", bbox, [identifier], .95,
        decoded=formula, decoder="qwen-ocr:math-auto", validation_score=1,
    )


def compile_page(tmp_path, units):
    page = PageLayout(0, 1000, 1400, "unused.png", units=units)
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Synthetic display equations")
    result = compile_pdf(tex_path)
    compile_log = (tex_path.parent / "compile.log").read_text(encoding="utf-8")
    assert result["ok"], compile_log
    assert tex_path.with_suffix(".pdf").exists()
    assert "Bad math environment delimiter" not in compile_log
    return tex_path.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
def test_real_compilation_accepts_equation_and_align_with_trailing_break_and_prose(tmp_path):
    # Synthetic formulas reproduce the two environment shapes found in OCR output.
    equation = r"\begin{equation}\frac{u}{v}=w\end{equation}"
    alignment = r"\begin{align*}x&=y+\text{synthetic explanation} \\\end{align*}"
    document = compile_page(tmp_path, [
        math_unit("equation", equation, BBox(50, 100, 850, 350)),
        math_unit("alignment", alignment, BBox(50, 500, 850, 850)),
    ])
    assert r"\frac{u}{v}=w" in document
    assert r"\begin{aligned}" in document
    assert r"\text{synthetic explanation}" in document
    assert not re.search(r"\\(?:begin|end)\{(?:equation\*?|align\*?)\}", document)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable in test host")
@pytest.mark.parametrize("formula", [
    r"\begin{gather*}x=1\\y=2\end{gather*}",
    r"\begin{multline*}x+y+z=1\\u+v+w=2\end{multline*}",
    r"\begin{eqnarray*}x&=&y+z\\u&=&v+w\end{eqnarray*}",
    r"\begin{alignat*}{2}x&=1 &\quad y&=2\\u&=3 &\quad v&=4\end{alignat*}",
    r"\begin{equation}\begin{split}x&=y+z\\&=1+2\end{split}\end{equation}",
])
def test_real_compilation_accepts_common_display_environment_variants(tmp_path, formula):
    compile_page(tmp_path, [math_unit("formula", formula, BBox(50, 100, 850, 500))])
