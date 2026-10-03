from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from app.models import DocumentLayout, ProcessingUnit
from app.services.table_latex import to_latex as table_to_latex

A4_W_MM = 210.0
A4_H_MM = 297.0

_TEXT_MATH = re.compile(
    r"(?<![\\$])\$\$((?:\\.|[^$])*)\$\$|(?<![\\$])\$(?!\$)((?:\\.|[^$])*)\$(?!\d)"
    r"|(?<!\\)\\\((.*?)\\\)|(?<!\\)\\\[(.*?)\\\]", re.S,
)

# Measure at a native Computer Modern size, then shrink graphics uniformly.
# This does not need an extra font package or substitute arbitrary font sizes.
# Tokens are measured separately so an unbreakable word or inline formula cannot
# protrude from the minipage before the whole paragraph is fitted vertically.
FIT_MACROS = r"""\newsavebox{\HandFitBoxRegister}
\newsavebox{\HandTextBoxRegister}
\newsavebox{\HandTokenBoxRegister}
\newcommand{\HandFitBox}[3]{%
  \begingroup
  \sbox{\HandFitBoxRegister}{#3}%
  \ifdim\wd\HandFitBoxRegister>#1\relax
    \sbox{\HandFitBoxRegister}{\resizebox{#1}{!}{\usebox{\HandFitBoxRegister}}}%
  \fi
  \ifdim\dimexpr\ht\HandFitBoxRegister+\dp\HandFitBoxRegister\relax>#2\relax
    \resizebox*{!}{#2}{\usebox{\HandFitBoxRegister}}%
  \else
    \usebox{\HandFitBoxRegister}%
  \fi
  \endgroup
}
\newcommand{\HandTextToken}[2]{%
  \begingroup
  \sbox{\HandTokenBoxRegister}{#2}%
  \ifdim\wd\HandTokenBoxRegister>#1\relax
    \resizebox{#1}{!}{\usebox{\HandTokenBoxRegister}}%
  \else
    \usebox{\HandTokenBoxRegister}%
  \fi
  \endgroup
}
\def\HandWordStop{\HandEndWords}
\def\HandTextWords#1 {%
  \def\HandCurrentWord{#1}%
  \ifx\HandCurrentWord\HandWordStop
    \let\HandNextWord\relax
  \else
    \HandTextToken{\HandTextLimit}{#1}\space
    \let\HandNextWord\HandTextWords
  \fi
  \HandNextWord
}
\newcommand{\HandTextLine}[2]{%
  \begingroup
  \def\HandTextLimit{#1}%
  \strut\HandTextWords #2 {\HandEndWords} \strut
  \endgroup
}
\newcommand{\HandFitText}[3]{%
  \begingroup
  \sbox{\HandTextBoxRegister}{%
    \begin{minipage}[t]{#1}%
      \fontsize{10}{11.8}\selectfont\raggedright
      \setlength{\parindent}{0pt}\setlength{\parskip}{0pt}%
      #3%
    \end{minipage}%
  }%
  \HandFitBox{#1}{#2}{\usebox{\HandTextBoxRegister}}%
  \endgroup
}"""


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def clean_math(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:latex|tex)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    # OCR can return one wrapped formula or several adjacent wrapped formulas.
    # Strip each segment before the renderer places the result in math mode.
    # Escaped dollars (e.g. \text{cost \$5}) belong to the formula itself.
    wrapped = re.compile(
        r"\$\$((?:\\.|[^$])*)\$\$|\$((?:\\.|[^$])*)\$"
        r"|\\\[(.*?)\\\]|\\\((.*?)\\\)", re.S,
    )
    segments: list[str] = []
    position = 0
    while position < len(value):
        match = wrapped.match(value, position)
        if not match:
            break
        segments.append(next(group for group in match.groups() if group is not None).strip())
        position = match.end()
        while position < len(value) and value[position].isspace():
            position += 1
    if segments and position == len(value):
        value = r"\quad ".join(segment for segment in segments if segment)
    # Display environments open their own math mode and cannot be nested in the
    # resizebox's $...$. Keep their formula content and use inner environments
    # for row/column alignment. Existing matrices, arrays and aligned stay intact.
    inner_environments = {
        "equation": "", "displaymath": "",
        "align": "aligned", "eqnarray": "aligned", "split": "aligned",
        "alignat": "alignedat", "gather": "gathered", "multline": "gathered",
    }

    def inner_environment(match: re.Match[str]) -> str:
        linebreaks, command, environment = match.groups()
        replacement = inner_environments[environment.rstrip("*")]
        return linebreaks + (rf"\{command}{{{replacement}}}" if replacement else "")

    value = re.sub(
        # An environment command may immediately follow a \\ row break. Match
        # pairs of preceding backslashes so the real command is still converted.
        r"(?<!\\)((?:\\\\)*)\\(begin|end)\s*\{(equation\*?|displaymath|align\*?|eqnarray\*?|split|alignat\*?|gather\*?|multline\*?)\}",
        inner_environment, value,
    )
    return value.strip()


def _text_content(content: str, width_mm: float) -> str:
    """Escape prose while keeping explicit math as one measurable TeX token."""
    paragraphs: list[str] = []
    parts: list[str] = []

    def finish_line() -> None:
        value = "".join(parts).strip()
        if value:
            paragraphs.append(rf"\HandTextLine{{{width_mm:.3f}mm}}{{{value}}}")
        parts.clear()

    def prose(value: str) -> None:
        for part in re.split(r"(\r\n|\r|\n)", value):
            if part in {"\r\n", "\r", "\n"}:
                finish_line()
            else:
                parts.append(escape_tex(re.sub(r"[\t\f\v ]+", " ", part)))

    position = 0
    for match in _TEXT_MATH.finditer(content):
        prose(content[position:match.start()])
        formula = clean_math(next(group for group in match.groups() if group is not None))
        if formula:
            # Braces protect spaces inside a formula from the word measurement.
            parts.append("{$" + formula + "$}")
        position = match.end()
    prose(content[position:])
    finish_line()
    return r"\par ".join(paragraphs) + (r"\par" if paragraphs else "")


def _available_height(unit: ProcessingUnit, page_h: int, units: list[ProcessingUnit]) -> float:
    """Limit rendering to the original box, page bottom and next overlapping row."""
    top = unit.bbox.y1 / page_h * A4_H_MM
    bottom = min(A4_H_MM, unit.bbox.y2 / page_h * A4_H_MM)
    for other in units:
        if other is unit or other.category == "figure" or other.bbox.y1 <= unit.bbox.y1:
            continue
        if unit.bbox.horizontal_overlap_ratio(other.bbox) >= .5:
            next_top = other.bbox.y1 / page_h * A4_H_MM
            bottom = min(bottom, next_top - .35)
    return max(0.0, bottom - top)


def _text_box(unit: ProcessingUnit, page_w: int, page_h: int, fit_height_mm: float | None = None) -> str:
    x = unit.bbox.x1 / page_w * A4_W_MM
    top = unit.bbox.y1 / page_h * A4_H_MM
    w = max(4.0, unit.bbox.width / page_w * A4_W_MM)
    h = max(3.0, unit.bbox.height / page_h * A4_H_MM)
    y = A4_H_MM - top - h
    content = unit.decoded.strip()
    fit_width = max(0.0, min(unit.bbox.width / page_w * A4_W_MM, A4_W_MM - x))
    fit_height = max(0.0, min(unit.bbox.height / page_h * A4_H_MM, A4_H_MM - top))
    if fit_height_mm is not None:
        fit_height = min(fit_height, fit_height_mm)
    # Leave a small allowance for glyph bearings and extraction rounding.
    fit_width = max(0.0, fit_width - .15)
    fit_height = max(0.0, fit_height - .15)

    if unit.category == "figure":
        path = Path(unit.crop_path).as_posix()
        body = rf"\includegraphics[width={w:.3f}mm,height={h:.3f}mm,keepaspectratio]{{{path}}}"
    elif unit.category == "math":
        math = clean_math(content)
        body = rf"\HandFitBox{{{fit_width:.3f}mm}}{{{fit_height:.3f}mm}}{{${{\displaystyle {math}}}$}}" if math and fit_width > 0 and fit_height > 0 else ""
    elif unit.category == "table":
        tabular = table_to_latex(content)
        if tabular:
            body = rf"\HandFitBox{{{fit_width:.3f}mm}}{{{fit_height:.3f}mm}}{{{tabular}}}" if fit_width > 0 and fit_height > 0 else ""
        else:
            path = Path(unit.crop_path).as_posix()
            body = rf"\includegraphics[width={w:.3f}mm,height={h:.3f}mm,keepaspectratio]{{{path}}}"
    else:
        escaped = _text_content(content, fit_width)
        body = rf"\HandFitText{{{fit_width:.3f}mm}}{{{fit_height:.3f}mm}}{{{escaped}}}" if escaped and fit_width > 0 and fit_height > 0 else ""

    return (
        rf"\put({x:.3f},{y:.3f}){{\parbox[b][{h:.3f}mm][t]{{{w:.3f}mm}}{{{body}}}}}" + "\n"
    )


def build_tex(layout: DocumentLayout, output_dir: Path, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)

    # Copy all crops needed by the final document to a stable relative directory.
    for page in layout.pages:
        for unit in page.units:
            if unit.category in {"figure", "table"} and unit.crop_path:
                src = Path(unit.crop_path)
                dst = assets / src.name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                unit.crop_path = str(Path("assets") / dst.name)

    pages_tex: list[str] = []
    for page in layout.pages:
        parts = [r"\noindent\begin{picture}(210,297)"]
        for unit in page.units:
            height = _available_height(unit, page.height_px, page.units)
            parts.append(_text_box(unit, page.width_px, page.height_px, height))
        parts.append(r"\end{picture}")
        pages_tex.append("\n".join(parts))

    safe_title = escape_tex(title)
    joined_pages = "\n\\newpage\n".join(pages_tex)
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[paperwidth=210mm,paperheight=297mm,margin=0mm]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,array}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\unitlength}}{{1mm}}
{FIT_MACROS}
\begin{{document}}
{joined_pages}
\end{{document}}
"""
    path = output_dir / "main.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def compile_pdf(tex_path: Path) -> dict[str, object]:
    engine = shutil.which("pdflatex")
    if not engine:
        return {"ok": False, "reason": "pdflatex not installed", "pdf": None}
    logs: list[str] = []
    ok = True
    for _ in range(2):
        proc = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True, timeout=90,
        )
        logs.append(proc.stdout[-12000:] + "\n" + proc.stderr[-4000:])
        if proc.returncode != 0:
            ok = False
            break
    (tex_path.parent / "compile.log").write_text("\n\n--- PASS ---\n\n".join(logs), encoding="utf-8")
    pdf = tex_path.with_suffix(".pdf")
    return {"ok": ok and pdf.exists(), "reason": None if ok else "pdflatex failed", "pdf": str(pdf) if pdf.exists() else None}
