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
        command, environment = match.groups()
        replacement = inner_environments[environment.rstrip("*")]
        return rf"\{command}{{{replacement}}}" if replacement else ""

    value = re.sub(
        r"(?<!\\)\\(begin|end)\s*\{(equation\*?|displaymath|align\*?|eqnarray\*?|split|alignat\*?|gather\*?|multline\*?)\}",
        inner_environment, value,
    )
    return value.strip()


def _font_size_for(unit: ProcessingUnit, page_h: int, text: str) -> float:
    h_ratio = unit.bbox.height / max(1, page_h)
    lines = max(1, len(text.splitlines()))
    approx = (h_ratio * A4_H_MM * 2.6) / lines
    return max(7.0, min(15.0, approx))


def _text_box(unit: ProcessingUnit, page_w: int, page_h: int) -> str:
    x = unit.bbox.x1 / page_w * A4_W_MM
    top = unit.bbox.y1 / page_h * A4_H_MM
    w = max(4.0, unit.bbox.width / page_w * A4_W_MM)
    h = max(3.0, unit.bbox.height / page_h * A4_H_MM)
    y = A4_H_MM - top - h
    content = unit.decoded.strip()

    if unit.category == "figure":
        path = Path(unit.crop_path).as_posix()
        body = rf"\includegraphics[width={w:.3f}mm,height={h:.3f}mm,keepaspectratio]{{{path}}}"
    elif unit.category == "math":
        math = clean_math(content)
        body = rf"\resizebox{{{w:.3f}mm}}{{!}}{{${{\displaystyle {math}}}$}}" if math else ""
    elif unit.category == "table":
        tabular = table_to_latex(content)
        if tabular:
            body = rf"\resizebox{{{w:.3f}mm}}{{!}}{{{tabular}}}"
        else:
            path = Path(unit.crop_path).as_posix()
            body = rf"\includegraphics[width={w:.3f}mm,height={h:.3f}mm,keepaspectratio]{{{path}}}"
    else:
        fs = _font_size_for(unit, page_h, content)
        # OCR paragraphs can include blank lines. Consecutive \\ commands fail
        # with "There's no line here to end"; explicit paragraphs are safe.
        escaped = r"\par ".join(escape_tex(line) for line in content.splitlines() if line.strip())
        body = rf"\fontsize{{{fs:.2f}}}{{{fs*1.18:.2f}}}\selectfont\raggedright {escaped}\par" if escaped else ""

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
            parts.append(_text_box(unit, page.width_px, page.height_px))
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
