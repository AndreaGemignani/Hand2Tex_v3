from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from app.models import DocumentLayout, PageLayout, ProcessingUnit
from app.services.document_structure import reading_order, starts_paragraph
from app.services.table_latex import to_latex as table_to_latex

_TEXT_MATH = re.compile(
    r"(?<![\\$])\$\$((?:\\.|[^$])*)\$\$|(?<![\\$])\$(?!\$)((?:\\.|[^$])*)\$(?!\d)"
    r"|(?<!\\)\\\((.*?)\\\)|(?<!\\)\\\[(.*?)\\\]", re.S,
)
_CONTENT_MATH = re.compile(
    _TEXT_MATH.pattern + r"|(?<!\\)\\begin\{(?P<environment>equation\*?|displaymath|align\*?|alignat\*?|gather\*?|multline\*?|eqnarray\*?|split)\}.*?\\end\{(?P=environment)\}",
    re.S,
)
_LIST_ITEM = re.compile(r"^\s*(?:([-*•])\s+|(\d+)[.)]\s+)(.+)$")
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+)$")


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def clean_math(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^\x60{3}(?:latex|tex)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*\x60{3}$", "", value)
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
        r"(?<!\\)((?:\\\\)*)\\(begin|end)\s*\{(equation\*?|displaymath|align\*?|eqnarray\*?|split|alignat\*?|gather\*?|multline\*?)\}",
        inner_environment, value,
    )
    return value.strip()


def render_text(content: str) -> str:
    """Soft-wrap OCR lines into prose and keep explicit semantic formatting."""
    value = content.strip().replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"^\x60{3}(?:markdown|text|latex|tex)?\s*\n", "", value, flags=re.I)
    value = re.sub(r"\n\s*\x60{3}$", "", value)
    formulas: dict[str, str] = {}

    def protect_math(match: re.Match[str]) -> str:
        raw_environment = match.group("environment") is not None
        formula = clean_math(match.group(0) if raw_environment else next(group for group in match.groups()[:4] if group is not None))
        token = f"\x00M{len(formulas)}\x00"
        display = raw_environment or match.group(0).startswith(("$$", r"\["))
        formulas[token] = ("\\[\n" + formula + "\n\\]" if display else r"\(" + formula + r"\)")
        return token

    # A currency dollar can otherwise consume the opening delimiter of a later
    # formula. Leave that dollar literal and restart before the later delimiter.
    protected: list[str] = []
    position = 0
    while match := _CONTENT_MATH.search(value, position):
        body = match.group(2)
        currency = (body is not None and re.match(r"\d", body)
                    and re.search(r"\s+[^\W\d_]{2,}", body)
                    and match.end() < len(value) and (value[match.end()].isalnum() or value[match.end()] == "\\"))
        if currency:
            protected.append(value[position:match.start() + 1])
            position = match.start() + 1
        else:
            protected.append(value[position:match.start()])
            protected.append(protect_math(match))
            position = match.end()
    protected.append(value[position:])
    value = "".join(protected)

    def inline(source: str) -> str:
        result: list[str] = []
        position = 0
        for match in re.finditer(r"\x00M\d+\x00|\*\*([^*\n]+)\*\*", source):
            result.append(escape_tex(source[position:match.start()]))
            token = match.group(0)
            result.append(formulas[token] if token in formulas else r"\textbf{" + inline(match.group(1)) + "}")
            position = match.end()
        result.append(escape_tex(source[position:]))
        return "".join(result)

    blocks: list[str] = []
    prose: list[str] = []
    items: list[str] = []
    list_kind = ""

    def flush_prose() -> None:
        if prose:
            blocks.append(inline(" ".join(prose)))
            prose.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if items:
            blocks.append(r"\begin{" + list_kind + "}\n" + "\n".join(items) + "\n" + r"\end{" + list_kind + "}")
            items.clear()
            list_kind = ""

    for raw_line in value.split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if not line:
            flush_prose()
            flush_list()
            continue
        heading = _HEADING.match(line)
        item = _LIST_ITEM.match(line)
        if heading:
            flush_prose()
            flush_list()
            level = min(len(heading.group(1)), 3)
            command = {1: "section", 2: "subsection", 3: "subsubsection"}[level]
            blocks.append("\\" + command + "*{" + inline(heading.group(2)) + "}")
        elif item:
            flush_prose()
            kind = "itemize" if item.group(1) else "enumerate"
            if list_kind and kind != list_kind:
                flush_list()
            list_kind = kind
            # Preserve explicitly recognized numbering, including non-1 starts.
            label = "[" + item.group(2) + ".]" if item.group(2) else ""
            items.append(r"\item" + label + " " + inline(item.group(3)))
        elif items:
            items[-1] += " " + inline(line)
        else:
            prose.append(line)
    flush_prose()
    flush_list()
    return "\n\n".join(blocks)


def _source_background(unit: ProcessingUnit, page: PageLayout) -> bool:
    labels = {block.source_label for block in page.blocks if block.id in unit.member_ids}
    return bool(labels & {"residual_background", "whole_page_no_text"}) or any(
        member.endswith("_residual") for member in unit.member_ids
    )


def _figure(unit: ProcessingUnit, page: PageLayout) -> str:
    if not unit.crop_path:
        return ""
    # Approximate relative size only: no source position or scanned paper.
    fraction = max(.25, min(1.0, unit.bbox.width / max(1, page.width_px)))
    path = Path(unit.crop_path).as_posix()
    return (
        "\\begin{center}\n"
        + rf"\includegraphics[width={fraction:.3f}\linewidth,height=0.55\textheight,keepaspectratio]{{{path}}}"
        + "\n\\end{center}"
    )


def _render_document(pages: list[PageLayout]) -> str:
    blocks: list[str] = []
    prose = ""
    previous: ProcessingUnit | None = None
    previous_page: int | None = None

    def flush() -> None:
        nonlocal prose, previous, previous_page
        if prose.strip():
            blocks.append(render_text(prose))
        prose, previous, previous_page = "", None, None

    ordered = [(page, unit) for page in sorted(pages, key=lambda page: page.index) for unit in reading_order(page)]
    for page, unit in ordered:
        if _source_background(unit, page):
            continue
        if unit.category in {"text", "unknown"}:
            content = unit.decoded.strip()
            if not content:
                continue
            if prose:
                structural = any(_LIST_ITEM.match(line) or _HEADING.match(line)
                                 for line in (prose.splitlines()[-1], content.splitlines()[0]))
                boundary = previous_page == page.index and starts_paragraph(previous, unit, page)
                separator = "\n\n" if boundary else "\n" if structural else " "
                prose += separator
            prose += content
            previous = unit
            previous_page = page.index
            continue
        flush()
        if unit.category == "math":
            formula = clean_math(unit.decoded)
            if formula:
                blocks.append("\\[\n" + formula + "\n\\]")
        elif unit.category == "table":
            tabular = table_to_latex(unit.decoded)
            if tabular:
                blocks.append("\\begin{center}\n" + tabular + "\n\\end{center}")
            elif unit.crop_path:
                blocks.append(_figure(unit, page))
        elif unit.category == "figure":
            blocks.append(_figure(unit, page))
    flush()
    return "\n\n".join(block for block in blocks if block)


def build_tex(layout: DocumentLayout, output_dir: Path, title: str) -> Path:
    """Author a standard editable document; LaTeX handles wrapping and pages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    for page in layout.pages:
        for unit in page.units:
            if unit.category not in {"figure", "table"} or not unit.crop_path or _source_background(unit, page):
                continue
            src = Path(unit.crop_path)
            if not src.is_absolute() and (output_dir / src).is_file():
                continue
            assets.mkdir(exist_ok=True)
            identifier = re.sub(r"[^A-Za-z0-9_.-]", "_", unit.id)
            dst = assets / f"p{page.index:04d}_{identifier}{src.suffix.lower()}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            unit.crop_path = str(Path("assets") / dst.name)

    body = _render_document(layout.pages)
    heading = "\\title{" + escape_tex(title.strip()) + "}\n\\date{}\n\\maketitle\n" if title.strip() else ""
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=25mm]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,array}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\emergencystretch}}{{2em}}
\linespread{{1.08}}
\begin{{document}}
{heading}{body}
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
