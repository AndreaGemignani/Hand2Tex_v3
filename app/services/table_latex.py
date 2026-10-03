from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str):
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _esc(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in html.unescape(value))


def _rows_to_latex(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    normalized = [r + [""] * (cols - len(r)) for r in rows]
    row_sep = r" \\ \hline" + "\n"
    body = row_sep.join(" & ".join(_esc(c) for c in row) for row in normalized)
    return (
        r"\begin{tabular}{|" + "l|" * cols + "}\n"
        + r"\hline" + "\n"
        + body
        + row_sep
        + r"\end{tabular}"
    )

def to_latex(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:html|markdown|latex|tex)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    if "\\begin{tabular" in text:
        return text
    if "<table" in text.lower():
        parser = _TableParser()
        parser.feed(text)
        converted = _rows_to_latex(parser.rows)
        if converted:
            return converted
    lines = [ln.strip() for ln in text.splitlines() if "|" in ln]
    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and not all(set(c) <= {"-", ":", " "} for c in cells):
            rows.append(cells)
    converted = _rows_to_latex(rows)
    return converted or ""
