from __future__ import annotations

import re


def _balanced(text: str, left: str, right: str) -> bool:
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == left:
            depth += 1
        elif ch == right:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate_text(text: str) -> float:
    clean = text.strip()
    if not clean:
        return 0.0
    q_ratio = clean.count("?") / max(1, len(clean))
    replacement_ratio = clean.count("�") / max(1, len(clean))
    weird = len(re.findall(r"[\uFFFD]", clean)) / max(1, len(clean))
    score = 1.0 - min(0.8, q_ratio * 4 + replacement_ratio * 6 + weird * 6)
    if len(clean) < 2:
        score *= 0.5
    return max(0.0, min(1.0, score))


def validate_math(text: str) -> float:
    clean = text.strip().replace("```latex", "").replace("```tex", "").replace("```", "").strip()
    if not clean:
        return 0.0
    score = 0.45
    if _balanced(clean, "{", "}"):
        score += 0.25
    if any(token in clean for token in ("\\frac", "\\begin", "\\sqrt", "^", "_", "=", "\\sum", "\\int")):
        score += 0.20
    if "?" not in clean and "�" not in clean:
        score += 0.10
    return max(0.0, min(1.0, score))


def validate_table(text: str) -> float:
    clean = text.strip().lower()
    if not clean:
        return 0.0
    if "\\begin{tabular" in clean:
        return 0.95
    if "<table" in clean and "</table>" in clean:
        return 0.90
    lines = [line for line in clean.splitlines() if line.strip()]
    if len(lines) >= 2 and sum("|" in line for line in lines) >= 2:
        return 0.82
    return 0.45


def validate(category: str, text: str) -> float:
    if category == "math":
        return validate_math(text)
    if category == "table":
        return validate_table(text)
    return validate_text(text)
