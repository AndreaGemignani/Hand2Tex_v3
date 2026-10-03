from __future__ import annotations

from collections import defaultdict

from app.models import BBox, LayoutBlock, ProcessingUnit


def _dedupe(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Remove child text inside figures/tables and trivial duplicate boxes."""
    blocks = sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1, -b.score))
    keep: list[LayoutBlock] = []
    for block in blocks:
        drop = False
        for other in blocks:
            if other.id == block.id:
                continue
            if block.category == "text" and other.category in {"figure", "table"}:
                if other.source_label == "residual_background":
                    continue
                if block.bbox.contained_fraction(other.bbox) >= 0.82:
                    drop = True
                    break
            if block.category == other.category and block.score <= other.score:
                if block.bbox.contained_fraction(other.bbox) >= 0.93 and other.bbox.contained_fraction(block.bbox) >= 0.72:
                    drop = True
                    break
        if not drop:
            keep.append(block)
    return keep


def _should_merge(a: LayoutBlock, b: LayoutBlock, page_w: int, page_h: int, category: str) -> bool:
    if a.category != b.category or a.category != category:
        return False
    vertical_gap = b.bbox.y1 - a.bbox.y2
    if vertical_gap < -0.01 * page_h:
        return False
    hoverlap = a.bbox.horizontal_overlap_ratio(b.bbox)
    center_delta = abs((a.bbox.x1 + a.bbox.x2) / 2 - (b.bbox.x1 + b.bbox.x2) / 2)
    if category == "math":
        return vertical_gap <= 0.035 * page_h and (hoverlap >= 0.20 or center_delta <= 0.18 * page_w)
    if category == "text":
        return vertical_gap <= 0.018 * page_h and hoverlap >= 0.55
    return False


def build_processing_units(blocks: list[LayoutBlock], page_w: int, page_h: int, page_index: int) -> list[ProcessingUnit]:
    blocks = _dedupe(blocks)
    by_cat: dict[str, list[LayoutBlock]] = defaultdict(list)
    for block in blocks:
        by_cat[block.category].append(block)

    units: list[ProcessingUnit] = []
    counter = 0
    for category in ("text", "math", "table", "figure", "unknown"):
        cat_blocks = sorted(by_cat.get(category, []), key=lambda b: (b.bbox.y1, b.bbox.x1))
        i = 0
        while i < len(cat_blocks):
            current = [cat_blocks[i]]
            bbox = cat_blocks[i].bbox
            j = i + 1
            if category in {"text", "math"}:
                while j < len(cat_blocks) and _should_merge(current[-1], cat_blocks[j], page_w, page_h, category):
                    current.append(cat_blocks[j])
                    bbox = bbox.union(cat_blocks[j].bbox)
                    j += 1
            unit_id = f"p{page_index:04d}_{category[:1].upper()}{counter:04d}"
            units.append(ProcessingUnit(
                id=unit_id,
                page=page_index,
                category=category,  # type: ignore[arg-type]
                bbox=bbox,
                member_ids=[b.id for b in current],
                detector_score=sum(b.score for b in current) / len(current),
                hint_text="\n".join(b.hint_text for b in current if b.hint_text).strip(),
            ))
            counter += 1
            i = j
    return sorted(units, key=lambda u: (u.bbox.y1, u.bbox.x1))
