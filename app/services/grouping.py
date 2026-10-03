from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from app.models import BBox, LayoutBlock, ProcessingUnit


def _bbox_polygon(box: BBox) -> list[float]:
    return [box.x1, box.y1, box.x2, box.y1, box.x2, box.y2, box.x1, box.y2]


def _has_polygon(block: LayoutBlock) -> bool:
    polygon = getattr(block, "polygon", [])
    return len(polygon) == 8 and all(math.isfinite(value) for value in polygon)


def _points(block: LayoutBlock) -> list[tuple[float, float]]:
    polygon = block.polygon if _has_polygon(block) else _bbox_polygon(block.bbox)
    return list(zip(polygon[::2], polygon[1::2]))


def _signed_area(points: list[tuple[float, float]]) -> float:
    return sum(
        x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
        for i, (x, y) in enumerate(points)
    ) / 2 if points else 0.0


def _intersection_area(a: LayoutBlock, b: LayoutBlock) -> float:
    """Intersect convex text quadrilaterals rather than inflated envelopes."""
    subject, clip = _points(a), _points(b)
    orientation = 1 if _signed_area(clip) >= 0 else -1
    for i, (ax, ay) in enumerate(clip):
        bx, by = clip[(i + 1) % len(clip)]

        def side(point: tuple[float, float]) -> float:
            return orientation * ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax))

        previous, output = subject, []
        if not previous:
            return 0.0
        start = previous[-1]
        for end in previous:
            ds, de = side(start), side(end)
            if (ds >= 0) != (de >= 0):
                fraction = ds / (ds - de)
                output.append((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
            if de >= 0:
                output.append(end)
            start = end
        subject = output
    return abs(_signed_area(subject))


@dataclass(frozen=True)
class _Line:
    left: float
    right: float
    center_y: float
    slope: float
    height: float

    def y_at(self, x: float) -> float:
        return self.center_y + self.slope * (x - (self.left + self.right) / 2)


def _line(block: LayoutBlock) -> _Line:
    points = _points(block)
    lengths = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    # Midpoints of short sides are the endpoints of the text centerline.
    short = 0 if lengths[0] + lengths[2] <= lengths[1] + lengths[3] else 1
    endpoints = [
        ((points[i][0] + points[(i + 1) % 4][0]) / 2, (points[i][1] + points[(i + 1) % 4][1]) / 2)
        for i in (short, (short + 2) % 4)
    ]
    left, right = sorted(endpoints)
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = math.hypot(dx, dy)
    if not _has_polygon(block) or dx <= 0 or abs(dy) > dx or length <= 0:
        box = block.bbox
        return _Line(box.x1, box.x2, (box.y1 + box.y2) / 2, 0.0, max(1.0, box.height))
    # Spacing is measured vertically at a shared x, so use vertical strip
    # thickness (area/dx), not thickness perpendicular to the sloped baseline.
    return _Line(left[0], right[0], (left[1] + right[1]) / 2, dy / dx, max(1.0, abs(_signed_area(points)) / dx))


def _dedupe(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Keep one representative of a redundant region; preserve separate lines."""
    eligible = []
    for block in blocks:
        if block.category == "text" and any(
            other.category in {"figure", "table"}
            and other.source_label != "residual_background"
            and _intersection_area(block, other) / max(1.0, abs(_signed_area(_points(block)))) >= .82
            for other in blocks if other.id != block.id
        ):
            continue
        eligible.append(block)

    # Compare only with retained representatives. Equal scores cannot mutually
    # delete duplicates; stable ID ties make the result independent of input order.
    keep: list[LayoutBlock] = []
    for block in sorted(eligible, key=lambda b: (-b.score, b.id)):
        redundant = False
        for other in keep:
            if block.category != other.category or "residual_background" in (block.source_label, other.source_label):
                continue
            a, b = _line(block), _line(other)
            if abs(a.center_y - b.center_y) > .15 * min(a.height, b.height):
                continue
            area = _intersection_area(block, other)
            if all(area / max(1.0, abs(_signed_area(_points(candidate)))) >= .90 for candidate in (block, other)):
                redundant = True
                break
        if not redundant:
            keep.append(block)
    return sorted(keep, key=lambda b: (_line(b).center_y, _line(b).left, b.id))


def _should_merge(a: LayoutBlock, b: LayoutBlock, page_w: int, page_h: int, category: str) -> bool:
    if a.category != b.category or a.category != category:
        return False
    ga, gb = _line(a), _line(b)
    overlap = max(0.0, min(ga.right, gb.right) - max(ga.left, gb.left))
    hoverlap = overlap / max(1.0, min(ga.right - ga.left, gb.right - gb.left))
    x = (max(ga.left, gb.left) + min(ga.right, gb.right)) / 2 if overlap else (ga.left + ga.right + gb.left + gb.right) / 4
    center_step = gb.y_at(x) - ga.y_at(x)
    gap = center_step - (ga.height + gb.height) / 2
    if center_step < .35 * min(ga.height, gb.height):
        return False
    if category == "math":
        center_delta = abs((ga.left + ga.right - gb.left - gb.right) / 2)
        return gap >= -.01 * page_h and gap <= .035 * page_h and (hoverlap >= .20 or center_delta <= .18 * page_w)
    if category == "text":
        # Centerline spacing remains stable under tilt; envelope gaps do not.
        widths = [ga.right - ga.left, gb.right - gb.left]
        if min(widths) < .15 * page_w and max(widths) > 2.5 * max(1.0, min(widths)):
            # A denominator or short label next to a wider row must not drag a
            # following paragraph into the neighbouring formula's envelope.
            return False
        same_size = max(ga.height, gb.height) <= 1.75 * min(ga.height, gb.height)
        aligned_left = abs(ga.left - gb.left) <= max(.035 * page_w, .25 * min(ga.right - ga.left, gb.right - gb.left))
        max_gap = min(.018 * page_h, .65 * (ga.height + gb.height) / 2)
        return same_size and aligned_left and gap >= -.75 * min(ga.height, gb.height) and gap <= max_gap and hoverlap >= .55
    return False


def _bridges_columns(current: list[LayoutBlock], candidate: LayoutBlock, context: list[LayoutBlock], page_w: int) -> bool:
    """A spanning header must not pull a neighboring column into its group."""
    lines = [_line(block) for block in current + [candidate]]
    left, right = min(line.left for line in lines), max(line.right for line in lines)
    members = {block.id for block in current + [candidate]}
    for anchor in (current[-1], candidate):
        ga = _line(anchor)
        if ga.right - ga.left < .10 * page_w:
            continue
        for other in context:
            if other.id in members:
                continue
            go = _line(other)
            if go.right - go.left < .10 * page_w or abs(ga.center_y - go.center_y) > .6 * max(ga.height, go.height):
                continue
            gutter = max(ga.left, go.left) - min(ga.right, go.right)
            if gutter >= .035 * page_w and min(right, go.right) - max(left, go.left) >= .25 * (go.right - go.left):
                return True
    return False


def _text_group_fits(current: list[LayoutBlock], candidate: LayoutBlock, page_h: int) -> bool:
    # Six source lines and one fifth of a page per text crop prevent transitive
    # chains across sections without forcing one paid OCR request per line.
    lines = [_line(block) for block in current + [candidate]]
    height = max(line.center_y + line.height / 2 for line in lines) - min(line.center_y - line.height / 2 for line in lines)
    return len(lines) <= 6 and height <= .20 * page_h


def build_processing_units(blocks: list[LayoutBlock], page_w: int, page_h: int, page_index: int) -> list[ProcessingUnit]:
    blocks = _dedupe(blocks)
    by_cat: dict[str, list[LayoutBlock]] = defaultdict(list)
    for block in blocks:
        by_cat[block.category].append(block)

    units: list[ProcessingUnit] = []
    counter = 0
    for category in ("text", "math", "table", "figure", "unknown"):
        cat_blocks = sorted(by_cat.get(category, []), key=lambda b: (_line(b).center_y, _line(b).left, b.id))
        groups: list[list[LayoutBlock]] = []
        for block in cat_blocks:
            candidates = [
                group for group in groups
                if category in {"text", "math"}
                and _should_merge(group[-1], block, page_w, page_h, category)
                and (category != "text" or _text_group_fits(group, block, page_h))
                and not _bridges_columns(group, block, cat_blocks, page_w)
            ]
            if candidates:
                # Interleaved columns can join their own previous row even when
                # another column appears between them in global reading order.
                closest = min(candidates, key=lambda group: (
                    _line(block).center_y - _line(group[-1]).center_y,
                    abs(_line(block).left - _line(group[-1]).left), group[0].id,
                ))
                closest.append(block)
            else:
                groups.append([block])

        for current in groups:
            bbox = current[0].bbox
            for block in current[1:]:
                bbox = bbox.union(block.bbox)
            unit_id = f"p{page_index:04d}_{category[:1].upper()}{counter:04d}"
            polygons = [list(block.polygon) if _has_polygon(block) else _bbox_polygon(block.bbox) for block in current] if any(_has_polygon(block) for block in current) else []
            units.append(ProcessingUnit(
                id=unit_id,
                page=page_index,
                category=category,  # type: ignore[arg-type]
                bbox=bbox,
                member_ids=[b.id for b in current],
                detector_score=sum(b.score for b in current) / len(current),
                hint_text="\n".join(b.hint_text for b in current if b.hint_text).strip(),
                polygons=polygons,
            ))
            counter += 1
    return sorted(units, key=lambda u: (u.bbox.y1, u.bbox.x1))
