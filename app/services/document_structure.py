"""Use source geometry to recover reading order, not output coordinates.

OCR processing units are request-sized chunks.  Their boundaries have no
typographic meaning: adjacent chunks can belong to the same paragraph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median

from app.models import BBox, PageLayout, ProcessingUnit


@dataclass(frozen=True)
class _Line:
    left: float
    right: float
    center_y: float
    height: float
    slope: float = 0.0

    def y_at(self, x: float) -> float:
        return self.center_y + self.slope * (x - (self.left + self.right) / 2)


def _polygon_line(polygon: list[float]) -> _Line | None:
    if len(polygon) != 8 or not all(math.isfinite(value) for value in polygon):
        return None
    points = list(zip(polygon[::2], polygon[1::2]))
    lengths = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    short = 0 if lengths[0] + lengths[2] <= lengths[1] + lengths[3] else 1
    endpoints = [
        ((points[i][0] + points[(i + 1) % 4][0]) / 2,
         (points[i][1] + points[(i + 1) % 4][1]) / 2)
        for i in (short, (short + 2) % 4)
    ]
    left, right = sorted(endpoints)
    dx, dy = right[0] - left[0], right[1] - left[1]
    area = abs(sum(
        x * points[(i + 1) % 4][1] - y * points[(i + 1) % 4][0]
        for i, (x, y) in enumerate(points)
    )) / 2
    if dx <= 0 or area <= 0 or abs(dy) > dx:
        return None
    return _Line(left[0], right[0], (left[1] + right[1]) / 2,
                 max(1.0, area / dx), dy / dx)


def _lines(unit: ProcessingUnit) -> list[_Line]:
    lines = [line for polygon in unit.polygons if (line := _polygon_line(polygon))]
    if lines:
        return sorted(lines, key=lambda line: (line.center_y, line.left))
    # Legacy layout detectors supply one envelope for a multi-line crop.
    # Estimate line thickness instead of treating the entire crop as one line.
    count = max(1, len(unit.member_ids))
    height = max(1.0, unit.bbox.height / count)
    return [
        _Line(unit.bbox.x1, unit.bbox.x2, unit.bbox.y1 + (i + .5) * height, height)
        for i in range(count)
    ]


def _flow_bounds(unit: ProcessingUnit) -> BBox:
    if unit.category not in {"text", "unknown", "math"} or not unit.polygons:
        return unit.bbox
    lines = _lines(unit)
    # A sloped line's axis envelope can overlap several subsequent lines.
    # Its centerline and true thickness give more useful semantic ordering.
    return BBox(min(line.left for line in lines),
                min(line.center_y - line.height / 2 for line in lines),
                max(line.right for line in lines),
                max(line.center_y + line.height / 2 for line in lines))


def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(left[1], right[0]) for left, right in zip(merged, merged[1:])]


def reading_order(page: PageLayout) -> list[ProcessingUnit]:
    """Return every unit once, reading each column before the column to its right.

Spanning headings, formulas and figures divide the page into successive bands.
Within a band a substantial vertical gutter separates columns. No decoded
content is compared or removed, including legitimately repeated expressions.
"""
    bounds = {id(unit): _flow_bounds(unit) for unit in page.units}

    def key(unit: ProcessingUnit) -> tuple[float, float, str]:
        box = bounds[id(unit)]
        return box.y1, box.x1, unit.id

    def order(units: list[ProcessingUnit]) -> list[ProcessingUnit]:
        if len(units) <= 1:
            return list(units)
        x_gaps = _gaps([(bounds[id(unit)].x1, bounds[id(unit)].x2) for unit in units])
        for left, right in sorted(x_gaps, key=lambda gap: (-(gap[1] - gap[0]), gap[0])):
            if right - left < .035 * page.width_px:
                continue
            first = [unit for unit in units if bounds[id(unit)].x2 <= left]
            second = [unit for unit in units if bounds[id(unit)].x1 >= right]
            if not first or not second:
                continue
            # A short annotation beside a paragraph is not a separate column.
            widths = [max(bounds[id(unit)].x2 for unit in group) -
                      min(bounds[id(unit)].x1 for unit in group)
                      for group in (first, second)]
            if min(widths) < .12 * page.width_px:
                continue
            spans = [(min(bounds[id(unit)].y1 for unit in group),
                      max(bounds[id(unit)].y2 for unit in group))
                     for group in (first, second)]
            if min(span[1] for span in spans) <= max(span[0] for span in spans):
                continue
            return order(first) + order(second)
        # A spanning heading blocks vertical cuts, but whitespace above/below it
        # allows the columns in the remaining band to be considered separately.
        y_gaps = _gaps([(bounds[id(unit)].y1, bounds[id(unit)].y2) for unit in units])
        if y_gaps:
            boxes = [bounds[id(unit)] for unit in units]
            gutters: list[float] = []
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    if min(a.width, b.width) < .12 * page.width_px:
                        continue
                    if min(a.y2, b.y2) <= max(a.y1, b.y1):
                        continue
                    left, right = sorted((a, b), key=lambda box: box.x1)
                    if right.x1 - left.x2 >= .035 * page.width_px:
                        gutters.append((left.x2 + right.x1) / 2)

            def band_boundary(gap: tuple[float, float]) -> bool:
                return any(
                    (box.width >= .60 * page.width_px or
                     any(box.x1 < gutter < box.x2 for gutter in gutters)) and
                    (box.y2 == gap[0] or box.y1 == gap[1])
                    for box in boxes
                )

            top, bottom = max(y_gaps, key=lambda gap: (
                band_boundary(gap), gap[1] - gap[0], -gap[0],
            ))
            before = [unit for unit in units if bounds[id(unit)].y2 <= top]
            after = [unit for unit in units if bounds[id(unit)].y1 >= bottom]
            return order(before) + order(after)
        return sorted(units, key=key)

    return order(list(page.units))


def starts_paragraph(previous: ProcessingUnit, current: ProcessingUnit, page: PageLayout) -> bool:
    """Infer a paragraph boundary between consecutive prose transport chunks.

Explicit blank lines inside decoded text are handled by the renderer. Here
source spacing, indentation, heading size and column transitions matter; the
number of crops or API requests does not.
"""
    if previous.category not in {"text", "unknown"} or current.category not in {"text", "unknown"}:
        return True
    a, b = _lines(previous)[-1], _lines(current)[0]
    overlap = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    if overlap / max(1.0, min(a.right - a.left, b.right - b.left)) < .25:
        return True
    x = (max(a.left, b.left) + min(a.right, b.right)) / 2
    center_step = b.y_at(x) - a.y_at(x)
    height = median([a.height, b.height])
    # Moving to the top of the next column is a new paragraph even when its
    # bounding box happens to overlap a wide last line of the preceding column.
    if center_step <= -.5 * height:
        return True
    if max(a.height, b.height) > 1.75 * min(a.height, b.height):
        return True
    gap = center_step - (a.height + b.height) / 2
    if gap > max(.008 * page.height_px, .85 * height):
        return True
    left_reference = min(line.left for line in _lines(previous))
    # Positive indentation starts a paragraph. A short final line can end an
    # ordinary crop, so its right edge alone must never create a boundary.
    return b.left - left_reference > max(.025 * page.width_px, .8 * height)
