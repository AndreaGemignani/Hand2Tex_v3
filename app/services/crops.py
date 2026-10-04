from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw

from app.models import BBox, PageLayout, ProcessingUnit


@dataclass(frozen=True)
class ReviewCrop:
    path: str
    source_bbox: BBox
    target_bbox: BBox


def _line_quad(polygon: list[float], width: int, height: int) -> list[tuple[float, float]] | None:
    """Validate TL/TR/BR/BL corners, then clip them to the source page."""
    try:
        if not isinstance(polygon, (list, tuple)) or len(polygon) != 8:
            return None
        coordinates = [float(value) for value in polygon]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    points = list(zip(coordinates[::2], coordinates[1::2]))

    def convex(corners: list[tuple[float, float]]) -> bool:
        crosses = []
        for index in range(4):
            a, b, c = (corners[(index + offset) % 4] for offset in range(3))
            crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        # TL/TR/BR/BL is clockwise in image coordinates. Reject collapsed or
        # self-crossing quads rather than silently producing an incomplete line.
        return all(cross > 0.01 for cross in crosses)

    if not convex(points):
        return None
    points = [(min(width, max(0.0, x)), min(height, max(0.0, y))) for x, y in points]
    return points if convex(points) else None


def _upright_line(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image | None:
    top_left, top_right, bottom_right, bottom_left = points
    line_width = (math.dist(top_left, top_right) + math.dist(bottom_left, bottom_right)) / 2
    line_height = (math.dist(top_left, bottom_left) + math.dist(top_right, bottom_right)) / 2
    if min(line_width, line_height) < 1:
        return None
    size = (math.ceil(line_width), math.ceil(line_height))

    # Work on a small local source image, and whiten everything outside the
    # polygon before resampling. A neighbouring line inside its axis envelope
    # must never become OCR input for this line.
    left = max(0, math.floor(min(point[0] for point in points)) - 2)
    top = max(0, math.floor(min(point[1] for point in points)) - 2)
    right = min(image.width, math.ceil(max(point[0] for point in points)) + 2)
    bottom = min(image.height, math.ceil(max(point[1] for point in points)) + 2)
    source = image.crop((left, top, right, bottom))
    local = [(x - left, y - top) for x, y in points]
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).polygon(local, fill=255)
    source = Image.composite(source, Image.new("RGB", source.size, "white"), mask)
    # Pillow QUAD expects top-left, bottom-left, bottom-right, top-right.
    quad = tuple(value for index in (0, 3, 2, 1) for value in local[index])
    return source.transform(size, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC, fillcolor="white")


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(points[index][0] * points[(index + 1) % len(points)][1]
                   - points[(index + 1) % len(points)][0] * points[index][1]
                   for index in range(len(points)))) / 2


def _intersection_area(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    # Clip one clockwise convex polygon against the other's four edges. Using
    # geometric area avoids treating a shared one-pixel raster boundary as ink
    # overlap, and does not allocate full-page masks for every candidate pair.
    clipped = first
    for index, edge_start in enumerate(second):
        edge_end = second[(index + 1) % 4]

        def side(point: tuple[float, float]) -> float:
            return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
                    - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]))

        if not clipped:
            return 0.0
        output = []
        previous = clipped[-1]
        previous_side = side(previous)
        for current in clipped:
            current_side = side(current)
            if (current_side >= 0) != (previous_side >= 0):
                ratio = previous_side / (previous_side - current_side)
                output.append((previous[0] + ratio * (current[0] - previous[0]),
                               previous[1] + ratio * (current[1] - previous[1])))
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
        clipped = output
    return _polygon_area(clipped) if clipped else 0.0


def _overlapping_lines(quads: list[list[tuple[float, float]]]) -> bool:
    for index, first in enumerate(quads):
        for second in quads[index + 1:]:
            left = max(min(x for x, _ in first), min(x for x, _ in second))
            top = max(min(y for _, y in first), min(y for _, y in second))
            right = min(max(x for x, _ in first), max(x for x, _ in second))
            bottom = min(max(y for _, y in first), max(y for _, y in second))
            if left >= right or top >= bottom:
                continue
            if _intersection_area(first, second) >= max(4, .02 * min(_polygon_area(first), _polygon_area(second))):
                return True
    return False


def _union_crop(image: Image.Image, quads: list[list[tuple[float, float]]], padding_px: int) -> Image.Image:
    points = [point for quad in quads for point in quad]
    left = max(0, math.floor(min(x for x, _ in points)))
    top = max(0, math.floor(min(y for _, y in points)))
    right = min(image.width, math.ceil(max(x for x, _ in points)))
    bottom = min(image.height, math.ceil(max(y for _, y in points)))
    source = image.crop((left, top, right, bottom))
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    angles = []
    for quad in quads:
        draw.polygon([(x - left, y - top) for x, y in quad], fill=255)
        angles.append(math.atan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]))
    source = Image.composite(source, Image.new("RGB", source.size, "white"), mask)
    padding = max(0, padding_px)
    crop = Image.new("RGB", (source.width + 2 * padding, source.height + 2 * padding), "white")
    crop.paste(source, (padding, padding))
    angle = math.degrees(math.atan2(sum(math.sin(value) for value in angles),
                                    sum(math.cos(value) for value in angles)))
    # Rotate the masked union once: overlapping detector boxes must not duplicate
    # strokes in a packed OCR image. Positive image slopes need a CCW deskew.
    return crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")


def _polygon_crop(image: Image.Image, polygons: list[list[float]], padding_px: int) -> Image.Image | None:
    if not isinstance(polygons, (list, tuple)) or not polygons:
        return None
    quads = []
    for polygon in polygons:
        points = _line_quad(polygon, image.width, image.height)
        if points is None:
            return None
        quads.append(points)
    if len(quads) > 1 and _overlapping_lines(quads):
        return _union_crop(image, quads, padding_px)
    lines = []
    for points in quads:
        line = _upright_line(image, points)
        if line is None:
            return None
        lines.append(line)
    padding = max(0, padding_px)
    gap = max(4, padding)
    width = max(line.width for line in lines) + 2 * padding
    height = sum(line.height for line in lines) + gap * (len(lines) - 1) + 2 * padding
    crop = Image.new("RGB", (width, height), "white")
    cursor = padding
    for line in lines:
        crop.paste(line, (padding, cursor))
        cursor += line.height + gap
    return crop


def materialize_crops(page: PageLayout, out_dir: Path, padding_px: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page.image_path) as image:
        image = image.convert("RGB")
        for unit in page.units:
            bbox = unit.bbox.expand(padding_px, page.width_px, page.height_px)
            left, top, right, bottom = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            if right <= left or bottom <= top:
                continue
            crop = None
            if unit.category in {"text", "math"}:
                crop = _polygon_crop(image, getattr(unit, "polygons", []), padding_px)
            if crop is None:
                crop = image.crop((left, top, right, bottom))
            path = out_dir / f"{unit.id}.png"
            crop.save(path, format="PNG", optimize=True)
            unit.crop_path = str(path)


def _review_line_height(unit: ProcessingUnit, width: int, height: int) -> float:
    """Estimate stroke-line height independently of a slanted axis envelope."""
    heights = []
    polygons = getattr(unit, "polygons", [])
    if isinstance(polygons, (list, tuple)):
        for polygon in polygons:
            points = _line_quad(polygon, width, height)
            if points is None:
                continue
            opposite_edges = [
                (math.dist(points[0], points[1]) + math.dist(points[2], points[3])) / 2,
                (math.dist(points[1], points[2]) + math.dist(points[3], points[0])) / 2,
            ]
            heights.append(min(opposite_edges))
    return median(heights) if heights else unit.bbox.height


def materialize_review_crops(page: PageLayout, out_dir: Path, padding_px: int = 8) -> dict[str, ReviewCrop]:
    """Keep source geometry for semantic review, separate from compact OCR input.

    Reviewers see the unwarped source envelope and a little surrounding context.
    Nothing is masked, resized, deskewed or packed: fractions, matrices and mixed
    prose/formulas retain their source positions. Target coordinates identify the
    assigned content inside the crop so adjacent context is not transcribed twice.
    Existing processing-unit geometry and primary crop paths remain unchanged.
    """
    reviews: dict[str, ReviewCrop] = {}
    try:
        source = Image.open(page.image_path)
    except (OSError, ValueError):
        return reviews
    with source:
        image = source.convert("RGB")
        out_dir.mkdir(parents=True, exist_ok=True)
        for unit in page.units:
            if unit.category not in {"text", "math", "table"}:
                continue
            coordinates = (unit.bbox.x1, unit.bbox.y1, unit.bbox.x2, unit.bbox.y2)
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in coordinates):
                continue
            target = unit.bbox.clamp(image.width, image.height)
            if target.width <= 0 or target.height <= 0:
                continue
            line_height = _review_line_height(unit, image.width, image.height)
            # One source line plus room for superscripts or a fraction denominator.
            # Caps keep unrelated neighboring rows/columns out of remote review.
            context = max(0.0, float(padding_px), line_height * 1.25)
            pad_x = min(context, image.width * .04)
            pad_y = min(context, image.height * .04)
            left = max(0, math.floor(target.x1 - pad_x))
            top = max(0, math.floor(target.y1 - pad_y))
            right = min(image.width, math.ceil(target.x2 + pad_x))
            bottom = min(image.height, math.ceil(target.y2 + pad_y))
            if right <= left or bottom <= top:
                continue
            path = out_dir / f"{unit.id}_review.png"
            try:
                image.crop((left, top, right, bottom)).save(path, format="PNG", optimize=True)
            except (OSError, ValueError):
                continue
            reviews[unit.id] = ReviewCrop(
                path=str(path),
                source_bbox=BBox(left, top, right, bottom),
                target_bbox=BBox(target.x1 - left, target.y1 - top, target.x2 - left, target.y2 - top),
            )
    return reviews
