"""Find local source illustrations without reproducing the scanned page.

Geometry and thresholding select regions only. Accepted crops retain the source
pixels (apart from OCR regions erased to prevent duplicate handwritten text).
This is deliberately conservative: uncertain or overlarge residuals are reported
so the pipeline can include a separate source attachment instead of a background.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFilter

from app.models import BBox, LayoutBlock


@dataclass
class FigureExtraction:
    blocks: list[LayoutBlock]
    crops: dict[str, str]
    warnings: list[str]


def background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    rgb = rgb.resize((min(160, rgb.width), min(160, rgb.height)), Image.Resampling.NEAREST)
    pixels = list(rgb.getdata())
    neutral = [pixel for pixel in pixels if min(pixel) >= 180 and max(pixel) - min(pixel) <= 20]
    if not neutral:
        cutoff = sorted(sum(pixel) for pixel in pixels)[int((len(pixels) - 1) * .8)]
        neutral = [pixel for pixel in pixels if sum(pixel) >= cutoff]
    return tuple(int(median(pixel[channel] for pixel in neutral)) for channel in range(3))


def residual_image(image: Image.Image, blocks: list[LayoutBlock], *, exclude_figures: bool = False) -> Image.Image:
    result = image.convert("RGB").copy()
    bg = background_color(result)
    draw = ImageDraw.Draw(result)
    pad = max(3, round(min(result.size) * .0025))
    for block in blocks:
        if block.category not in {"text", "math", "table"} and not exclude_figures:
            continue
        if len(block.polygon) == 8 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in block.polygon):
            points = list(zip(block.polygon[::2], block.polygon[1::2]))
            draw.polygon(points, fill=bg)
            draw.line([*points, points[0]], fill=bg, width=pad * 2 + 1, joint="curve")
        else:
            box = block.bbox.expand(pad, result.width, result.height)
            draw.rectangle((int(box.x1), int(box.y1), int(box.x2), int(box.y2)), fill=bg)
    return result


def _components(mask: Image.Image) -> list[tuple[int, int, int, int]]:
    """Eight-connected components on a bounded, at most 960-pixel-wide mask."""
    width, height = mask.size
    pixels = bytearray(mask.tobytes())
    components = []
    for start in range(len(pixels)):
        if not pixels[start]:
            continue
        pixels[start] = 0
        queue = deque([start])
        left = right = start % width
        top = bottom = start // width
        while queue:
            point = queue.popleft()
            x, y = point % width, point // width
            left, right, top, bottom = min(left, x), max(right, x), min(top, y), max(bottom, y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = ny * width + nx
                    if pixels[neighbor]:
                        pixels[neighbor] = 0
                        queue.append(neighbor)
        components.append((left, top, right + 1, bottom + 1))
    return components


def _ink_mask(image: Image.Image) -> Image.Image:
    # Squared paper and colored highlighters are usually much lighter than pen
    # strokes. Threshold before reducing size so thin pen strokes survive.
    ink = image.convert("L").point(lambda value: 255 if value < 135 else 0)
    scale = min(1.0, 960 / max(image.size))
    if scale < 1:
        ink = ink.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BOX)
        ink = ink.point(lambda value: 255 if value >= 32 else 0)
    width, height = ink.size
    pixels = ink.tobytes()
    # Long straight rules/borders must not connect the whole page into a figure.
    rows = [y for y in range(height) if sum(bool(value) for value in pixels[y * width:(y + 1) * width]) > .72 * width]
    columns = [x for x in range(width) if sum(bool(pixels[y * width + x]) for y in range(height)) > .72 * height]
    def thin_runs(values: list[int], maximum: int) -> list[int]:
        # A dark page or a filled illustration is not a paper rule. Only remove
        # contiguous narrow runs of rows/columns that span most of the page.
        runs: list[list[int]] = []
        for value in values:
            if runs and value == runs[-1][-1] + 1:
                runs[-1].append(value)
            else:
                runs.append([value])
        return [value for run in runs if len(run) <= maximum for value in run]

    rows = thin_runs(rows, max(3, round(height * .006)))
    columns = thin_runs(columns, max(3, round(width * .006)))
    draw = ImageDraw.Draw(ink)
    for y in rows:
        draw.line((0, y, width, y), fill=0)
    for x in columns:
        draw.line((x, 0, x, height), fill=0)
    return ink


def extract_figures(page_path: Path, blocks: list[LayoutBlock], out_dir: Path, page_index: int) -> FigureExtraction:
    """Return bounded local source fragments, never a full-page residual image."""
    with Image.open(page_path) as source:
        residual = residual_image(source, blocks, exclude_figures=True)
    ink = _ink_mask(residual)
    # Join nearby pieces of a drawing, with a small radius relative to the page.
    radius = max(1, min(4, round(min(ink.size) * .004)))
    components = _components(ink.filter(ImageFilter.MaxFilter(2 * radius + 1)))
    accepted = []
    uncertain = False
    for left, top, right, bottom in components:
        local = ink.crop((left, top, right, bottom))
        count = local.histogram()[255]
        if count < max(10, ink.width * ink.height * .000015):
            continue  # Isolated specks, not content regions.
        width, height = right - left, bottom - top
        # Eliminate thin rules. Their ink has practically no two-dimensional
        # extent even after the small component-joining dilation.
        raw_box = local.getbbox()
        if raw_box is None:
            continue
        raw_width, raw_height = raw_box[2] - raw_box[0], raw_box[3] - raw_box[1]
        if min(raw_width, raw_height) <= 2 and max(raw_width, raw_height) >= .15 * min(ink.size):
            continue
        if width < .060 * ink.width or height < .035 * ink.height:
            uncertain = True
            continue
        if width * height > .40 * ink.width * ink.height or (width > .90 * ink.width and height > .60 * ink.height):
            uncertain = True
            continue
        # Pale, broken pieces of a photographed grid can survive the initial
        # threshold. A figure needs meaningful dark-ink support in its local
        # source region; uncertain pencil marks are kept in the source attachment.
        source_box = (math.floor(left * residual.width / ink.width), math.floor(top * residual.height / ink.height),
                      math.ceil(right * residual.width / ink.width), math.ceil(bottom * residual.height / ink.height))
        candidate = residual.crop(source_box).convert("L")
        if sum(candidate.histogram()[:80]) < max(20, candidate.width * candidate.height * .005):
            uncertain = True
            continue
        accepted.append((left, top, right, bottom))
    if len(accepted) > 32:
        # A page of missed handwriting must not become dozens of fake figures.
        accepted = []
        uncertain = True
    figure_blocks: list[LayoutBlock] = []
    crops: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    x_scale, y_scale = residual.width / ink.width, residual.height / ink.height
    for index, rect in enumerate(sorted(accepted, key=lambda box: (box[1], box[0]))):
        left, top, right, bottom = rect
        box = BBox(math.floor(left * x_scale), math.floor(top * y_scale), math.ceil(right * x_scale), math.ceil(bottom * y_scale)).expand(4, residual.width, residual.height)
        identifier = f"p{page_index:04d}_source_figure_{index:04d}"
        target = out_dir / f"{identifier}.png"
        residual.crop(tuple(map(int, (box.x1, box.y1, box.x2, box.y2)))).save(target, format="PNG", optimize=True)
        figure_blocks.append(LayoutBlock(identifier, page_index, "figure", "local_source_fragment", .6, box, provider="deterministic-figure-extraction"))
        crops[identifier] = str(target)
    warnings = []
    if accepted:
        warnings.append("Local unrecognized source fragments were preserved as illustrations; inspect them for missed handwriting.")
    if uncertain:
        warnings.append("Some residual marks could not be classified safely; the original page is attached separately for review.")
    return FigureExtraction(figure_blocks, crops, warnings)
