import json
import math

import httpx
import pytest
from PIL import Image, ImageDraw

from app.config import Settings
from app.models import BBox, LayoutBlock
from app.pipeline import _background_color, _make_residual_page, _qwen_words_to_blocks
from app.services.qwen_ocr import QwenOCRClient


async def locate(tmp_path, content, page_size=(800, 1200)):
    path = tmp_path / "page.png"
    Image.new("RGB", page_size, "white").save(path)

    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = Settings(trustedrouter_api_key="test-polygons")
    return await QwenOCRClient(settings, transport=httpx.MockTransport(handler)).locate_text_lines(path)


@pytest.mark.asyncio
async def test_rotated_csv_preserves_thin_line_polygon_and_original_placement_envelope(tmp_path):
    angle = math.radians(88)
    corners = []
    for dx, dy in [(-10, -300), (10, -300), (10, 300), (-10, 300)]:
        corners.append(((500 + dx * math.cos(angle) - dy * math.sin(angle)) * .8,
                        (500 + dx * math.sin(angle) + dy * math.cos(angle)) * 1.2))
    # The long axis was the native rectangle's height; rotate its corner sequence.
    expected = [value for point in corners[-1:] + corners[:-1] for value in point]
    result = await locate(tmp_path, "500,500,20,600,88")
    word = result.words_info[0]
    assert word["polygon"] == pytest.approx(expected)
    xs, ys = expected[::2], expected[1::2]
    assert word["location"] == pytest.approx([
        min(xs), min(ys), max(xs), min(ys), max(xs), max(ys), min(xs), max(ys),
    ])
    block = _qwen_words_to_blocks(result.words_info, 0)[0]
    assert block.polygon == pytest.approx(expected)
    assert [block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2] == pytest.approx([min(xs), min(ys), max(xs), max(ys)])


@pytest.mark.asyncio
async def test_native_pixel_quad_stays_in_pixels_and_keeps_rotation(tmp_path):
    polygon = [40, 100, 300, 80, 306, 120, 46, 140]
    content = json.dumps({"words_info": [{"location": polygon, "text": "Synthetic line"}]})
    result = await locate(tmp_path, content)
    assert result.words_info[0]["polygon"] == polygon
    assert result.words_info[0]["location"] == [40, 80, 306, 80, 306, 140, 40, 140]


@pytest.mark.asyncio
async def test_nonfinite_native_corner_does_not_become_a_usable_polygon(tmp_path):
    content = json.dumps({"words_info": [{"location": [40, 100, 300, 80, float("nan"), 120, 46, 140]}]})
    result = await locate(tmp_path, content)
    assert result.words_info == []


@pytest.mark.asyncio
async def test_page_edge_polygons_are_clipped_without_changing_envelope_contract(tmp_path):
    result = await locate(tmp_path, "10,10,40,100,0")
    word = result.words_info[0]
    assert word["polygon"] == pytest.approx([0, 0, 24, 0, 24, 72, 0, 72])
    assert word["location"] == pytest.approx(word["polygon"])


def test_neutral_paper_color_is_not_tinted_by_highlighted_corners():
    image = Image.new("RGB", (200, 200), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    for x, y in [(0, 0), (180, 0), (0, 180), (180, 180)]:
        draw.rectangle((x, y, x + 19, y + 19), fill=(255, 255, 90))
    draw.rectangle((30, 40, 170, 55), fill="black")
    assert _background_color(image) == (248, 248, 248)


def test_residual_erases_line_polygon_but_preserves_drawing_inside_envelope(tmp_path):
    source = tmp_path / "page.png"
    result = tmp_path / "residual.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    polygon = [10, 40, 80, 10, 90, 30, 20, 60]
    draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill="black")
    draw.rectangle((55, 50, 70, 58), fill="blue")
    image.save(source)
    block = LayoutBlock("line", 0, "text", "qwen", .95, BBox(10, 10, 90, 60), polygon=polygon)
    _make_residual_page(source, [block], result)
    with Image.open(result) as residual:
        assert residual.getpixel((45, 30)) == (255, 255, 255)
        assert residual.getpixel((60, 54)) == (0, 0, 255)


def test_figure_blocks_are_preserved_in_residual_layer(tmp_path):
    source = tmp_path / "figure.png"
    result = tmp_path / "residual.png"
    image = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 30, 30), fill="blue")
    image.save(source)
    block = LayoutBlock("figure", 0, "figure", "figure", .95, BBox(10, 10, 30, 30))
    _make_residual_page(source, [block], result)
    with Image.open(result) as residual:
        assert residual.tobytes() == image.tobytes()
