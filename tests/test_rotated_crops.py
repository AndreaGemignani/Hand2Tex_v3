import math

import pytest
from PIL import Image, ImageDraw

from app.models import BBox, PageLayout, ProcessingUnit
from app.services.crops import materialize_crops


def _unit(category, bbox, polygons):
    return ProcessingUnit("unit", 0, category, bbox, ["line"], .95, polygons=polygons)


def _crop(tmp_path, image, unit, padding=8):
    path = tmp_path / "page.png"
    image.save(path)
    page = PageLayout(0, image.width, image.height, str(path), units=[unit])
    materialize_crops(page, tmp_path / "crops", padding)
    assert unit.crop_path == str(tmp_path / "crops" / "unit.png")
    with Image.open(unit.crop_path) as saved:
        return saved.copy()


def _points(polygon):
    return list(zip(polygon[::2], polygon[1::2]))


def _colored_pixels(image, channel):
    return sum(pixel[channel] > 180 and all(value < 80 for index, value in enumerate(pixel) if index != channel)
               for pixel in image.getdata())


@pytest.mark.parametrize("category", ["text", "math"])
def test_rotated_line_is_upright_and_excludes_neighbour_inside_axis_envelope(tmp_path, category):
    image = Image.new("RGB", (190, 140), "white")
    polygon = [20, 30, 160, 79, 160, 91, 20, 42]
    neighbour = [20, 52, 160, 101, 160, 113, 20, 64]
    draw = ImageDraw.Draw(image)
    draw.polygon(_points(polygon), fill="red")
    draw.polygon(_points(neighbour), fill="blue")
    unit = _unit(category, BBox(20, 30, 160, 91), [polygon])
    crop = _crop(tmp_path, image, unit)
    assert crop.size == (math.ceil(math.hypot(140, 49)) + 16, 12 + 16)
    assert _colored_pixels(crop, 2) == 0
    assert _colored_pixels(crop, 0) > 140 * 8
    # The slanted top and bottom are horizontal after the transform.
    assert all(crop.getpixel((x, 14))[0] > 180 and crop.getpixel((x, 14))[2] < 80 for x in range(12, crop.width - 12))
    assert crop.getpixel((0, 0)) == (255, 255, 255)
    assert unit.bbox == BBox(20, 30, 160, 91)


def test_multiple_lines_keep_member_order_and_white_gap(tmp_path):
    image = Image.new("RGB", (180, 150), "white")
    first = [15, 90, 135, 114, 135, 124, 15, 100]
    second = [25, 20, 105, 36, 105, 46, 25, 30]
    draw = ImageDraw.Draw(image)
    draw.polygon(_points(first), fill="red")
    draw.polygon(_points(second), fill="blue")
    unit = _unit("text", BBox(15, 20, 135, 124), [first, second])
    crop = _crop(tmp_path, image, unit, padding=6)
    assert crop.size == (math.ceil(math.hypot(120, 24)) + 12, 10 + 6 + 10 + 12)
    assert crop.getpixel((25, 10))[0] > 180
    assert crop.getpixel((25, 26))[2] > 180
    assert all(crop.getpixel((x, 18)) == (255, 255, 255) for x in range(crop.width))
    assert _colored_pixels(crop, 0) > 500
    assert _colored_pixels(crop, 2) > 500


def test_overlapping_quads_keep_shared_ink_once_and_preserve_line_order(tmp_path):
    image = Image.new("RGB", (150, 100), "white")
    first = [20, 20, 120, 20, 120, 50, 20, 50]
    second = [20, 40, 120, 40, 120, 70, 20, 70]
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 27, 105, 30), fill="red")
    draw.rectangle((35, 44, 105, 47), fill="blue")
    draw.rectangle((35, 60, 105, 63), fill=(0, 255, 0))
    unit = _unit("text", BBox(20, 20, 120, 70), [first, second])
    crop = _crop(tmp_path, image, unit)
    assert crop.size == (100 + 16, 50 + 16)
    assert _colored_pixels(crop, 2) == _colored_pixels(image, 2)
    color_rows = []
    for channel in (0, 2, 1):
        rows = [y for y in range(crop.height) for x in range(crop.width)
                if crop.getpixel((x, y))[channel] > 180
                and all(value < 80 for index, value in enumerate(crop.getpixel((x, y))) if index != channel)]
        color_rows.append(sum(rows) / len(rows))
    assert color_rows[0] < color_rows[1] < color_rows[2]


def test_overlapping_rotated_lines_deskew_union_once(tmp_path):
    image = Image.new("RGB", (150, 120), "white")
    first = [20, 15, 120, 45, 120, 80, 20, 50]
    second = [20, 40, 120, 70, 120, 105, 20, 75]
    shared_stroke = [35, 48, 105, 69, 105, 72, 35, 51]
    ImageDraw.Draw(image).polygon(_points(shared_stroke), fill="blue")
    unit = _unit("text", BBox(20, 15, 120, 105), [first, second])
    crop = _crop(tmp_path, image, unit)
    assert .6 * _colored_pixels(image, 2) < _colored_pixels(crop, 2) < 1.3 * _colored_pixels(image, 2)
    blue_rows = [y for y in range(crop.height) for x in range(crop.width)
                 if crop.getpixel((x, y))[2] > 180 and crop.getpixel((x, y))[0] < 80]
    assert max(blue_rows) - min(blue_rows) <= 6
    assert crop.getpixel((0, 0)) == (255, 255, 255)


def test_quads_sharing_only_a_slanted_boundary_still_pack_separate_lines(tmp_path):
    image = Image.new("RGB", (150, 100), "white")
    first = [20, 20, 120, 50, 120, 70, 20, 40]
    second = [20, 40, 120, 70, 120, 90, 20, 60]
    draw = ImageDraw.Draw(image)
    draw.polygon(_points(first), fill="red")
    draw.polygon(_points(second), fill="blue")
    unit = _unit("text", BBox(20, 20, 120, 90), [first, second])
    crop = _crop(tmp_path, image, unit)
    assert crop.size == (math.ceil(math.hypot(100, 30)) + 16, 20 + 8 + 20 + 16)
    assert all(crop.getpixel((x, 31)) == (255, 255, 255) for x in range(crop.width))


@pytest.mark.parametrize("category", ["figure", "table", "unknown"])
def test_non_ocr_categories_preserve_original_axis_crop_pixels(tmp_path, category):
    image = Image.new("RGB", (70, 70))
    image.putdata([(x, y, (x + y) % 256) for y in range(70) for x in range(70)])
    unit = _unit(category, BBox(15, 20, 45, 50), [[15, 20, 45, 30, 45, 50, 15, 40]])
    crop = _crop(tmp_path, image, unit, padding=4)
    expected = image.crop((11, 16, 49, 54))
    assert crop.size == expected.size
    assert crop.tobytes() == expected.tobytes()


@pytest.mark.parametrize("polygons", [
    [], None, 42,
    [[1, 2, 3]],
    [[10, 10, 50, 10, 50, float("nan"), 10, 30]],
    [[10, 10, 50, 30, 50, 10, 10, 30]],
    [[10, 10, 50, 10, 50, 10, 10, 10]],
    [[10, 10, 10, 30, 50, 30, 50, 10]],
    [[10, 10, 50, 10, 50, 30, 10, 30], None],
    [[-100, -100, -50, -100, -50, -50, -100, -50]],
])
def test_empty_or_malformed_geometry_falls_back_without_losing_original_pixels(tmp_path, polygons):
    image = Image.new("RGB", (70, 70))
    image.putdata([(x, y, (x + y) % 256) for y in range(70) for x in range(70)])
    unit = _unit("text", BBox(15, 20, 45, 50), polygons)
    crop = _crop(tmp_path, image, unit, padding=4)
    expected = image.crop((11, 16, 49, 54))
    assert crop.size == expected.size
    assert crop.tobytes() == expected.tobytes()


def test_quad_touching_page_edge_has_white_padding_and_keeps_visible_ink(tmp_path):
    image = Image.new("RGB", (100, 60), "white")
    polygon = [-8, -6, 80, 14, 80, 26, -8, 6]
    ImageDraw.Draw(image).polygon(_points(polygon), fill="red")
    unit = _unit("text", BBox(0, 0, 80, 26), [polygon])
    crop = _crop(tmp_path, image, unit)
    assert _colored_pixels(crop, 0) > 300
    assert all(crop.getpixel((x, 0)) == (255, 255, 255) for x in range(crop.width))
    assert all(crop.getpixel((0, y)) == (255, 255, 255) for y in range(crop.height))
    assert (0, 0, 0) not in crop.getdata()


def test_zero_padding_keeps_the_whole_line_including_bottom_edge(tmp_path):
    image = Image.new("RGB", (70, 50), "white")
    polygon = [10, 10, 60, 10, 60, 30, 10, 30]
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 27, 50, 30), fill="black")
    unit = _unit("text", BBox(10, 10, 60, 30), [polygon])
    crop = _crop(tmp_path, image, unit, padding=0)
    assert crop.size == (50, 20)
    assert crop.getpixel((20, 19)) == (0, 0, 0)
