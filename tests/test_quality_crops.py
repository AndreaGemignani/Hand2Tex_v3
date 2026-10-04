import math

import pytest
from PIL import Image, ImageDraw

from app.models import BBox, PageLayout, ProcessingUnit
from app.services.crops import materialize_crops, materialize_review_crops


def source_page(tmp_path, image, units):
    path = tmp_path / "source.png"
    image.save(path)
    return PageLayout(0, image.width, image.height, str(path), units=units)


def unit(identifier, category, box, polygons=None):
    return ProcessingUnit(identifier, 0, category, box, [identifier], .95, polygons=polygons or [])


def test_context_review_preserves_source_pixels_and_primary_ocr_crop(tmp_path):
    image = Image.new("RGB", (240, 240), "white")
    target = [30, 60, 190, 100, 190, 120, 30, 80]
    neighbor = [30, 90, 190, 130, 190, 150, 30, 110]
    draw = ImageDraw.Draw(image)
    draw.polygon(list(zip(target[::2], target[1::2])), fill="red")
    draw.polygon(list(zip(neighbor[::2], neighbor[1::2])), fill="blue")
    text = unit("paragraph", "text", BBox(30, 60, 190, 120), [target])
    page = source_page(tmp_path, image, [text])
    materialize_crops(page, tmp_path / "ocr")
    original_path = text.crop_path
    original_bytes = (tmp_path / "ocr/paragraph.png").read_bytes()
    review = materialize_review_crops(page, tmp_path / "review")[text.id]
    assert text.crop_path == original_path
    assert text.bbox == BBox(30, 60, 190, 120)
    assert text.polygons == [target]
    assert (tmp_path / "ocr/paragraph.png").read_bytes() == original_bytes
    box = review.source_bbox
    with Image.open(review.path) as crop:
        assert crop.tobytes() == image.crop((box.x1, box.y1, box.x2, box.y2)).tobytes()
        # Neighbour/context ink remains at its original angle; it is never masked.
        assert crop.getpixel((80 - int(box.x1), 115 - int(box.y1))) == image.getpixel((80, 115)) == (0, 0, 255)
    assert review.target_bbox == BBox(30 - box.x1, 60 - box.y1, 190 - box.x1, 120 - box.y1)


def test_fraction_numerator_and_denominator_keep_vertical_and_horizontal_alignment(tmp_path):
    image = Image.new("RGB", (500, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((220, 310, 230, 325), fill="red")  # Numerator 1.
    draw.line((205, 330, 245, 330), fill="black", width=2)
    draw.rectangle((210, 342, 238, 354), fill="blue")  # Root/denominator.
    numerator = unit("numerator", "math", BBox(200, 300, 260, 330), [[200, 300, 260, 300, 260, 330, 200, 330]])
    denominator = unit("denominator", "text", BBox(205, 338, 245, 356))
    page = source_page(tmp_path, image, [numerator, denominator])
    reviews = materialize_review_crops(page, tmp_path / "review")
    review = reviews["numerator"]
    assert review.source_bbox.y2 >= 356
    with Image.open(review.path) as crop:
        x0, y0 = int(review.source_bbox.x1), int(review.source_bbox.y1)
        assert crop.getpixel((225 - x0, 315 - y0)) == (255, 0, 0)
        assert crop.getpixel((225 - x0, 330 - y0)) == (0, 0, 0)
        assert crop.getpixel((225 - x0, 350 - y0)) == (0, 0, 255)
        assert crop.tobytes() == image.crop(tuple(map(int, (review.source_bbox.x1, review.source_bbox.y1, review.source_bbox.x2, review.source_bbox.y2)))).tobytes()
    assert reviews["denominator"].target_bbox.y1 >= 0


def test_matrix_entries_and_mixed_prose_are_not_stacked_or_left_aligned(tmp_path):
    image = Image.new("RGB", (400, 600), "white")
    draw = ImageDraw.Draw(image)
    # Four entries in a 2x2 matrix plus adjacent handwritten prose.
    draw.rectangle((100, 200, 110, 210), fill="red")
    draw.rectangle((180, 200, 190, 210), fill="blue")
    draw.rectangle((100, 240, 110, 250), fill="green")
    draw.rectangle((180, 240, 190, 250), fill="black")
    draw.line((230, 220, 300, 220), fill="purple", width=4)
    polygons = [[90, 190, 200, 190, 200, 215, 90, 215], [90, 235, 200, 235, 200, 260, 90, 260]]
    matrix = unit("matrix", "math", BBox(90, 190, 310, 260), polygons)
    page = source_page(tmp_path, image, [matrix])
    review = materialize_review_crops(page, tmp_path / "review")[matrix.id]
    x0, y0 = int(review.source_bbox.x1), int(review.source_bbox.y1)
    with Image.open(review.path) as crop:
        assert crop.getpixel((105 - x0, 205 - y0)) == (255, 0, 0)
        assert crop.getpixel((185 - x0, 205 - y0)) == (0, 0, 255)
        assert crop.getpixel((105 - x0, 245 - y0)) == (0, 128, 0)
        assert crop.getpixel((185 - x0, 245 - y0)) == (0, 0, 0)
        assert crop.getpixel((250 - x0, 220 - y0)) == (128, 0, 128)


def test_fractional_geometry_and_page_edge_are_clipped_without_resampling(tmp_path):
    image = Image.new("RGB", (100, 200))
    image.putdata([(x, y, (x + y) % 256) for y in range(200) for x in range(100)])
    text = unit("edge", "text", BBox(-2.25, 0.5, 35.75, 28.125))
    page = source_page(tmp_path, image, [text])
    review = materialize_review_crops(page, tmp_path / "review")[text.id]
    assert review.source_bbox == BBox(0, 0, 40, 37)
    assert review.target_bbox == BBox(0, .5, 35.75, 28.125)
    assert text.bbox == BBox(-2.25, .5, 35.75, 28.125)
    with Image.open(review.path) as crop:
        assert crop.tobytes() == image.crop((0, 0, 40, 37)).tobytes()


@pytest.mark.parametrize("box", [BBox(1, 2, 1, 5), BBox(40, 50, 20, 10), BBox(-20, -20, -5, -5), BBox(1, math.nan, 20, 30)])
def test_invalid_target_geometry_returns_no_review_crop(tmp_path, box):
    image = Image.new("RGB", (100, 200), "white")
    text = unit("invalid", "text", box)
    page = source_page(tmp_path, image, [text])
    assert materialize_review_crops(page, tmp_path / "review") == {}
    assert text.crop_path == ""


def test_malformed_polygons_use_target_envelope_and_context_stays_bounded(tmp_path):
    image = Image.new("RGB", (300, 600), "white")
    table = unit("table", "table", BBox(50, 100, 250, 500), [[1, 2, math.nan]])
    figure = unit("figure", "figure", BBox(30, 30, 70, 70))
    page = source_page(tmp_path, image, [table, figure])
    review = materialize_review_crops(page, tmp_path / "review")[table.id]
    assert review.source_bbox == BBox(38, 76, 262, 524)
    assert len(list((tmp_path / "review").glob("*.png"))) == 1


def test_missing_source_returns_no_review(tmp_path):
    page = PageLayout(0, 100, 200, str(tmp_path / "missing.png"), units=[unit("text", "text", BBox(10, 10, 60, 40))])
    assert materialize_review_crops(page, tmp_path / "review") == {}
