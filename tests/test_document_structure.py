import pytest

from app.models import BBox, PageLayout, ProcessingUnit
from app.services.document_structure import reading_order, starts_paragraph


def unit(identifier, box, *, content="Text", category="text", members=1, polygons=None):
    return ProcessingUnit(identifier, 0, category, BBox(*box),
                          [f"{identifier}-{i}" for i in range(members)], .95,
                          decoded=content, polygons=polygons or [])


def page(units):
    return PageLayout(0, 1000, 1400, "source.png", units=units)


def ids(units):
    return [item.id for item in units]


@pytest.mark.parametrize("reverse", [False, True])
def test_reading_order_reads_whole_columns_beneath_spanning_heading(reverse):
    units = [
        unit("heading", (60, 40, 940, 90)),
        unit("left-first", (100, 120, 430, 180)),
        unit("right-first", (570, 120, 900, 180)),
        unit("left-second", (100, 220, 430, 280)),
        unit("right-second", (570, 220, 900, 280)),
        unit("footer", (60, 350, 940, 390)),
    ]
    if reverse:
        units.reverse()
    assert ids(reading_order(page(units))) == [
        "heading", "left-first", "left-second", "right-first", "right-second", "footer",
    ]


def test_spanning_midpage_figure_separates_two_column_bands():
    units = [
        unit("right-bottom", (570, 420, 900, 480)),
        unit("right-top", (570, 100, 900, 160)),
        unit("left-bottom", (100, 420, 430, 480)),
        unit("diagram", (100, 220, 900, 350), category="figure"),
        unit("left-top", (100, 100, 430, 160)),
    ]
    assert ids(reading_order(page(units))) == [
        "left-top", "right-top", "diagram", "left-bottom", "right-bottom",
    ]


def test_short_centered_heading_spans_a_gutter_without_filling_page_width():
    units = [
        unit("heading", (350, 40, 650, 90)),
        unit("left-first", (100, 120, 430, 180)),
        unit("right-first", (570, 120, 900, 180)),
        unit("left-second", (100, 230, 430, 290)),
        unit("right-second", (570, 230, 900, 290)),
    ]
    assert ids(reading_order(page(units))) == [
        "heading", "left-first", "left-second", "right-first", "right-second",
    ]


def test_single_column_orders_math_figure_and_prose_without_deduplicating_content():
    units = [
        unit("last", (100, 600, 900, 640), content="Repeated text"),
        unit("figure", (200, 280, 700, 520), category="figure"),
        unit("formula", (350, 200, 650, 250), category="math"),
        unit("first", (100, 100, 900, 140), content="Repeated text"),
    ]
    result = reading_order(page(units))
    assert ids(result) == ["first", "formula", "figure", "last"]
    assert [item.decoded for item in result if item.category == "text"] == ["Repeated text"] * 2
    assert all(any(item is source for source in units) for item in result)


def test_narrow_side_annotation_does_not_create_a_column():
    units = [
        unit("paragraph", (100, 100, 750, 180), members=2),
        unit("annotation", (820, 120, 880, 150)),
        unit("continuation", (100, 210, 750, 280), members=2),
    ]
    assert ids(reading_order(page(units))) == ["paragraph", "annotation", "continuation"]


def test_sloped_polygon_order_uses_centerline_instead_of_inflated_envelope_top():
    first = unit("first", (100, 80, 900, 420), polygons=[[100, 80, 900, 380, 900, 420, 100, 120]])
    # A steeper later line starts higher at its left corner but is lower through
    # the middle of the page. Axis-envelope sorting would put it first.
    second = unit("second", (100, 40, 900, 620), polygons=[[100, 40, 900, 580, 900, 620, 100, 80]])
    assert ids(reading_order(page([second, first]))) == ["first", "second"]


def test_geometry_ties_are_stable_without_removing_units():
    one = unit("a", (100, 100, 800, 130))
    two = unit("b", (100, 100, 800, 130))
    assert ids(reading_order(page([two, one]))) == ["a", "b"]


def test_six_line_ocr_chunk_continues_the_same_paragraph():
    first = unit("chunk-a", (100, 100, 850, 310), members=6)
    second = unit("chunk-b", (100, 320, 850, 355))
    assert not starts_paragraph(first, second, page([first, second]))


def test_short_last_line_does_not_create_an_automatic_paragraph():
    first = unit("chunk-a", (100, 100, 350, 135), content="end.")
    second = unit("chunk-b", (100, 145, 850, 180), content="Continues the same paragraph.")
    assert not starts_paragraph(first, second, page([first, second]))


def test_blank_source_line_and_indent_create_paragraph_boundaries():
    first = unit("chunk-a", (100, 100, 850, 130))
    after_gap = unit("after-gap", (100, 180, 850, 210))
    indented = unit("indented", (150, 140, 850, 170))
    assert starts_paragraph(first, after_gap, page([first, after_gap]))
    assert starts_paragraph(first, indented, page([first, indented]))


def test_column_transition_and_heading_size_start_new_paragraphs():
    left_bottom = unit("left", (100, 500, 430, 530))
    right_top = unit("right", (570, 100, 900, 130))
    heading = unit("heading", (100, 100, 900, 170))
    body = unit("body", (100, 180, 900, 210))
    assert starts_paragraph(left_bottom, right_top, page([left_bottom, right_top]))
    assert starts_paragraph(heading, body, page([heading, body]))


def test_polygon_line_gaps_ignore_overlapping_rotated_envelopes():
    first = unit("first", (100, 50, 900, 490), polygons=[[100, 50, 900, 450, 900, 490, 100, 90]])
    second = unit("second", (100, 100, 900, 540), polygons=[[100, 100, 900, 500, 900, 540, 100, 140]])
    assert not starts_paragraph(first, second, page([first, second]))


def test_malformed_polygon_metadata_falls_back_to_legacy_bounds():
    first = unit("first", (100, 100, 850, 130), polygons=[[float("nan")] * 8])
    second = unit("second", (100, 140, 850, 170))
    assert ids(reading_order(page([second, first]))) == ["first", "second"]
    assert not starts_paragraph(first, second, page([first, second]))


def test_empty_page_and_nonprose_boundaries():
    assert reading_order(page([])) == []
    text = unit("text", (100, 100, 850, 130))
    formula = unit("math", (300, 140, 650, 170), category="math")
    assert starts_paragraph(text, formula, page([text, formula]))
