import pytest

from app.models import BBox, LayoutBlock
from app.services.grouping import build_processing_units


def line(identifier, y, *, left=100, right=700, height=30, slope=0, category="text", polygon=True, score=.95):
    rise = slope * (right - left) / 2
    quad = [
        left, y - rise - height / 2, right, y + rise - height / 2,
        right, y + rise + height / 2, left, y - rise + height / 2,
    ]
    block = LayoutBlock(
        identifier, 0, category, f"qwen_layout_{category}", score,
        BBox(min(quad[::2]), min(quad[1::2]), max(quad[::2]), max(quad[1::2])),
    )
    if polygon:
        block.polygon = quad
    return block


def members(units):
    return [identifier for unit in units for identifier in unit.member_ids]


def test_adjacent_tilted_lines_survive_envelope_overlap_and_group_by_centerlines():
    # Their envelopes overlap by more than 93%, but the actual text strips do not.
    blocks = [line("first", 400, right=900, height=18, slope=.4), line("second", 420, right=900, height=18, slope=.4)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert len(units) == 1
    assert units[0].member_ids == ["first", "second"]
    assert units[0].polygons == [block.polygon for block in blocks]
    assert units[0].bbox == blocks[0].bbox.union(blocks[1].bbox)


def test_steep_tilt_uses_the_same_vertical_spacing_as_horizontal_lines():
    # Forty pixels between centers and thirty pixels of vertical text thickness
    # leave a ten-pixel gap, regardless of the baseline slope.
    blocks = [line("first", 400, height=30, slope=.8), line("second", 440, height=30, slope=.8)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert [unit.member_ids for unit in units] == [["first", "second"]]


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_score_duplicate_keeps_one_stable_representative(reverse):
    blocks = [line("a", 200), line("b", 200)]
    units = build_processing_units(blocks[::-1] if reverse else blocks, 1000, 1400, 0)
    assert members(units) == ["a"]


def test_duplicate_prefers_the_higher_score():
    blocks = [line("a", 200, score=.90), line("b", 200, score=.99)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert members(units) == ["b"]
    assert units[0].detector_score == .99


def test_text_in_figure_envelope_but_outside_its_polygon_is_preserved():
    figure = LayoutBlock("figure", 0, "figure", "figure", .99, BBox(0, 0, 200, 200))
    figure.polygon = [100, 0, 200, 100, 100, 200, 0, 100]
    text = line("corner-text", 20, left=10, right=30, height=20)
    units = build_processing_units([figure, text], 1000, 1400, 0)
    assert set(members(units)) == {"figure", "corner-text"}


@pytest.mark.parametrize("label", ["residual_background", "local_source_fragment"])
def test_source_fragment_envelope_never_suppresses_text(label):
    residual = LayoutBlock("residual", 0, "figure", label, 1, BBox(0, 0, 1000, 1400))
    text = line("text", 200)
    units = build_processing_units([residual, text], 1000, 1400, 0)
    assert set(members(units)) == {"residual", "text"}


def test_interleaved_columns_group_with_their_own_rows():
    blocks = [
        line(f"{column}-{row}", 100 + row * 40, left=left, right=right)
        for row in range(3)
        for column, left, right in [("left", 100, 400), ("right", 600, 900)]
    ]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert sorted(unit.member_ids for unit in units) == [
        ["left-0", "left-1", "left-2"], ["right-0", "right-1", "right-2"],
    ]
    assert all(len(unit.polygons) == 3 for unit in units)


def test_spanning_header_does_not_bridge_the_columns_below_it():
    blocks = [line("header", 100, left=50, right=950)] + [
        line(f"{column}-{row}", 140 + row * 40, left=left, right=right)
        for row in range(2)
        for column, left, right in [("left", 100, 430), ("right", 600, 920)]
    ]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert sorted(unit.member_ids for unit in units) == [
        ["header"], ["left-0", "left-1"], ["right-0", "right-1"],
    ]


def test_long_text_chain_is_bounded_without_one_ocr_call_per_line():
    blocks = [line(f"line-{i:02d}", 200 + i * 40) for i in range(13)]
    units = build_processing_units(blocks, 1000, 2400, 0)
    assert len(units) == 3
    assert sorted(members(units)) == [f"line-{i:02d}" for i in range(13)]
    assert all(len(unit.member_ids) <= 6 and unit.bbox.height <= .20 * 2400 for unit in units)


def test_cumulative_page_height_limit_splits_a_dense_long_crop():
    blocks = [line(f"line-{i:02d}", 100 + i * 45, height=40) for i in range(6)]
    units = build_processing_units(blocks, 1000, 1000, 0)
    assert len(units) == 2
    assert all(unit.bbox.height <= 200 for unit in units)
    assert len(members(units)) == 6


def test_paragraph_gap_and_large_heading_change_start_new_text_groups():
    blocks = [line("one", 100), line("two", 140), line("after-gap", 230), line("heading", 285, height=70)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert [unit.member_ids for unit in units] == [["one", "two"], ["after-gap"], ["heading"]]


def test_consecutive_formulas_keep_the_existing_grouping_range():
    blocks = [line("math-one", 100, category="math"), line("math-two", 165, category="math")]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert len(units) == 1
    assert units[0].member_ids == ["math-one", "math-two"]


def test_group_mixed_geometry_carries_a_quad_for_each_member():
    blocks = [line("polygon", 100), line("legacy", 140, polygon=False)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    assert len(units) == 1
    assert units[0].polygons == [blocks[0].polygon, [100, 125, 700, 125, 700, 155, 100, 155]]


def test_legacy_groups_keep_empty_polygon_metadata():
    units = build_processing_units([line("legacy", 100, polygon=False)], 1000, 1400, 0)
    assert units[0].polygons == []
