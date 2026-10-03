from app.models import BBox, LayoutBlock
from app.services.grouping import build_processing_units
from app.services.layout_detector import category_for_label


def b(i, cat, y1, y2, x1=100, x2=700, score=.9):
    return LayoutBlock(f"b{i}", 0, cat, cat, score, BBox(x1, y1, x2, y2))


def test_label_mapping():
    assert category_for_label("formula") == "math"
    assert category_for_label("inline_formula") == "math"
    assert category_for_label("display_formula") == "math"
    assert category_for_label("table") == "table"
    assert category_for_label("figure") == "figure"
    assert category_for_label("paragraph_title") == "text"


def test_consecutive_formulas_are_grouped():
    blocks = [b(1, "math", 100, 160), b(2, "math", 170, 225), b(3, "text", 400, 500)]
    units = build_processing_units(blocks, 1000, 1400, 0)
    math = [u for u in units if u.category == "math"]
    assert len(math) == 1
    assert math[0].member_ids == ["b1", "b2"]


def test_text_inside_figure_is_not_double_processed():
    fig = LayoutBlock("fig", 0, "figure", "figure", .95, BBox(100,100,800,800))
    txt = LayoutBlock("txt", 0, "text", "text", .90, BBox(200,200,400,260))
    units = build_processing_units([fig, txt], 1000, 1400, 0)
    assert [u.category for u in units] == ["figure"]


def test_qwen_layout_math_heuristic():
    from app.pipeline import _qwen_words_to_blocks
    words = [
        {"location": [10,10,100,10,100,30,10,30], "text": "Hello world"},
        {"location": [10,40,100,40,100,60,10,60], "text": "F = ma"},
    ]
    blocks = _qwen_words_to_blocks(words, 0)
    assert [b.category for b in blocks] == ["text", "math"]
