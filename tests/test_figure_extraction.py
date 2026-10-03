import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import app.pipeline as pipeline_module
from app.config import Settings
from app.models import BBox, LayoutBlock
from app.pipeline import Hand2TeXPipeline
from app.services.figure_extraction import extract_figures
from app.services.qwen_ocr import DecodeResult, LocateResult


def save_page(tmp_path, drawing, size=(400, 600)):
    path = tmp_path / "page.png"
    image = Image.new("RGB", size, "white")
    drawing(ImageDraw.Draw(image))
    image.save(path)
    return path, image


def test_blank_and_squared_paper_never_become_a_figure(tmp_path):
    def paper(draw):
        for x in range(0, 400, 20):
            draw.line((x, 0, x, 599), fill=(180, 205, 220))
        for y in range(0, 600, 20):
            draw.line((0, y, 399, y), fill=(180, 205, 220))
        draw.rectangle((0, 0, 399, 599), outline="black", width=2)

    path, _ = save_page(tmp_path, paper)
    result = extract_figures(path, [], tmp_path / "figures", 0)
    assert result.blocks == []
    assert result.crops == {}
    assert result.warnings == []


def test_broken_photographed_grid_does_not_become_a_local_illustration(tmp_path):
    def paper(draw):
        for x in range(70, 350, 20):
            draw.line((x, 100, x, 500), fill=(100, 100, 100))
        for y in range(100, 510, 20):
            draw.line((70, y, 350, y), fill=(100, 100, 100))
        # OCR masking breaks the rules into irregular short fragments.
        for y in range(115, 480, 55):
            draw.rectangle((50, y, 310, y + 35), fill="white")

    path, _ = save_page(tmp_path, paper)
    result = extract_figures(path, [], tmp_path / "figures", 0)
    assert result.blocks == []


def test_local_drawing_crop_preserves_true_source_pixels_without_fullpage(tmp_path):
    def content(draw):
        draw.rectangle((30, 30, 320, 55), fill="black")
        draw.rectangle((200, 200, 310, 290), outline="blue", width=4)
        draw.line((200, 290, 310, 200), fill=(10, 60, 120), width=3)

    path, image = save_page(tmp_path, content)
    blocks = [LayoutBlock("text", 0, "text", "text", .95, BBox(30, 30, 320, 55))]
    result = extract_figures(path, blocks, tmp_path / "figures", 0)
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.bbox.area < .1 * 400 * 600
    assert block.bbox.y1 > 55
    with Image.open(result.crops[block.id]) as crop:
        assert crop.tobytes() == image.crop(tuple(map(int, (block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2)))).tobytes()
    assert result.warnings  # Source fragments can include missed text, so do not claim OCR completeness.


def test_explicit_figures_are_not_extracted_a_second_time(tmp_path):
    path, _ = save_page(tmp_path, lambda draw: draw.rectangle((100, 200, 300, 350), outline="black", width=4))
    blocks = [LayoutBlock("f", 0, "figure", "figure", .95, BBox(100, 200, 300, 350))]
    assert extract_figures(path, blocks, tmp_path / "figures", 0).blocks == []


def test_grid_and_ocr_line_removal_do_not_erase_drawing_outside_rotated_polygon(tmp_path):
    polygon = [20, 140, 180, 90, 190, 110, 30, 160]
    def content(draw):
        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill="black")
        draw.rectangle((125, 135, 170, 175), outline="blue", width=4)

    path, image = save_page(tmp_path, content)
    blocks = [LayoutBlock("text", 0, "text", "text", .95, BBox(20, 90, 190, 160), polygon=polygon)]
    result = extract_figures(path, blocks, tmp_path / "figures", 0)
    assert len(result.blocks) == 1
    block = result.blocks[0]
    with Image.open(result.crops[block.id]) as crop:
        x, y = int(170 - block.bbox.x1), int(170 - block.bbox.y1)
        assert crop.getpixel((x, y)) == image.getpixel((170, 170)) == (0, 0, 255)


def test_page_sized_residual_is_warned_and_never_exported_as_background(tmp_path):
    path, _ = save_page(tmp_path, lambda draw: draw.rectangle((10, 10, 390, 590), fill="black"))
    result = extract_figures(path, [], tmp_path / "figures", 0)
    assert result.blocks == []
    assert result.warnings


def test_excessive_fragment_count_is_bounded_and_warned(tmp_path):
    def marks(draw):
        for x in range(20, 560, 60):
            for y in range(20, 940, 80):
                draw.rectangle((x, y, x + 34, y + 34), outline="black", width=3)

    path, _ = save_page(tmp_path, marks, size=(600, 1000))
    result = extract_figures(path, [], tmp_path / "figures", 0)
    assert result.blocks == []
    assert result.warnings


class NoLayoutQwen:
    available = True
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def locate_text_lines(self, path, **kwargs):
        return LocateResult([], {}, {})

    async def decode(self, path, category):
        self.calls.append((Path(path), category))
        return DecodeResult(self.text, {}, {})


class LocatedQwen(NoLayoutQwen):
    async def locate_text_lines(self, path, **kwargs):
        return LocateResult([{"location": [20, 20, 300, 20, 300, 50, 20, 50], "text": "Typed content in a normal paragraph."}], {}, {})


@pytest.mark.asyncio
async def test_pipeline_preserves_local_drawing_and_keeps_full_scan_outside_tex(tmp_path, monkeypatch):
    def content(draw):
        draw.rectangle((20, 20, 300, 50), fill="black")
        draw.ellipse((180, 200, 310, 330), outline="blue", width=4)

    path, _ = save_page(tmp_path, content)
    monkeypatch.setattr(pipeline_module, "compile_pdf", lambda tex_path: {"ok": True, "reason": None})
    settings = Settings(trustedrouter_api_key="test", enable_qwen_rescue=False, enable_mistral_layout_rescue=False, jev_enabled=False)
    result = await Hand2TeXPipeline(settings, qwen=LocatedQwen("")).run([path], tmp_path / "work", "Drawing")
    layout = json.loads((result["result_dir"] / "layout.json").read_text(encoding="utf-8"))
    figures = [unit for unit in layout["pages"][0]["units"] if unit["category"] == "figure"]
    assert len(figures) == 1
    assert figures[0]["bbox"]["y1"] > 50
    assert figures[0]["bbox"]["x2"] - figures[0]["bbox"]["x1"] < 200
    tex = (result["result_dir"] / "main.tex").read_text(encoding="utf-8")
    assert "Typed content in a normal paragraph." in tex
    assert tex.count("includegraphics") == 1
    assert "sources/" not in tex
    assert result["manifest"]["source_pages"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transcription", ["Normal prose from the fallback page.", ""])
async def test_no_layout_attempts_transcription_and_attaches_source_outside_document(tmp_path, monkeypatch, transcription):
    path, _ = save_page(tmp_path, lambda draw: draw.rectangle((20, 20, 300, 50), fill="black"))
    def compiled(tex_path):
        tex_path.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")
        return {"ok": True, "reason": None}

    monkeypatch.setattr(pipeline_module, "compile_pdf", compiled)
    client = NoLayoutQwen(transcription)
    settings = Settings(trustedrouter_api_key="test", enable_qwen_rescue=False, enable_mistral_layout_rescue=False, jev_enabled=False)
    result = await Hand2TeXPipeline(settings, qwen=client).run([path], tmp_path / "work", "Fallback")
    assert len(client.calls) == 1 and client.calls[0][1] == "text"
    manifest = result["manifest"]
    assert manifest["status"] == "warning"
    assert manifest["output_mode"] == "flow-document"
    assert manifest["source_pages"][0]["path"] == "sources/page_0000.png"
    with Image.open(result["result_dir"] / "sources/page_0000.png") as attached, Image.open(path) as original:
        assert attached.size == original.size
        assert attached.tobytes() == original.tobytes()
    layout = json.loads((result["result_dir"] / "layout.json").read_text(encoding="utf-8"))
    assert layout["pages"][0]["units"][0]["category"] == "text"
    tex = (result["result_dir"] / "main.tex").read_text(encoding="utf-8")
    assert "includegraphics" not in tex
    if transcription:
        assert transcription in tex
    else:
        assert any(warning["stage"] == "ocr" for warning in manifest["content_warnings"])
