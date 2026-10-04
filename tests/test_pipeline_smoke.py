from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.config import Settings
from app.models import BBox, LayoutBlock
from app.pipeline import Hand2TeXPipeline
from app.services.qwen_ocr import DecodeResult


class FakeDetector:
    def detect(self, path, page_index):
        return [
            LayoutBlock("t", page_index, "text", "text", .98, BBox(50,50,450,180)),
            LayoutBlock("m", page_index, "math", "formula", .97, BBox(50,230,450,350)),
            LayoutBlock("f", page_index, "figure", "figure", .99, BBox(520,50,900,420)),
        ], .98


class FakeQwen:
    available = True
    async def decode(self, path, category):
        value = {"text": "Force equals mass times acceleration", "math": r"F=ma", "table": ""}[category]
        return DecodeResult(value, {"input_tokens": 10, "output_tokens": 5}, {})


class FakeRescue:
    available = False


class FakeMistral:
    available = False


class FakeJev:
    async def should_rescue(self, state):
        return None


@pytest.mark.asyncio
async def test_pipeline_harmony(monkeypatch, tmp_path):
    # Pipeline test needs pdflatex; skip only the external compiler if absent.
    import shutil
    if not shutil.which("pdflatex"):
        pytest.skip("pdflatex unavailable in test host")
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "fake")
    img = tmp_path / "page.png"
    canvas = Image.new("RGB", (1000,1400), "white")
    d = ImageDraw.Draw(canvas); d.rectangle((520,50,900,420), outline="black", width=4)
    canvas.save(img)
    pipeline = Hand2TeXPipeline(Settings(enable_content_review=False), FakeDetector(), FakeQwen(), FakeRescue(), FakeMistral(), FakeJev())
    result = await pipeline.run([img], tmp_path / "work", "Integration")
    result_dir = result["result_dir"]
    assert (result_dir / "main.pdf").exists()
    assert (result_dir / "main.tex").exists()
    assert (result_dir / "layout.json").exists()
    assert (result_dir / "manifest.json").exists()

from app.services.qwen_ocr import LocateResult


class FakeQwenLayout(FakeQwen):
    async def locate_text_lines(self, path):
        return LocateResult([
            {"location": [50, 50, 450, 50, 450, 110, 50, 110], "text": "Newton second law"},
            {"location": [50, 150, 300, 150, 300, 210, 50, 210], "text": "F = ma"},
        ], {"input_tokens": 30, "output_tokens": 8}, {})


@pytest.mark.asyncio
async def test_qwen_layout_render_free_path(monkeypatch, tmp_path):
    import shutil
    if not shutil.which("pdflatex"):
        pytest.skip("pdflatex unavailable in test host")
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "fake")
    monkeypatch.setenv("LAYOUT_BACKEND", "qwen")
    img = tmp_path / "page_qwen.png"
    canvas = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle((600, 300, 900, 600), outline="black", width=4)
    canvas.save(img)
    pipeline = Hand2TeXPipeline(Settings(enable_content_review=False), None, FakeQwenLayout(), FakeRescue(), FakeMistral(), FakeJev())
    result = await pipeline.run([img], tmp_path / "work_qwen", "Qwen Layout")
    result_dir = result["result_dir"]
    assert (result_dir / "main.pdf").exists()
    manifest = result["manifest"]
    assert manifest["detectors"][0]["provider"].startswith("qwen-vl-ocr-layout")
    decoded = (result_dir / "decoded.json").read_text(encoding="utf-8")
    assert "Newton second law" in decoded
