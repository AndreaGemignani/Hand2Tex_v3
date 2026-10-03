import json

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.services.qwen_ocr import QwenOCRClient
from app.services.qwen_rescue import QwenRescueClient


def settings(monkeypatch):
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "test")
    return Settings()


@pytest.mark.asyncio
async def test_qwen_formula_payload(monkeypatch, tmp_path):
    img = tmp_path / "formula.png"
    Image.new("RGB", (64, 32), "white").save(img)
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": r"\frac{a}{b}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        })

    client = QwenOCRClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    result = await client.decode(img, "math")
    assert result.text == r"\frac{a}{b}"
    assert seen["model"] == "qwen/qwen-vl-ocr"
    assert seen["provider"]["sort"] == "price"
    assert seen["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_qwen_rescue_payload(monkeypatch, tmp_path):
    img = tmp_path / "text.png"
    Image.new("RGB", (64, 32), "white").save(img)
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })

    client = QwenRescueClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    result = await client.decode(img, "text")
    assert result.text == "hello"
    assert seen["model"] == "qwen/qwen3.8-max"


@pytest.mark.asyncio
async def test_qwen_layout_json(monkeypatch, tmp_path):
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)

    async def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "words_info": [
                    {"bbox": [50, 100, 500, 300], "category": "text", "text": "hello"},
                    {"bbox": [50, 400, 500, 600], "category": "math", "text": "F=ma"},
                ]
            })}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        })

    client = QwenOCRClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    result = await client.locate_text_lines(img)
    assert result.words_info[0]["text"] == "hello"
    # normalized 0..1000 -> pixel coordinates
    assert result.words_info[0]["location"][0] == 10
    assert result.words_info[0]["location"][1] == 10
    assert result.words_info[1]["category"] == "math"

@pytest.mark.asyncio
async def test_qwen_layout_nested_ocr_result_pixels(monkeypatch, tmp_path):
    img = tmp_path / "page_nested.png"
    Image.new("RGB", (1663, 2420), "white").save(img)

    async def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": [
                {"ocr_result": {"words_info": [
                    {"location": [100, 200, 900, 200, 900, 260, 100, 260], "text": "Simple handwritten text"}
                ]}}
            ]}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        })

    client = QwenOCRClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    result = await client.locate_text_lines(img)
    assert result.words_info[0]["text"] == "Simple handwritten text"
    # Official advanced-recognition `location` values are pixel coordinates.
    assert result.words_info[0]["location"][0] == 100
    assert result.words_info[0]["location"][1] == 200
