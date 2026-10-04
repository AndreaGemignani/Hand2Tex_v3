import asyncio
import json

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.models import BBox, DocumentLayout, PageLayout, ProcessingUnit
from app.pipeline import Hand2TeXPipeline
from app.services.qwen_ocr import DECODE_PROMPTS, QwenOCRClient
from app.services.renderer import build_tex
from app.services.validators import validate_math, validate_text


async def decode_position_only_crop(tmp_path, text_result, math_result=None):
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (160, 80), "white").save(crop_path)
    calls = []
    settings = Settings(
        trustedrouter_api_key="sk-tr-test-math-auto",
        enable_qwen_rescue=False,
        jev_enabled=False,
    )

    async def handler(request):
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"][0]["text"]
        if prompt == DECODE_PROMPTS["text"]:
            calls.append("text")
            decoded = text_result
        else:
            assert prompt == DECODE_PROMPTS["math"]
            calls.append("math")
            assert math_result is not None, "Prose should not trigger a mathematical OCR call"
            decoded = math_result
        return httpx.Response(200, json={
            "choices": [{"message": {"content": decoded}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    qwen = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    pipeline = Hand2TeXPipeline(settings, qwen=qwen)
    unit = ProcessingUnit(
        "crop", 0, "text", BBox(20, 40, 380, 180), ["position-only"], .95,
        crop_path=str(crop_path), hint_text="",
    )
    await pipeline._decode_unit(unit, asyncio.Semaphore(1))
    return unit, calls


def test_valid_math_score_ties_perfect_text_score_exactly():
    assert validate_text("F=ma") == 1.0
    assert validate_math(r"\frac{F}{m}=a") == 1.0
    assert validate_math(r"\frac{F}{m}=a?") == .9


@pytest.mark.asyncio
@pytest.mark.parametrize("text_result,math_result", [
    ("F=ma", r"\frac{F}{m}=a"),
    (r"\begin{bmatrix}1&2\\3&4\end{bmatrix}", r"\begin{bmatrix}1&2\\3&4\end{bmatrix}"),
    (
        "\\begin{aligned}\nF&=ma\\\\\na&=\\frac{F}{m}\n\\end{aligned}",
        r"\begin{aligned}F&=ma\\a&=\frac{F}{m}\end{aligned}",
    ),
])
async def test_position_only_math_is_promoted_on_valid_equal_scores(tmp_path, text_result, math_result):
    unit, calls = await decode_position_only_crop(tmp_path, text_result, math_result)
    assert calls == ["text", "math"]
    assert unit.category == "math"
    assert unit.decoder == "qwen-ocr:math-auto"
    assert unit.decoded == math_result
    assert unit.validation_score == 1.0
    assert len(unit.usage_events) == 2
    assert unit.bbox == BBox(20, 40, 380, 180)
    page = PageLayout(0, 400, 600, "unused.png", units=[unit])
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Math promotion")
    tex = tex_path.read_text(encoding="utf-8")
    assert math_result in tex
    assert "\\[\n" + math_result + "\n\\]" in tex
    assert r"\resizebox" not in tex


@pytest.mark.asyncio
async def test_lower_quality_math_candidate_keeps_the_text_result(tmp_path):
    unit, calls = await decode_position_only_crop(tmp_path, "F=ma", r"\frac{F}{m}=a?")
    assert calls == ["text", "math"]
    assert unit.category == "text"
    assert unit.decoder == "qwen-ocr:text"
    assert unit.decoded == "F=ma"
    assert unit.validation_score == 1.0
    assert len(unit.usage_events) == 2


@pytest.mark.asyncio
async def test_mixed_prose_and_formulas_remain_text_without_math_ocr(tmp_path):
    prose = (
        "La spiegazione descrive il significato fisico della forza e della massa.\n"
        "Le formule F=ma e a=F/m accompagnano il ragionamento e chiariscono la relazione.\n"
        "Queste parole devono rimanere nel documento insieme alle formule originali."
    )
    unit, calls = await decode_position_only_crop(tmp_path, prose)
    assert calls == ["text"]
    assert unit.category == "text"
    assert unit.decoder == "qwen-ocr:text"
    assert unit.decoded == prose
    assert len(unit.usage_events) == 1
    page = PageLayout(0, 400, 600, "unused.png", units=[unit])
    tex_path = build_tex(DocumentLayout([page]), tmp_path / "result", "Preserved explanation")
    tex = tex_path.read_text(encoding="utf-8")
    assert "La spiegazione descrive il significato fisico" in tex
    assert "Queste parole devono rimanere nel documento" in tex
    assert "F=ma" in tex and "a=F/m" in tex
    assert r"\displaystyle" not in tex


@pytest.mark.asyncio
@pytest.mark.parametrize("prose", [
    r"Attenzione: $C_L=0.58$ e $\sigma=1/\sqrt{3}$.",
    r"Usa $C_L=0.58$; $\tau=\sigma/\sqrt{3}$.",
    r"Attenzione $\frac{1}{\sqrt{3}}$: $C_L=0.58$.",
])
async def test_short_instruction_with_formulas_is_never_replaced_by_math_only_ocr(tmp_path, prose):
    unit, calls = await decode_position_only_crop(tmp_path, prose)
    assert calls == ["text"]
    assert unit.category == "text"
    assert unit.decoded == prose
    assert unit.decoder == "qwen-ocr:text"
    assert len(unit.usage_events) == 1
