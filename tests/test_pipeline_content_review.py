import asyncio
import json
import httpx
import pytest
from PIL import Image, ImageDraw

import app.pipeline as pipeline_module
from app.config import Settings
from app.models import BBox, DocumentLayout, LayoutBlock, PageLayout, ProcessingUnit
from app.pipeline import Hand2TeXPipeline
from app.services.content_review import ContentReviewClient, ReviewBatchResult, ReviewRecord
from app.services.costs import estimate_cost
from app.services.qwen_ocr import DecodeResult


class NoRescue:
    available = False


class NoJev:
    async def should_rescue(self, state):
        return None


class Detector:
    def detect(self, path, page_index):
        return [
            LayoutBlock("prose", page_index, "text", "text", .99, BBox(20, 20, 280, 70)),
            LayoutBlock("formula", page_index, "math", "formula", .99, BBox(20, 180, 280, 260)),
        ], .99


class DraftOCR:
    available = True

    async def decode(self, path, category):
        return DecodeResult(
            "The rate is positive." if category == "text" else "x=12",
            {"input_tokens": 20, "output_tokens": 5}, {},
        )


def settings(**kwargs):
    return Settings(trustedrouter_api_key="test-review-key", enable_content_review=True,
                    enable_qwen_rescue=False, enable_mistral_layout_rescue=False,
                    jev_enabled=False, **kwargs)


def pipeline(config, reviewer):
    return Hand2TeXPipeline(config, Detector(), DraftOCR(), NoRescue(), NoRescue(), NoJev(), reviewer)


@pytest.fixture
def compiler_stub(monkeypatch):
    def compile_stub(path):
        path.with_suffix(".pdf").write_bytes(b"%PDF-test\n")
        return {"ok": True, "reason": None}
    monkeypatch.setattr(pipeline_module, "compile_pdf", compile_stub)


@pytest.mark.asyncio
async def test_every_structurally_valid_region_reviewed_before_render(tmp_path, compiler_stub):
    image = tmp_path / "source.png"
    canvas = Image.new("RGB", (320, 400), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "The rate is positive.", fill="black")
    draw.text((25, 190), "x = 1", fill="black")
    draw.line((55, 215, 85, 215), fill="black", width=2)
    draw.text((65, 225), "2", fill="black")
    canvas.save(image)
    calls = []

    def review_response(request):
        payload = json.loads(request.content)
        calls.append(payload)
        meta = json.loads(payload["messages"][0]["content"][0]["text"].split("TARGET METADATA:\n")[1])
        entries = []
        for item in meta:
            is_math = item["category"] == "math"
            entries.append({"id": item["id"], "text": r"x=\frac{1}{2}" if is_math else item["ocr_draft"],
                            "category": item["category"], "status": "corrected" if is_math else "verified",
                            "issues": ["Fraction bar visible"] if is_math else []})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"units": entries})}}],
                                        "usage": {"prompt_tokens": 200, "completion_tokens": 50}})

    config = settings()
    reviewer = ContentReviewClient(config, httpx.MockTransport(review_response))
    result = await pipeline(config, reviewer).run([image], tmp_path / "work", "Rates", include_debug=True)
    assert len(calls) == 1
    manifest = result["manifest"]
    assert manifest["status"] == "ok"
    assert manifest["content_review"]["verified"] == 1
    assert manifest["content_review"]["corrected"] == 1
    assert manifest["cost_estimate"]["content_review_input_tokens"] == 200
    assert manifest["cost_estimate"]["content_review_output_tokens"] == 50
    decoded = json.loads((result["result_dir"] / "decoded.json").read_text(encoding="utf-8"))
    math = next(unit for unit in decoded if unit["category"] == "math")
    assert math["raw_decoded"] == "x=12"
    assert math["decoded"] == r"x=\frac{1}{2}"
    assert all(unit["validation_kind"] == "structure" for unit in decoded)
    tex = (result["result_dir"] / "main.tex").read_text(encoding="utf-8")
    assert r"x=\frac{1}{2}" in tex and "x=12" not in tex
    report = json.loads((result["result_dir"] / "quality_review.json").read_text(encoding="utf-8"))
    assert all((result["result_dir"] / unit["source_crop"]).is_file() for unit in report["units"])
    assert "test-review-key" not in (result["result_dir"] / "quality_review_raw.json").read_text(encoding="utf-8")


def document(tmp_path, drafts=("x=1", "y=2")):
    units = []
    for i, draft in enumerate(drafts):
        crop = tmp_path / f"crop{i}.png"
        Image.new("RGB", (120, 60), "white").save(crop)
        unit = ProcessingUnit(f"m{i}", 0, "math", BBox(0, i*50, 100, i*50+40), [f"b{i}"], .99,
                              decoded=draft, validation_score=1.0, review_crop_path=str(crop))
        units.append(unit)
    page = PageLayout(0, 200, 400, "", units=units)
    return DocumentLayout([page]), units


@pytest.mark.asyncio
async def test_invalid_billed_review_preserves_drafts_warns_and_keeps_usage(tmp_path):
    layout, units = document(tmp_path)
    config = settings()
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "invalid JSON"}}],
                                        "usage": {"input_tokens": 140, "output_tokens": 30}})
    client = ContentReviewClient(config, httpx.MockTransport(handler))
    events = []
    summary, warnings = await pipeline(config, client)._review_document(layout, tmp_path, events, asyncio.Semaphore(1), True)
    assert summary["uncertain"] == 2 and len(warnings) == 2
    assert [unit.decoded for unit in units] == ["x=1", "y=2"]
    assert all(unit.quality_review["mark_pdf"] for unit in units)
    assert estimate_cost(config, units, extra_ocr_events=events)["content_review_input_tokens"] == 140


@pytest.mark.asyncio
async def test_network_failure_redacted_and_preserves_all_content(tmp_path):
    layout, units = document(tmp_path)
    config = settings()
    class Failing:
        available = True
        async def review(self, batch):
            raise RuntimeError("request failed test-review-key")
    summary, warnings = await pipeline(config, Failing())._review_document(layout, tmp_path, [], asyncio.Semaphore(1), True)
    assert summary["uncertain"] == 2 and warnings
    assert all(unit.decoded == unit.raw_decoded for unit in units)
    assert "test-review-key" not in (tmp_path / "quality_review_raw.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_unscheduled_long_region_and_full_page_get_explicit_warning(tmp_path):
    layout, units = document(tmp_path, ("x"*500, "y=2"))
    units[1].member_ids = ["p0000_fullpage"]
    class NeverCalled:
        available = True
        async def review(self, batch):
            pytest.fail("Unbounded regions must not create review API calls")
    summary, warnings = await pipeline(settings(content_review_max_input_chars=200), NeverCalled())._review_document(
        layout, tmp_path, [], asyncio.Semaphore(1), False)
    assert summary["not_reviewed"] == 2 and len(warnings) == 2
    assert all(unit.quality_review["mark_pdf"] for unit in units)


@pytest.mark.asyncio
async def test_disabled_review_makes_no_calls_or_source_warnings(tmp_path):
    layout, units = document(tmp_path)
    class NeverCalled:
        available = True
        async def review(self, batch):
            pytest.fail("Disabled review called")
    config = Settings(enable_content_review=False)
    summary, warnings = await pipeline(config, NeverCalled())._review_document(layout, tmp_path, [], asyncio.Semaphore(1), False)
    assert not warnings and not summary["enabled"] and summary["not_reviewed"] == 2
    assert all(not unit.quality_review.get("mark_pdf") for unit in units)


@pytest.mark.asyncio
async def test_connected_formula_merge_keeps_originals_without_duplicate_content(tmp_path):
    layout, units = document(tmp_path, ("1", "2"))
    units[1].bbox = BBox(0, 42, 100, 65)
    class Merger:
        available = True
        async def review(self, batch):
            return ReviewBatchResult(records={"m0": ReviewRecord("m0", r"\frac{1}{2}", "math", "corrected",
                                     ["One fraction split in two"], ["m0", "m1"])},
                                     usage={"input_tokens": 20}, model="test")
    summary, warnings = await pipeline(settings(), Merger())._review_document(layout, tmp_path, [], asyncio.Semaphore(1), False)
    assert summary["merged_regions"] == 1 and not warnings
    assert units[0].decoded == r"\frac{1}{2}" and units[1].decoded == ""
    assert units[1].raw_decoded == "2" and units[1].quality_review["merged_into"] == "m0"


@pytest.mark.asyncio
async def test_failed_review_reaches_download_with_source_and_visible_notes(tmp_path, compiler_stub):
    source = tmp_path / "source.png"
    Image.new("RGB", (320, 400), "white").save(source)
    class Rejected:
        available = True
        async def review(self, batch):
            return ReviewBatchResult(usage={"input_tokens": 75, "output_tokens": 4},
                                     errors=["Invalid JSON"], model="test")
    result = await pipeline(settings(), Rejected()).run([source], tmp_path / "work", "Draft")
    manifest = result["manifest"]
    assert manifest["status"] == "warning"
    assert manifest["content_review"]["uncertain"] == 2
    assert manifest["cost_estimate"]["content_review_input_tokens"] == 75
    assert len(manifest["source_pages"]) == 1
    assert (result["result_dir"] / manifest["source_pages"][0]["path"]).is_file()
    tex = (result["result_dir"] / "main.tex").read_text(encoding="utf-8")
    assert tex.count("Trascrizione da verificare sull’originale.") == 2
    assert "The rate is positive." in tex and "x=12" in tex


@pytest.mark.asyncio
async def test_rejected_syntax_rescue_is_still_counted_as_a_paid_candidate(tmp_path):
    class Rescue:
        available = True
        async def decode(self, path, category):
            return DecodeResult("?", {"input_tokens": 13, "output_tokens": 2}, {})
    class RequestRescue:
        async def should_rescue(self, state):
            return True
    crop = tmp_path / "crop.png"
    Image.new("RGB", (100, 50), "white").save(crop)
    unit = ProcessingUnit("t", 0, "text", BBox(0, 0, 100, 50), ["b"], .99, crop_path=str(crop))
    config = Settings(enable_content_review=False)
    worker = Hand2TeXPipeline(config, Detector(), DraftOCR(), Rescue(), NoRescue(), RequestRescue())
    await worker._decode_unit(unit, asyncio.Semaphore(1))
    assert unit.decoded == "The rate is positive." and not unit.rescued
    cost = estimate_cost(config, [unit])
    assert cost["qwen_rescue_input_tokens"] == 13 and cost["qwen_rescue_output_tokens"] == 2

