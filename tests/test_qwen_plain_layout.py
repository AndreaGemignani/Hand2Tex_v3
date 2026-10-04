import base64
import csv
import io
import json
import math
from pathlib import Path

import httpx
import pytest
from PIL import Image

import app.pipeline as pipeline_module
from app.config import Settings
from app.pipeline import Hand2TeXPipeline
from app.services.layout_diagnostics import DEBUG_VERSION, LayoutDiagnostics
from app.services.qwen_ocr import DECODE_PROMPTS, QwenOCRClient, _rotated_rect_envelope


CSV_FIXTURE = Path(__file__).parent / "fixtures" / "qwen_layout_coordinates.csv"
LAYOUT_REQUEST_TEXT = (
    "Locate all text lines and return the coordinates of the rotated rectangle "
    "([cx, cy, width, height, angle])."
)
PAGE_SIZE = (1663, 2420)


@pytest.fixture
def settings():
    return Settings(
        trustedrouter_api_key="sk-tr-test-plain-layout",
        layout_backend="qwen",
        enable_qwen_rescue=False,
        enable_content_review=False,
        enable_mistral_layout_rescue=False,
        jev_enabled=False,
    )


@pytest.fixture
def coordinate_csv():
    return CSV_FIXTURE.read_text(encoding="utf-8")


def blank_page(tmp_path, size=PAGE_SIZE):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", size, "white").save(image_path)
    return image_path


def response_for(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def expected_location(rect, page_size):
    """Use rotated half extents in normalized space, then scale each page axis."""
    cx, cy, rect_width, rect_height, angle = rect
    radians = math.radians(angle)
    half_x = (abs(math.cos(radians)) * rect_width + abs(math.sin(radians)) * rect_height) / 2
    half_y = (abs(math.sin(radians)) * rect_width + abs(math.cos(radians)) * rect_height) / 2
    page_width, page_height = page_size
    x1 = min(page_width, max(0, (cx - half_x) * page_width / 1000))
    x2 = min(page_width, max(0, (cx + half_x) * page_width / 1000))
    y1 = min(page_height, max(0, (cy - half_y) * page_height / 1000))
    y2 = min(page_height, max(0, (cy + half_y) * page_height / 1000))
    return [x1, y1, x2, y1, x2, y2, x1, y2]


@pytest.mark.asyncio
async def test_plain_csv_recovers_all_boxes_after_unchanged_router_fallback(
    settings, coordinate_csv, tmp_path,
):
    sent = []

    async def handler(request):
        assert request.headers["Authorization"] == f"Bearer {settings.trustedrouter_api_key}"
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(400, json={"error": {"message": "unsupported field ocr_options"}})
        return httpx.Response(200, json=response_for(coordinate_csv))

    result_dir = tmp_path / "result"
    diagnostics = LayoutDiagnostics(result_dir, settings.trustedrouter_api_key)
    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    located = await client.locate_text_lines(
        blank_page(tmp_path), diagnostics=diagnostics.for_page(0, *PAGE_SIZE),
    )
    assert len(sent) == 2
    assert sent[0]["model"] == "qwen/qwen-vl-ocr-2025-11-20"
    assert sent[0]["provider"] == {"sort": "price", "usage": "credits"}
    assert sent[0]["temperature"] == 0
    assert sent[0]["max_tokens"] == 6144
    assert sent[0]["ocr_options"] == {"task": "advanced_recognition"}
    assert sent[0]["messages"][0]["content"][0] == {"type": "text", "text": LAYOUT_REQUEST_TEXT}
    assert sent[1] == {key: value for key, value in sent[0].items() if key != "ocr_options"}
    assert len(located.words_info) == 44
    rows = [list(map(float, row)) for row in csv.reader(coordinate_csv.splitlines())]
    assert len(rows) == 44
    for word, row in zip(located.words_info, rows):
        assert word["location"] == pytest.approx(expected_location(row, PAGE_SIZE))
        assert word["text"] == ""
    # A normalized bottom-page center must become a pixel position near y=2333.
    last = located.words_info[-1]["location"]
    assert last[1] > .9 * PAGE_SIZE[1]
    assert [last[0], last[1], last[4], last[5]] == pytest.approx([
        61.5 * PAGE_SIZE[0] / 1000,
        948.5 * PAGE_SIZE[1] / 1000,
        714.5 * PAGE_SIZE[0] / 1000,
        979.5 * PAGE_SIZE[1] / 1000,
    ])
    requests = json.loads((result_dir / "request_payload_sanitized.json").read_text(encoding="utf-8"))
    assert requests["version"] == DEBUG_VERSION
    assert "ocr_options" in requests["pages"][0]["attempts"][0]["payload"]
    assert "ocr_options" not in requests["pages"][0]["attempts"][1]["payload"]
    raw = json.loads((result_dir / "qwen_layout_raw.json").read_text(encoding="utf-8"))
    assert raw["version"] == DEBUG_VERSION
    assert raw["pages"][0]["attempts"][1]["response"]["choices"][0]["message"]["content"] == coordinate_csv


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [PAGE_SIZE, (640, 480)])
async def test_ninety_degree_rotation_precedes_independent_axis_scaling(settings, tmp_path, page_size):
    async def handler(request):
        return httpx.Response(200, json=response_for("480,494,31,901,90"))

    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    located = await client.locate_text_lines(blank_page(tmp_path, page_size))
    assert len(located.words_info) == 1
    width, height = page_size
    # A 90-degree rotation swaps the normalized 31 and 901 dimensions first.
    expected = [
        29.5 * width / 1000, 478.5 * height / 1000,
        930.5 * width / 1000, 478.5 * height / 1000,
        930.5 * width / 1000, 509.5 * height / 1000,
        29.5 * width / 1000, 509.5 * height / 1000,
    ]
    assert located.words_info[0]["location"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("fence_language", [None, "", "csv"])
async def test_optional_code_fences_keep_csv_geometry(settings, tmp_path, fence_language):
    text = "480,494,31,901,90"
    if fence_language is not None:
        text = f"```{fence_language}\n{text}\n```"

    async def handler(request):
        return httpx.Response(200, json=response_for(text))

    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    located = await client.locate_text_lines(blank_page(tmp_path))
    assert len(located.words_info) == 1
    assert located.words_info[0]["location"] == pytest.approx(
        expected_location([480, 494, 31, 901, 90], PAGE_SIZE),
    )


@pytest.mark.asyncio
async def test_csv_envelopes_are_clipped_to_the_page(settings, tmp_path):
    csv_text = "20,30,80,100,0\n980,990,100,60,0"

    async def handler(request):
        return httpx.Response(200, json=response_for(csv_text))

    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    located = await client.locate_text_lines(blank_page(tmp_path))
    assert len(located.words_info) == 2
    assert located.words_info[0]["location"] == pytest.approx(
        expected_location([20, 30, 80, 100, 0], PAGE_SIZE),
    )
    assert located.words_info[1]["location"] == pytest.approx(
        expected_location([980, 990, 100, 60, 0], PAGE_SIZE),
    )
    assert located.words_info[0]["location"][:2] == [0, 0]
    assert located.words_info[1]["location"][4:6] == list(PAGE_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_csv", [
    "480,494,31,901",
    "480,494,31,901,90,extra",
    "480,494,wide,901,90",
    "Coordinates follow:\n480,494,31,901,90",
    "480,494,31,901,90\ninvalid,row",
    "nan,494,31,901,90",
    "480,inf,31,901,90",
    "480,494,nan,901,90",
    "480,494,31,inf,90",
    "480,494,31,901,-inf",
    "480,494,0,901,90",
    "480,494,-31,901,90",
    "480,494,31,0,90",
    "480,494,31,-901,90",
])
async def test_invalid_csv_is_rejected_as_a_whole(settings, tmp_path, invalid_csv):
    async def handler(request):
        return httpx.Response(200, json=response_for(invalid_csv))

    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    located = await client.locate_text_lines(blank_page(tmp_path))
    assert located.words_info == []


@pytest.mark.parametrize("rect", [
    [float("nan"), 100, 20, 80, 90],
    [100, float("inf"), 20, 80, 90],
    [100, 100, float("nan"), 80, 90],
    [100, 100, 20, float("inf"), 90],
    [100, 100, 20, 80, float("inf")],
    [100, 100, 0, 80, 90],
    [100, 100, -20, 80, 90],
    [100, 100, 20, 0, 90],
    [100, 100, 20, -80, 90],
])
def test_rotated_envelope_rejects_nonfinite_or_nonpositive_dimensions(rect):
    assert _rotated_rect_envelope(rect) is None


@pytest.mark.asyncio
async def test_plain_coordinate_layout_runs_crop_decoding_without_scan_background(
    settings, coordinate_csv, monkeypatch, tmp_path,
):
    layout_attempts = []
    crop_sizes = []
    decoded_text = "Decoded sample text from this crop."

    async def handler(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        assert body["model"] == settings.qwen_ocr_model
        if content[0]["text"] == LAYOUT_REQUEST_TEXT:
            layout_attempts.append(body)
            if "ocr_options" in body:
                return httpx.Response(400, json={"error": {"message": "unsupported field ocr_options"}})
            return httpx.Response(200, json=response_for(coordinate_csv))
        assert content[0]["text"] == DECODE_PROMPTS["text"]
        assert body["max_tokens"] == 4096
        assert "ocr_options" not in body
        image_url = content[1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as crop:
            crop_sizes.append(crop.size)
        return httpx.Response(200, json=response_for(decoded_text))

    def compiled_pdf(tex_path):
        tex_path.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")
        (tex_path.parent / "compile.log").write_text("Test compiler", encoding="utf-8")
        return {"ok": True, "reason": None, "pdf": str(tex_path.with_suffix(".pdf"))}

    monkeypatch.setattr(pipeline_module, "compile_pdf", compiled_pdf)
    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    pipeline = Hand2TeXPipeline(settings, qwen=client)
    result = await pipeline.run(
        [blank_page(tmp_path)], tmp_path / "work", "Coordinate layout", include_debug=True,
    )
    result_dir = result["result_dir"]
    assert len(layout_attempts) == 2
    assert crop_sizes
    assert all(0 < width < PAGE_SIZE[0] and 0 < height < PAGE_SIZE[1] for width, height in crop_sizes)
    parsed = json.loads((result_dir / "parsed_layout.json").read_text(encoding="utf-8"))
    assert parsed["version"] == DEBUG_VERSION
    page = parsed["pages"][0]
    assert page["status"] == "ok"
    assert page["usable_box_count"] == 44
    assert page["fallback"] is None
    assert len(page["words_info"]) == 44
    assert all(word["text"] == "" for word in page["words_info"])
    text_blocks = [block for block in page["blocks"] if block["category"] == "text"]
    assert len(text_blocks) == 44
    assert all(block["hint_text"] == "" for block in text_blocks)
    assert not any(block["source_label"] == "residual_background" for block in page["blocks"])
    decoded = json.loads((result_dir / "decoded.json").read_text(encoding="utf-8"))
    text_units = [unit for unit in decoded if unit["category"] == "text"]
    assert len(text_units) == len(crop_sizes)
    assert all(unit["decoded"] == decoded_text and unit["decoder"] == "qwen-ocr:text" for unit in text_units)
    assert decoded_text in (result_dir / "main.tex").read_text(encoding="utf-8")
    assert (result_dir / "main.pdf").exists()
    assert result["manifest"]["status"] == "ok"
    assert result["manifest"]["detector_errors"] == []
    layout = json.loads((result_dir / "layout.json").read_text(encoding="utf-8"))
    assert not any(unit["category"] == "figure" for unit in layout["pages"][0]["units"])
    assert result["manifest"]["output_mode"] == "flow-document"
