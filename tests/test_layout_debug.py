import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

import app.pipeline as pipeline_module
from app.config import Settings
from app.pipeline import Hand2TeXPipeline, PipelineError
from app.services.layout_diagnostics import DEBUG_VERSION, LayoutDiagnostics
from app.services.qwen_ocr import QwenOCRClient


API_KEY = "sk-tr-test-debug-secret"
DEBUG_FILES = (
    "qwen_layout_raw.json",
    "request_payload_sanitized.json",
    "parsed_layout.json",
)


@pytest.fixture
def settings():
    return Settings(
        trustedrouter_api_key=API_KEY,
        layout_backend="qwen",
        enable_qwen_rescue=False,
        enable_content_review=False,
        enable_mistral_layout_rescue=False,
        jev_enabled=False,
    )


@pytest.fixture
def compiled_pdf(monkeypatch):
    def compile_stub(tex_path):
        tex_path.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")
        (tex_path.parent / "compile.log").write_text("Test compiler", encoding="utf-8")
        return {"ok": True, "reason": None, "pdf": str(tex_path.with_suffix(".pdf"))}

    monkeypatch.setattr(pipeline_module, "compile_pdf", compile_stub)


def image_file(tmp_path, name="page.png", size=(400, 600)):
    image_path = tmp_path / name
    Image.new("RGB", size, "white").save(image_path)
    return image_path


def layout_response(text="Handwritten notes about this page.", *, words=None, **extra):
    if words is None:
        words = [{
            "location": [20, 40, 300, 40, 300, 90, 20, 90],
            "text": text,
            "category": "text",
        }]
    return {
        "id": "layout-response",
        "choices": [{"message": {"content": json.dumps({"words_info": words})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        **extra,
    }


def read_diagnostics(result_dir):
    documents = {}
    for name in DEBUG_FILES:
        documents[name] = json.loads((result_dir / name).read_text(encoding="utf-8"))
        assert documents[name]["version"] == DEBUG_VERSION
    return documents


def real_pipeline(settings, handler):
    qwen = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    return Hand2TeXPipeline(settings, qwen=qwen)


@pytest.mark.asyncio
async def test_debug_captures_complete_responses_and_layout_for_each_page(
    settings, compiled_pdf, tmp_path,
):
    pages = [image_file(tmp_path, f"page-{index}.png") for index in range(2)]
    responses = [
        layout_response(f"Handwritten notes from page {index}.", provider_metadata={"page": index})
        for index in range(2)
    ]
    sent_payloads = []

    async def handler(request):
        sent_payloads.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(sent_payloads) - 1])

    result = await real_pipeline(settings, handler).run(
        pages, tmp_path / "work", "Debug pages", include_debug=True,
    )
    documents = read_diagnostics(result["result_dir"])
    actual_layout = json.loads((result["result_dir"] / "layout.json").read_text(encoding="utf-8"))

    for document in documents.values():
        assert [page["page"] for page in document["pages"]] == [0, 1]
        assert all((page["width_px"], page["height_px"]) == (400, 600) for page in document["pages"])
    assert len(sent_payloads) == 2
    for index in range(2):
        raw = documents[DEBUG_FILES[0]]["pages"][index]["attempts"][0]
        assert raw["attempt"] == 1
        assert raw["status_code"] == 200
        assert raw["response"] == responses[index]
        request = documents[DEBUG_FILES[1]]["pages"][index]["attempts"][0]
        assert request["url"] == settings.trustedrouter_chat_url
        assert request["payload"]["model"] == settings.qwen_ocr_model
        assert request["payload"]["provider"] == sent_payloads[index]["provider"]
        assert request["payload"]["ocr_options"] == {"task": "advanced_recognition"}
        parsed = documents[DEBUG_FILES[2]]["pages"][index]
        assert parsed["status"] == "ok"
        assert parsed["usable_box_count"] == 1
        assert parsed["quality"] == pytest.approx(.95)
        assert parsed["fallback"] is None
        assert parsed["words_info"][0]["text"] == f"Handwritten notes from page {index}."
        assert parsed["blocks"] == actual_layout["pages"][index]["blocks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_debug", [True, False])
async def test_empty_layout_uses_text_fallback_with_zero_quality(
    settings, compiled_pdf, tmp_path, include_debug,
):
    response = layout_response(words=[])

    async def handler(request):
        body = json.loads(request.content)
        if body["messages"][0]["content"][0]["text"].startswith("Locate all text"):
            return httpx.Response(200, json=response)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Fallback transcription of the original page."}}]})

    result = await real_pipeline(settings, handler).run(
        [image_file(tmp_path)], tmp_path / "work", "Empty layout", include_debug=include_debug,
    )
    assert result["manifest"]["detectors"][0]["quality"] == 0
    assert any(error["page"] == 0 for error in result["manifest"]["detector_errors"])
    layout = json.loads((result["result_dir"] / "layout.json").read_text(encoding="utf-8"))
    assert layout["pages"][0]["blocks"][0]["source_label"] == "whole_page_ocr_fallback"
    assert layout["pages"][0]["units"][0]["decoded"] == "Fallback transcription of the original page."
    assert result["manifest"]["source_pages"][0]["path"] == "sources/page_0000.png"
    if include_debug:
        parsed = read_diagnostics(result["result_dir"])[DEBUG_FILES[2]]["pages"][0]
        assert parsed["status"] == "no_usable_boxes"
        assert parsed["words_info"] == []
        assert parsed["usable_box_count"] == 0
        assert parsed["quality"] == 0
        assert parsed["fallback"] == "whole_page_ocr_fallback"
    else:
        assert all(not (result["result_dir"] / name).exists() for name in DEBUG_FILES)


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_status", [400, 404, 422])
async def test_fallback_retains_both_requests_and_responses(
    settings, compiled_pdf, tmp_path, rejected_status,
):
    sent = []
    rejection = {"error": {"message": "unsupported field ocr_options"}}
    accepted = layout_response()

    async def handler(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(rejected_status, json=rejection)
        return httpx.Response(200, json=accepted)

    result = await real_pipeline(settings, handler).run(
        [image_file(tmp_path)], tmp_path / "work", "Fallback", include_debug=True,
    )
    documents = read_diagnostics(result["result_dir"])
    attempts = documents[DEBUG_FILES[1]]["pages"][0]["attempts"]
    assert [attempt["attempt"] for attempt in attempts] == [1, 2]
    assert attempts[0]["payload"]["ocr_options"] == {"task": "advanced_recognition"}
    assert "ocr_options" not in attempts[1]["payload"]
    assert len(sent) == 2
    assert "ocr_options" in sent[0] and "ocr_options" not in sent[1]
    responses = documents[DEBUG_FILES[0]]["pages"][0]["attempts"]
    assert [response["status_code"] for response in responses] == [rejected_status, 200]
    assert responses[0]["response"] == rejection
    assert responses[1]["response"] == accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "non_json", "parser", "network"])
async def test_diagnostics_survive_layout_errors(settings, tmp_path, failure):
    malformed_shape = {"unexpected_response": {"metadata": "complete response retained"}}
    http_error = {"error": {"message": "upstream unavailable"}}

    async def handler(request):
        if failure == "http":
            return httpx.Response(503, json=http_error)
        if failure == "non_json":
            return httpx.Response(200, text="upstream sent an HTML error page")
        if failure == "parser":
            return httpx.Response(200, json=malformed_shape)
        raise httpx.ConnectError("network unavailable", request=request)

    work_dir = tmp_path / "work"
    with pytest.raises(PipelineError):
        await real_pipeline(settings, handler).run(
            [image_file(tmp_path)], work_dir, "Failure", include_debug=True,
        )
    documents = read_diagnostics(work_dir / "result")
    assert documents[DEBUG_FILES[2]]["pages"][0]["status"] == "error"
    assert len(documents[DEBUG_FILES[1]]["pages"][0]["attempts"]) == 1
    attempts = documents[DEBUG_FILES[0]]["pages"][0]["attempts"]
    if failure == "http":
        assert attempts[0]["status_code"] == 503
        assert attempts[0]["response"] == http_error
    elif failure == "non_json":
        assert attempts[0]["status_code"] == 200
        assert attempts[0]["response_text"] == "upstream sent an HTML error page"
    elif failure == "parser":
        assert attempts[0]["response"] == malformed_shape


@pytest.mark.asyncio
async def test_debug_redacts_key_echoes_headers_and_image_data(
    settings, compiled_pdf, tmp_path,
):
    response = layout_response(
        extra_metadata={"api_key": API_KEY, "nested": [f"echoed token: {API_KEY}"]},
    )
    sent = []

    async def handler(request):
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    result = await real_pipeline(settings, handler).run(
        [image_file(tmp_path)], tmp_path / "work", "Sanitization", include_debug=True,
    )
    documents = read_diagnostics(result["result_dir"])
    for name in DEBUG_FILES:
        assert API_KEY not in (result["result_dir"] / name).read_text(encoding="utf-8")
    request = documents[DEBUG_FILES[1]]["pages"][0]["attempts"][0]
    assert "headers" not in request
    assert "Authorization" not in json.dumps(request)
    original_url = sent[0]["messages"][0]["content"][1]["image_url"]["url"]
    sanitized_url = request["payload"]["messages"][0]["content"][1]["image_url"]["url"]
    assert "[OMITTED]" in sanitized_url
    assert original_url not in json.dumps(documents)
    assert sent[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_debug_does_not_change_request_or_parser_output(settings, tmp_path):
    image_path = image_file(tmp_path)
    response = layout_response()
    sent = []

    async def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    client = QwenOCRClient(settings, transport=httpx.MockTransport(handler))
    baseline = await client.locate_text_lines(image_path)
    diagnostics = LayoutDiagnostics(tmp_path / "result", API_KEY)
    traced = await client.locate_text_lines(
        image_path, diagnostics=diagnostics.for_page(0, 400, 600),
    )
    assert traced == baseline
    assert sent[0] == sent[1]


@pytest.mark.asyncio
async def test_concurrent_conversions_keep_diagnostics_separate(
    settings, compiled_pdf, tmp_path,
):
    async def handler(request):
        body = json.loads(request.content)
        data_url = body["messages"][0]["content"][1]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
            width = image.width
        await asyncio.sleep(0)
        return httpx.Response(200, json=layout_response(
            f"Handwritten notes from width {width}.", response_page_width=width,
        ))

    pipeline = real_pipeline(settings, handler)
    widths = [400, 500]
    results = await asyncio.gather(*(
        pipeline.run(
            [image_file(tmp_path, f"page-{width}.png", (width, 600))],
            tmp_path / f"work-{width}", "Isolation", include_debug=True,
        )
        for width in widths
    ))
    for width, result in zip(widths, results):
        documents = read_diagnostics(result["result_dir"])
        assert all(len(document["pages"]) == 1 for document in documents.values())
        raw_page = documents[DEBUG_FILES[0]]["pages"][0]
        assert raw_page["width_px"] == width
        assert len(raw_page["attempts"]) == 1
        assert raw_page["attempts"][0]["response"]["response_page_width"] == width
        assert documents[DEBUG_FILES[2]]["pages"][0]["words_info"][0]["text"] == (
            f"Handwritten notes from width {width}."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("compiler_raises", [False, True])
async def test_layout_diagnostics_remain_when_compilation_fails(settings, monkeypatch, tmp_path, compiler_raises):
    def failed_compilation(tex_path):
        (tex_path.parent / "compile.log").write_text("TeX compiler failed", encoding="utf-8")
        if compiler_raises:
            raise TimeoutError("Compiler timed out")
        return {"ok": False, "reason": "pdflatex failed", "pdf": None}

    monkeypatch.setattr(pipeline_module, "compile_pdf", failed_compilation)

    async def handler(request):
        return httpx.Response(200, json=layout_response())

    work_dir = tmp_path / "work"
    error_type = TimeoutError if compiler_raises else PipelineError
    message = "Compiler timed out" if compiler_raises else "LaTeX compilation failed"
    with pytest.raises(error_type, match=message):
        await real_pipeline(settings, handler).run(
            [image_file(tmp_path)], work_dir, "Compilation failure", include_debug=True,
        )
    documents = read_diagnostics(work_dir / "result")
    assert documents[DEBUG_FILES[2]]["pages"][0]["status"] == "ok"
    assert (work_dir / "result" / "compile.log").read_text(encoding="utf-8") == "TeX compiler failed"
    decoded = json.loads((work_dir / "result" / "decoded.json").read_text(encoding="utf-8"))
    assert any(unit["decoded"] == "Handwritten notes about this page." for unit in decoded)
    assert (work_dir / "result" / "layout.json").exists()
    manifest = json.loads((work_dir / "result" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cost_estimate"]["qwen_ocr_input_tokens"] == 100
    assert manifest["status"] == "error"
    reason = "TimeoutError: Compiler timed out" if compiler_raises else "pdflatex failed"
    assert manifest["compilation"] == {"ok": False, "reason": reason}
