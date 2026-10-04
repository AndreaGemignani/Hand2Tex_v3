import json

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.services.layout_diagnostics import LayoutDiagnostics
from app.services.qwen_ocr import DECODE_PROMPTS, LAYOUT_PROMPT, QwenOCRClient, QwenOCRError


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (200, 300), "white").save(path)
    return path


def settings():
    return Settings(trustedrouter_api_key="test-native-tasks", trustedrouter_sort="price")


def response(payload):
    prompt = payload["messages"][0]["content"][0]["text"]
    if prompt == LAYOUT_PROMPT:
        content = json.dumps({"words_info": [{"location": [20, 30, 180, 30, 180, 60, 20, 60], "text": "Notes"}]})
    elif prompt == DECODE_PROMPTS["math"]:
        content = r"\frac{1}{\sqrt{3}}"
    elif prompt == DECODE_PROMPTS["table"]:
        content = r"\begin{tabular}{ll}a&b\\c&d\end{tabular}"
    else:
        assert prompt == DECODE_PROMPTS["text"]
        content = r"Attenzione: usa $C_L=0.58$, con $1/\sqrt{3}$ nella formula."
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 20, "completion_tokens": 5}}


async def invoke(client, page, category, diagnostics=None):
    if category == "layout":
        return await client.locate_text_lines(page, diagnostics=diagnostics)
    return await client.decode(page, category)


@pytest.mark.asyncio
@pytest.mark.parametrize("category,task", [
    ("text", "document_parsing"),
    ("math", "formula_recognition"),
    ("table", "table_parsing"),
])
async def test_native_task_matches_decoder_and_keeps_text_prompt(page, category, task):
    seen = []

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-native-tasks"
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=response(payload))

    client = QwenOCRClient(settings(), transport=httpx.MockTransport(handler))
    result = await client.decode(page, category)
    assert len(seen) == 1
    payload = seen[0]
    assert payload["ocr_options"] == {"task": task}
    assert payload["messages"][0]["content"][0] == {"type": "text", "text": DECODE_PROMPTS[category]}
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["model"] == settings().qwen_ocr_model
    assert payload["provider"] == {"sort": "price", "usage": "credits"}
    assert payload["temperature"] == 0 and payload["max_tokens"] == 4096
    assert result.text == response(payload)["choices"][0]["message"]["content"]
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("first_operation", ["text", "math", "table", "layout"])
async def test_unknown_native_field_retries_once_and_is_cached_for_layout_and_decoders(page, tmp_path, first_operation):
    seen = []

    async def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if len(seen) == 1:
            return httpx.Response(400, json={"error": {"message": "Unknown field: ocr_options"}})
        assert "ocr_options" not in payload
        return httpx.Response(200, json=response(payload))

    client = QwenOCRClient(settings(), transport=httpx.MockTransport(handler))
    await invoke(client, page, first_operation)
    assert len(seen) == 2
    assert "ocr_options" in seen[0]
    assert seen[1] == {key: value for key, value in seen[0].items() if key != "ocr_options"}
    # A later layout call starts with ocr_options in its own payload constructor;
    # the cached negotiation must still remove it without a second rejected call.
    diagnostics = LayoutDiagnostics(tmp_path / "diagnostics", settings().trustedrouter_api_key)
    for category in ["layout", "text", "math", "table", "layout"]:
        trace = diagnostics.for_page(len(diagnostics.pages), 200, 300) if category == "layout" else None
        result = await invoke(client, page, category, trace)
        assert result.usage == {"prompt_tokens": 20, "completion_tokens": 5}
    assert len(seen) == 7
    assert all("ocr_options" not in payload for payload in seen[1:])
    logged = json.loads((tmp_path / "diagnostics/request_payload_sanitized.json").read_text(encoding="utf-8"))
    assert len(logged["pages"]) == 2
    assert all(len(item["attempts"]) == 1 and "ocr_options" not in item["attempts"][0]["payload"] for item in logged["pages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["Invalid model name", "ocr_options has an invalid task value", "Insufficient credit balance"])
async def test_unrelated_bad_request_does_not_retry_or_disable_native_tasks(page, error):
    seen = []

    async def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if len(seen) == 1:
            return httpx.Response(400, json={"error": {"message": error}})
        return httpx.Response(200, json=response(payload))

    client = QwenOCRClient(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(QwenOCRError, match="400"):
        await client.decode(page, "text")
    assert len(seen) == 1
    await client.decode(page, "math")
    assert seen[1]["ocr_options"] == {"task": "formula_recognition"}
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_failed_fallback_propagates_without_a_third_request(page):
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(400, text="Unknown field ocr_options")
        return httpx.Response(400, text="Invalid image")

    client = QwenOCRClient(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(QwenOCRError, match="Invalid image"):
        await client.decode(page, "text")
    assert len(seen) == 2
    assert "ocr_options" in seen[0] and "ocr_options" not in seen[1]
