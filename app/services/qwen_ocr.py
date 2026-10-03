from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.config import Settings


class QwenOCRError(RuntimeError):
    pass


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _extract_text(raw: dict[str, Any]) -> str:
    try:
        content = raw["choices"][0]["message"]["content"]
    except Exception as exc:
        raise QwenOCRError(f"Unexpected TrustedRouter response shape: {raw}") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def _clean_json_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last > first:
        value = value[first:last + 1]
    return value


DECODE_PROMPTS = {
    "text": (
        "Transcribe all visible handwritten or printed text exactly. Preserve line breaks. "
        "Do not summarize, correct, translate, or add commentary. Output only the transcription."
    ),
    "math": (
        "Convert every visible mathematical expression to valid LaTeX. Preserve the exact mathematics. "
        "If several consecutive equations are present, use an aligned environment. "
        "Matrices must become proper LaTeX matrix environments. Output LaTeX only, no code fences or explanation."
    ),
    "table": (
        "Reconstruct this table faithfully as LaTeX. Preserve all visible rows and cells. "
        "Output only a LaTeX tabular environment, without code fences or explanation."
    ),
}

LAYOUT_PROMPT = """Locate all text lines and return the coordinates of the rotated rectangle ([cx, cy, width, height, angle]).
Return the recognized text for every located line. Do not omit mathematical expressions.
If JSON output is supported, use a words_info array where each item contains location and text.
No explanation is needed."""


def _find_words_info(value: Any) -> list[dict[str, Any]]:
    """Find a words_info list in provider-native or OpenAI-compatible responses.

    Qwen OCR gateways may expose the advanced-recognition result either as
    message text JSON or as a nested `ocr_result` object. Keep the parser
    provider-agnostic so the routing layer does not depend on one gateway.
    """
    if isinstance(value, dict):
        direct = value.get("words_info")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, dict)]
        for child in value.values():
            found = _find_words_info(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_words_info(child)
            if found:
                return found
    return []



@dataclass
class DecodeResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class LocateResult:
    words_info: list[dict[str, Any]]
    usage: dict[str, Any]
    raw: dict[str, Any]


class QwenOCRClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    @property
    def available(self) -> bool:
        return bool(self.settings.trustedrouter_api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.trustedrouter_api_key}",
            "Content-Type": "application/json",
        }

    def _provider(self) -> dict[str, Any] | None:
        sort = self.settings.trustedrouter_sort
        if sort in {"price", "latency", "throughput"}:
            return {"sort": sort, "usage": "credits"}
        return {"usage": "credits"}

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(connect=20, read=self.settings.qwen_timeout_s, write=60, pool=20)
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            response = await client.post(self.settings.trustedrouter_chat_url, headers=self._headers(), json=payload)
        if response.status_code >= 400:
            raise QwenOCRError(f"TrustedRouter Qwen OCR {response.status_code}: {response.text[:800]}")
        try:
            return response.json()
        except Exception as exc:
            raise QwenOCRError(f"TrustedRouter returned non-JSON response: {response.text[:800]}") from exc

    async def decode(self, image_path: Path, category: str) -> DecodeResult:
        if not self.available:
            raise QwenOCRError("TRUSTEDROUTER_API_KEY is not configured")
        prompt = DECODE_PROMPTS.get(category)
        if not prompt:
            raise QwenOCRError(f"No Qwen OCR task for category {category}")
        payload: dict[str, Any] = {
            "model": self.settings.qwen_ocr_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 4096,
            "provider": self._provider(),
        }
        raw = await self._post(payload)
        return DecodeResult(text=_extract_text(raw), usage=raw.get("usage") or {}, raw=raw)

    async def locate_text_lines(self, image_path: Path) -> LocateResult:
        """Use the same Qwen-VL-OCR model via TrustedRouter to recover layout boxes.

        The gateway is OpenAI-compatible, so we request compact normalized JSON rather
        than relying on Alibaba-specific `advanced_recognition` request parameters.
        Drawings are intentionally omitted and later preserved as the residual layer.
        """
        if not self.available:
            raise QwenOCRError("TRUSTEDROUTER_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.settings.qwen_ocr_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": LAYOUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 6144,
            "provider": self._provider(),
        }
        raw = await self._post(payload)

        # First prefer provider-native advanced-recognition payloads. TrustedRouter
        # may preserve an `ocr_result.words_info` object instead of serializing it
        # into message text.
        items = _find_words_info(raw)
        text = _extract_text(raw)
        if not items and text:
            try:
                parsed = json.loads(_clean_json_text(text))
                items = _find_words_info(parsed)
            except Exception:
                # Some gateways may return prose around the JSON. If it cannot be
                # parsed, leave the list empty and let the pipeline preserve/rescue.
                items = []

        with Image.open(image_path) as im:
            width, height = im.size

        words: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            # Official Qwen advanced_recognition returns `location` as four pixel
            # corners. Custom JSON fallbacks may instead return a normalized bbox.
            location = item.get("location")
            bbox = item.get("bbox")
            rotate_rect = item.get("rotate_rect")
            coords = location if isinstance(location, list) else bbox if isinstance(bbox, list) else []

            try:
                if isinstance(coords, list) and len(coords) == 8:
                    xs = [float(coords[i]) for i in (0, 2, 4, 6)]
                    ys = [float(coords[i]) for i in (1, 3, 5, 7)]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    # `location` is already in image pixels in the official task.
                elif isinstance(coords, list) and len(coords) == 4:
                    x1, y1, x2, y2 = map(float, coords)
                    # Only our custom bbox convention is normalized 0..1000.
                    if bbox is not None and max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1000:
                        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
                        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
                elif isinstance(rotate_rect, list) and len(rotate_rect) >= 4:
                    cx, cy, rw, rh = map(float, rotate_rect[:4])
                    # Axis-aligned envelope is enough for our deterministic renderer.
                    x1, y1, x2, y2 = cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2
                else:
                    continue
            except (TypeError, ValueError):
                continue

            x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            recognized = str(item.get("text", "")).strip()
            if not recognized:
                continue
            words.append({
                "location": [x1, y1, x2, y1, x2, y2, x1, y2],
                "text": recognized,
                # Qwen advanced_recognition does not classify regions; downstream
                # heuristics classify math/text without another model call.
                "category": str(item.get("category", "")).strip().lower(),
            })
        return LocateResult(words_info=words, usage=raw.get("usage") or {}, raw=raw)
