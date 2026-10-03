from __future__ import annotations

import base64
import json
import mimetypes
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.config import Settings
from app.services.layout_diagnostics import LayoutTrace


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

LAYOUT_PROMPT = """Locate all text lines and return the coordinates of the rotated rectangle ([cx, cy, width, height, angle])."""


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


def _find_position_items(value: Any) -> list[dict[str, Any]]:
    """Recover position-only output such as the documented `pos_list` fallback."""
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        pos_list = value.get("pos_list")
        if isinstance(pos_list, list):
            for item in pos_list:
                if isinstance(item, dict) and (isinstance(item.get("rotate_rect"), list) or isinstance(item.get("location"), list)):
                    found.append(item)
        # Common generic vision box names from OpenAI-compatible gateways.
        if any(isinstance(value.get(k), list) for k in ("rotate_rect", "location", "bbox", "bbox_2d", "box", "coordinates")):
            found.append(value)
        for child in value.values():
            found.extend(_find_position_items(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_position_items(child))
    # De-duplicate by serialized geometry.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in found:
        key = json.dumps({k: item.get(k) for k in ("rotate_rect", "location", "bbox", "bbox_2d", "box", "coordinates")}, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _parse_plain_position_text(text: str) -> list[dict[str, Any]]:
    """Read TrustedRouter's position-only CSV in the model's 0..1000 space.

    The observed Qwen response has one cx,cy,width,height,angle row per line,
    without a JSON wrapper or recognized text. Only accept a complete numeric
    coordinate listing, so prose and mathematical transcriptions are not boxes.
    Native JSON location/rotate_rect values keep their existing pixel semantics.
    """
    value = text.strip()
    value = re.sub(r"^```(?:csv|text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    items: list[dict[str, Any]] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or not all(re.fullmatch(number, field) for field in fields):
            return []
        rect = [float(field) for field in fields]
        cx, cy, width, height, angle = rect
        if (
            not all(math.isfinite(coordinate) for coordinate in rect)
            or not (0 <= cx <= 1000 and 0 <= cy <= 1000)
            or width <= 0 or height <= 0
            or not -90 <= angle <= 90
        ):
            return []
        items.append({"rotate_rect": rect, "coordinate_space": "normalized_1000"})
    return items


def _ordered_polygon(corners: list[tuple[float, float]]) -> list[float]:
    """Keep a clockwise quad, starting at its top-left corner."""
    first = min(range(4), key=lambda index: sum(corners[index]))
    corners = corners[first:] + corners[:first]
    area = sum(corners[i][0] * corners[(i + 1) % 4][1] - corners[(i + 1) % 4][0] * corners[i][1] for i in range(4))
    if area < 0:
        corners = [corners[0], *reversed(corners[1:])]
    return [coordinate for point in corners for coordinate in point]


def _rotated_rect_polygon(rect: list[Any]) -> list[float] | None:
    """Retain [cx,cy,w,h,angle] corners for line crops instead of just an envelope."""
    if len(rect) < 4:
        return None
    try:
        cx, cy, rw, rh = map(float, rect[:4])
        angle = float(rect[4]) if len(rect) >= 5 else 0.0
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (cx, cy, rw, rh, angle)) or rw <= 0 or rh <= 0:
        return None
    theta = math.radians(angle)
    c, sn = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in ((-rw/2, -rh/2), (rw/2, -rh/2), (rw/2, rh/2), (-rw/2, rh/2)):
        x = cx + dx*c - dy*sn
        y = cy + dx*sn + dy*c
        corners.append((x, y))
    return _ordered_polygon(corners)


def _rotated_rect_envelope(rect: list[Any]) -> tuple[float, float, float, float] | None:
    polygon = _rotated_rect_polygon(rect)
    if polygon is None:
        return None
    return min(polygon[::2]), min(polygon[1::2]), max(polygon[::2]), max(polygon[1::2])



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

    async def _post(self, payload: dict[str, Any], diagnostics: LayoutTrace | None = None) -> dict[str, Any]:
        timeout = httpx.Timeout(connect=20, read=self.settings.qwen_timeout_s, write=60, pool=20)
        attempt = diagnostics.start_attempt(payload, self.settings.trustedrouter_chat_url) if diagnostics else None
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
                response = await client.post(self.settings.trustedrouter_chat_url, headers=self._headers(), json=payload)
        except Exception as exc:
            if diagnostics and attempt is not None:
                diagnostics.record_error(attempt, exc)
            raise
        # Capture the entire provider response before status checks or parsing.
        if diagnostics and attempt is not None:
            try:
                body = response.json()
            except ValueError:
                diagnostics.record_response(attempt, response.status_code, None, response.text)
            else:
                diagnostics.record_response(attempt, response.status_code, body)
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

    async def locate_text_lines(self, image_path: Path, *, diagnostics: LayoutTrace | None = None) -> LocateResult:
        """Use the same Qwen-VL-OCR model via TrustedRouter to recover layout boxes.

        The gateway may return native JSON or a position-only CSV listing.
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
            # Forward the vendor-native task when the gateway supports it.
            # If TrustedRouter rejects the non-standard field, retry with the
            # documented fixed prompt, which returns a `pos_list` fallback.
            "ocr_options": {"task": "advanced_recognition"},
        }
        try:
            raw = await self._post(payload, diagnostics)
        except QwenOCRError as exc:
            if any(code in str(exc) for code in (" 400:", " 404:", " 422:")):
                payload.pop("ocr_options", None)
                raw = await self._post(payload, diagnostics)
            else:
                raise

        # First prefer provider-native advanced-recognition payloads. TrustedRouter
        # may preserve an `ocr_result.words_info` object instead of serializing it
        # into message text.
        items = _find_words_info(raw)
        text = _extract_text(raw)
        parsed: Any = None
        if text:
            try:
                parsed = json.loads(_clean_json_text(text))
            except Exception:
                parsed = None
        if not items and parsed is not None:
            items = _find_words_info(parsed)
        if not items:
            # OpenAI-compatible Qwen OCR can return the official position-only
            # `pos_list` in message text while the richer `ocr_result` is omitted.
            items = _find_position_items(parsed if parsed is not None else raw)
        if not items and parsed is None:
            items = _parse_plain_position_text(text)

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
            if not isinstance(bbox, list):
                for key in ("bbox_2d", "box", "coordinates"):
                    if isinstance(item.get(key), list):
                        bbox = item.get(key)
                        break
            rotate_rect = item.get("rotate_rect")
            coords = location if isinstance(location, list) else bbox if isinstance(bbox, list) else []
            polygon: list[float] = []

            try:
                if isinstance(coords, list) and len(coords) == 8:
                    xs = [float(coords[i]) for i in (0, 2, 4, 6)]
                    ys = [float(coords[i]) for i in (1, 3, 5, 7)]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    polygon = _ordered_polygon(list(zip(xs, ys)))
                    # `location` is already in image pixels in the official task.
                elif isinstance(coords, list) and len(coords) == 4:
                    x1, y1, x2, y2 = map(float, coords)
                    # Only our custom bbox convention is normalized 0..1000.
                    if bbox is not None and max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1000:
                        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
                        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
                elif isinstance(rotate_rect, list) and len(rotate_rect) >= 4:
                    polygon = _rotated_rect_polygon(rotate_rect) or []
                    if not polygon:
                        continue
                    if item.get("coordinate_space") == "normalized_1000":
                        # Rotate in the model's coordinate space before restoring
                        # the page's aspect ratio; scaling the rectangle first
                        # would change its angle and give incorrect crops.
                        polygon = [value * (width if index % 2 == 0 else height) / 1000.0 for index, value in enumerate(polygon)]
                    x1, x2 = min(polygon[::2]), max(polygon[::2])
                    y1, y2 = min(polygon[1::2]), max(polygon[1::2])
                else:
                    continue
            except (TypeError, ValueError):
                continue

            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, *polygon)):
                continue
            x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            polygon = [max(0.0, min(float(width if index % 2 == 0 else height), value)) for index, value in enumerate(polygon)]
            recognized = str(item.get("text", item.get("word", ""))).strip()
            words.append({
                "location": [x1, y1, x2, y1, x2, y2, x1, y2],
                "text": recognized,
                # Qwen advanced_recognition does not classify regions; downstream
                # heuristics classify math/text without another model call.
                "category": str(item.get("category", "")).strip().lower(),
                "polygon": polygon,
            })
        return LocateResult(words_info=words, usage=raw.get("usage") or {}, raw=raw)
