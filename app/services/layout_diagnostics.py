from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any


DEBUG_VERSION = "2.15-debug"
_SECRET_FIELDS = {
    "authorization", "proxy_authorization", "api_key", "apikey", "x_api_key",
    "access_token", "password", "secret", "cookie", "set_cookie",
}


def redact_sensitive(value: Any, api_key: str = "", *, omit_images: bool = False) -> Any:
    """Keep diagnostic structure while excluding credentials and request image bytes."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower().replace("-", "_") in _SECRET_FIELDS
            else redact_sensitive(child, api_key, omit_images=omit_images)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(child, api_key, omit_images=omit_images) for child in value]
    if isinstance(value, str):
        if omit_images and value.startswith("data:") and ";base64," in value:
            return value.split(";base64,", 1)[0] + ";base64,[OMITTED]"
        return value.replace(api_key, "[REDACTED]") if api_key else value
    return value


class LayoutDiagnostics:
    """One collector per conversion; never store diagnostic state on the API client."""

    def __init__(self, result_dir: Path, api_key: str = ""):
        self.result_dir = result_dir
        self.api_key = api_key
        self.pages: list[LayoutTrace] = []
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save()

    def for_page(self, index: int, width: int, height: int) -> LayoutTrace:
        trace = LayoutTrace(self, index, width, height)
        self.pages.append(trace)
        self.save()
        return trace

    def save(self) -> None:
        for filename, attribute in (
            ("request_payload_sanitized.json", "requests"),
            ("qwen_layout_raw.json", "responses"),
            ("parsed_layout.json", "parsed"),
        ):
            data = {"version": DEBUG_VERSION, "pages": [getattr(page, attribute) for page in self.pages]}
            (self.result_dir / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
            )


class LayoutTrace:
    def __init__(self, owner: LayoutDiagnostics, index: int, width: int, height: int):
        self.owner = owner
        metadata = {"page": index, "width_px": width, "height_px": height}
        self.requests: dict[str, Any] = {**metadata, "attempts": []}
        self.responses: dict[str, Any] = {**metadata, "attempts": []}
        self.parsed: dict[str, Any] = {
            **metadata, "status": "pending", "words_info": [], "blocks": [],
            "usable_box_count": 0, "quality": 0.0, "fallback": None,
        }

    def start_attempt(self, payload: dict[str, Any], url: str) -> int:
        attempt = len(self.requests["attempts"]) + 1
        # Recursive sanitization also copies the payload before ocr_options is popped.
        self.requests["attempts"].append({
            "attempt": attempt,
            "url": redact_sensitive(url, self.owner.api_key),
            "payload": redact_sensitive(payload, self.owner.api_key, omit_images=True),
        })
        self.responses["attempts"].append({"attempt": attempt})
        self.owner.save()
        return attempt

    def record_response(self, attempt: int, status_code: int, response: Any, response_text: str | None = None) -> None:
        record = self.responses["attempts"][attempt - 1]
        record["status_code"] = status_code
        if response_text is None:
            record["response"] = redact_sensitive(response, self.owner.api_key)
        else:
            record["response_text"] = redact_sensitive(response_text, self.owner.api_key)
        self.owner.save()

    def record_error(self, attempt: int, error: Exception) -> None:
        self.responses["attempts"][attempt - 1]["error"] = redact_sensitive(
            f"{type(error).__name__}: {error}", self.owner.api_key,
        )
        self.owner.save()

    def record_parsed(self, words_info: list[dict[str, Any]], blocks: list[Any], quality: float, fallback: str | None = None) -> None:
        self.parsed.update(redact_sensitive({
            "status": "no_usable_boxes" if fallback else "ok",
            "words_info": words_info,
            "blocks": [asdict(block) for block in blocks],
            "usable_box_count": 0 if fallback else sum(block.source_label.startswith("qwen_layout_") for block in blocks),
            "quality": quality,
            "fallback": fallback,
        }, self.owner.api_key))
        self.owner.save()

    def fail(self, error: Exception) -> None:
        self.parsed.update({
            "status": "error",
            "error": redact_sensitive(f"{type(error).__name__}: {error}", self.owner.api_key),
        })
        self.owner.save()
