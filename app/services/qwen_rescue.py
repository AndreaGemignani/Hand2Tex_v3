from __future__ import annotations

from pathlib import Path

import httpx

from app.config import Settings
from app.services.qwen_ocr import DecodeResult, QwenOCRError, _data_url, _extract_text


PROMPTS = {
    "text": "Transcribe the handwritten or printed text exactly. Preserve line breaks. Do not summarize or add commentary. Output only the transcription.",
    "math": "Read all mathematical expressions in this crop. Output only valid LaTeX, without code fences or explanation. If several consecutive formulas are present, use an aligned environment and preserve their order.",
    "table": "Reconstruct this table faithfully. Output only a LaTeX tabular environment, without code fences or explanation. Preserve every visible cell and row.",
}


class QwenRescueClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    @property
    def available(self) -> bool:
        return bool(self.settings.trustedrouter_api_key and self.settings.enable_qwen_rescue)

    async def decode(self, image_path: Path, category: str) -> DecodeResult:
        if not self.available:
            raise QwenOCRError("Qwen rescue is not configured")
        payload = {
            "model": self.settings.qwen_rescue_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPTS.get(category, PROMPTS["text"])},
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                ],
            }],
            "temperature": 0,
            "max_tokens": 4096,
            "provider": {"sort": self.settings.trustedrouter_sort, "usage": "credits"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.trustedrouter_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=20, read=self.settings.qwen_timeout_s, write=60, pool=20)
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            response = await client.post(self.settings.trustedrouter_chat_url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise QwenOCRError(f"TrustedRouter Qwen rescue {response.status_code}: {response.text[:800]}")
        raw = response.json()
        return DecodeResult(text=_extract_text(raw), usage=raw.get("usage") or {}, raw=raw)
