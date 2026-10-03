from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings


class JevRouter:
    """Optional metadata-only decision layer. It never receives image pixels."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    @property
    def available(self) -> bool:
        return bool(self.settings.jev_enabled and self.settings.typesafe_api_key)

    async def should_rescue(self, state: dict[str, Any]) -> bool | None:
        if not self.available:
            return None
        payload = {
            "model": self.settings.jev_model,
            "state": state,
            "questions": {
                "rescue": {
                    "type": "noul",
                    "instructions": "Should this OCR region be escalated to the stronger rescue decoder because the current decoding is materially unreliable?",
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.settings.typesafe_api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20, transport=self.transport) as client:
            r = await client.post(self.settings.jev_endpoint, headers=headers, json=payload)
        if r.status_code >= 400:
            return None
        raw = r.json()
        answer = ((raw.get("answers") or {}).get("rescue") or {}).get("noul")
        if isinstance(answer, (int, float)):
            return float(answer) >= 0.5
        return None
