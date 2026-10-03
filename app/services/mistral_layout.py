from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.models import BBox, LayoutBlock
from app.services.layout_detector import category_for_label


class MistralLayoutError(RuntimeError):
    pass


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class MistralLayoutRescue:
    URL = "https://api.mistral.ai/v1/ocr"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.settings.mistral_api_key and self.settings.enable_mistral_layout_rescue)

    async def detect(self, page_path: Path, page_index: int, width: int, height: int) -> tuple[list[LayoutBlock], float, dict[str, Any]]:
        if not self.available:
            raise MistralLayoutError("Mistral layout rescue is not configured")
        payload = {
            "model": self.settings.mistral_model,
            "document": {"type": "image_url", "image_url": _data_url(page_path)},
            "include_blocks": True,
            "confidence_scores_granularity": "block",
            "include_image_base64": False,
            "table_format": "markdown",
        }
        headers = {"Authorization": f"Bearer {self.settings.mistral_api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(connect=20, read=120, write=60, pool=20)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise MistralLayoutError(f"Mistral OCR {response.status_code}: {response.text[:600]}")
        raw = response.json()
        pages = raw.get("pages") or []
        if not pages:
            raise MistralLayoutError("Mistral OCR returned no page")
        page = pages[0]
        dims = page.get("dimensions") or {}
        source_w = float(dims.get("width") or width)
        source_h = float(dims.get("height") or height)
        sx, sy = width / source_w, height / source_h
        blocks: list[LayoutBlock] = []
        scores: list[float] = []
        for idx, item in enumerate(page.get("blocks") or []):
            label = str(item.get("type", "unknown"))
            conf = item.get("confidence_scores") or {}
            score = conf.get("block_type_confidence_score")
            if not isinstance(score, (int, float)):
                score = conf.get("average_content_confidence_score")
            score = float(score) if isinstance(score, (int, float)) else 0.75
            bbox = BBox(
                float(item.get("top_left_x", 0)) * sx,
                float(item.get("top_left_y", 0)) * sy,
                float(item.get("bottom_right_x", 0)) * sx,
                float(item.get("bottom_right_y", 0)) * sy,
            ).clamp(width, height)
            blocks.append(LayoutBlock(
                id=f"p{page_index:04d}_m{idx:04d}", page=page_index,
                category=category_for_label(label),  # type: ignore[arg-type]
                source_label=label, score=score, bbox=bbox, provider="mistral",
            ))
            scores.append(score)
        quality = sum(scores) / len(scores) if scores else 0.0
        return blocks, quality, raw
