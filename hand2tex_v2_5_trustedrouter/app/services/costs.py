from __future__ import annotations

from app.config import Settings


def _tokens(usage: dict) -> tuple[int, int]:
    inp = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return int(inp), int(out)


def estimate_cost(settings: Settings, units: list, mistral_pages: int = 0, extra_ocr_events: list[dict] | None = None) -> dict[str, float]:
    ocr_in = ocr_out = rescue_in = rescue_out = 0
    events: list[dict] = []
    for unit in units:
        events.extend(getattr(unit, "usage_events", []) or [])
    events.extend(extra_ocr_events or [])
    for event in events:
        inp, out = _tokens(event)
        provider = event.get("provider", "")
        if provider == "qwen_rescue":
            rescue_in += inp
            rescue_out += out
        elif provider == "qwen_ocr":
            ocr_in += inp
            ocr_out += out

    qwen_ocr = (
        ocr_in * settings.qwen_ocr_input_per_million_usd
        + ocr_out * settings.qwen_ocr_output_per_million_usd
    ) / 1_000_000
    qwen_rescue = (
        rescue_in * settings.qwen_rescue_input_per_million_usd
        + rescue_out * settings.qwen_rescue_output_per_million_usd
    ) / 1_000_000
    mistral = mistral_pages * settings.mistral_ocr_per_page_usd
    return {
        "qwen_ocr_input_tokens": ocr_in,
        "qwen_ocr_output_tokens": ocr_out,
        "qwen_rescue_input_tokens": rescue_in,
        "qwen_rescue_output_tokens": rescue_out,
        "qwen_ocr_usd": round(qwen_ocr, 8),
        "qwen_rescue_usd": round(qwen_rescue, 8),
        "mistral_layout_rescue_usd": round(mistral, 8),
        "estimated_total_usd": round(qwen_ocr + qwen_rescue + mistral, 8),
    }
