from app.config import Settings
from app.models import BBox, ProcessingUnit
from app.services.costs import estimate_cost


def test_cost_counts_primary_and_rescue(monkeypatch):
    monkeypatch.setenv("QWEN_OCR_INPUT_PER_M_USD", "1")
    monkeypatch.setenv("QWEN_OCR_OUTPUT_PER_M_USD", "1")
    monkeypatch.setenv("QWEN_RESCUE_INPUT_PER_M_USD", "2")
    monkeypatch.setenv("QWEN_RESCUE_OUTPUT_PER_M_USD", "2")
    unit = ProcessingUnit("x", 0, "text", BBox(0, 0, 1, 1), ["b"], 1.0)
    unit.usage_events = [
        {"provider": "qwen_ocr", "input_tokens": 100, "output_tokens": 50},
        {"provider": "qwen_rescue", "prompt_tokens": 10, "completion_tokens": 5},
    ]
    out = estimate_cost(Settings(), [unit])
    assert out["qwen_ocr_input_tokens"] == 100
    assert out["qwen_rescue_input_tokens"] == 10
    assert out["qwen_ocr_usd"] == 0.00015
    assert out["qwen_rescue_usd"] == 0.00003
