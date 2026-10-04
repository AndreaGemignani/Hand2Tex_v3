from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Hand2TeX"))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 30))
    max_files: int = field(default_factory=lambda: _int("MAX_FILES", 20))
    max_pages: int = field(default_factory=lambda: _int("MAX_PAGES", 30))
    render_dpi: int = field(default_factory=lambda: _int("RENDER_DPI", 220))

    layout_backend: str = field(default_factory=lambda: os.getenv("LAYOUT_BACKEND", "qwen").strip().lower())
    paddle_model: str = field(default_factory=lambda: os.getenv("PADDLE_LAYOUT_MODEL", "PP-DocLayout-M"))
    paddle_device: str = field(default_factory=lambda: os.getenv("PADDLE_DEVICE", "cpu"))
    paddle_engine: str = field(default_factory=lambda: os.getenv("PADDLE_ENGINE", "paddle_static"))
    paddle_box_threshold: float = field(default_factory=lambda: _float("PADDLE_BOX_THRESHOLD", 0.35))
    layout_rescue_threshold: float = field(default_factory=lambda: _float("LAYOUT_RESCUE_THRESHOLD", 0.58))
    layout_min_blocks: int = field(default_factory=lambda: _int("LAYOUT_MIN_BLOCKS", 1))

    # TrustedRouter: same Qwen models, simpler OpenAI-compatible gateway.
    trustedrouter_api_key: str = field(default_factory=lambda: os.getenv("TRUSTEDROUTER_API_KEY", "").strip())
    trustedrouter_base_url: str = field(
        default_factory=lambda: os.getenv("TRUSTEDROUTER_BASE_URL", "https://api.trustedrouter.com/v1").rstrip("/")
    )
    qwen_ocr_model: str = field(default_factory=lambda: os.getenv("QWEN_OCR_MODEL", "qwen/qwen-vl-ocr-2025-11-20"))
    qwen_rescue_model: str = field(default_factory=lambda: os.getenv("QWEN_RESCUE_MODEL", "qwen/qwen3.8-max"))
    qwen_concurrency: int = field(default_factory=lambda: _int("QWEN_CONCURRENCY", 4))
    qwen_timeout_s: int = field(default_factory=lambda: _int("QWEN_TIMEOUT_S", 120))
    enable_qwen_rescue: bool = field(default_factory=lambda: _bool("ENABLE_QWEN_RESCUE", True))
    qwen_rescue_threshold: float = field(default_factory=lambda: _float("QWEN_RESCUE_THRESHOLD", 0.55))
    trustedrouter_sort: str = field(default_factory=lambda: os.getenv("TRUSTEDROUTER_SORT", "price").strip().lower())

    # Syntax scores cannot detect a fluent but incorrect transcription. Review
    # source-grounded content in compact batches, separately from cheap OCR.
    enable_content_review: bool = field(default_factory=lambda: _bool("ENABLE_CONTENT_REVIEW", True))
    content_review_model: str = field(default_factory=lambda: os.getenv("CONTENT_REVIEW_MODEL", os.getenv("QWEN_RESCUE_MODEL", "qwen/qwen3.8-max")).strip())
    content_review_batch_size: int = field(default_factory=lambda: _int("CONTENT_REVIEW_BATCH_SIZE", 6))
    content_review_max_pixels: int = field(default_factory=lambda: _int("CONTENT_REVIEW_MAX_PIXELS", 3_000_000))
    content_review_max_input_chars: int = field(default_factory=lambda: _int("CONTENT_REVIEW_MAX_INPUT_CHARS", 12_000))
    content_review_max_output_tokens: int = field(default_factory=lambda: _int("CONTENT_REVIEW_MAX_OUTPUT_TOKENS", 4096))

    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    mistral_model: str = field(default_factory=lambda: os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"))
    enable_mistral_layout_rescue: bool = field(default_factory=lambda: _bool("ENABLE_MISTRAL_LAYOUT_RESCUE", True))

    typesafe_api_key: str = field(default_factory=lambda: os.getenv("TYPESAFE_API_KEY", ""))
    jev_enabled: bool = field(default_factory=lambda: _bool("JEV_ENABLED", False))
    jev_model: str = field(default_factory=lambda: os.getenv("JEV_MODEL", "jev-latest"))
    jev_endpoint: str = field(default_factory=lambda: os.getenv("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone"))

    include_debug_by_default: bool = field(default_factory=lambda: _bool("INCLUDE_DEBUG", True))

    # Current TrustedRouter prepaid list prices; override via env if they change.
    qwen_ocr_input_per_million_usd: float = field(default_factory=lambda: _float("QWEN_OCR_INPUT_PER_M_USD", 0.0735))
    qwen_ocr_output_per_million_usd: float = field(default_factory=lambda: _float("QWEN_OCR_OUTPUT_PER_M_USD", 0.168))
    qwen_rescue_input_per_million_usd: float = field(default_factory=lambda: _float("QWEN_RESCUE_INPUT_PER_M_USD", 1.74075))
    qwen_rescue_output_per_million_usd: float = field(default_factory=lambda: _float("QWEN_RESCUE_OUTPUT_PER_M_USD", 5.223305))
    content_review_input_per_million_usd: float = field(default_factory=lambda: _float("CONTENT_REVIEW_INPUT_PER_M_USD", _float("QWEN_RESCUE_INPUT_PER_M_USD", 1.74075)))
    content_review_output_per_million_usd: float = field(default_factory=lambda: _float("CONTENT_REVIEW_OUTPUT_PER_M_USD", _float("QWEN_RESCUE_OUTPUT_PER_M_USD", 5.223305)))
    mistral_ocr_per_page_usd: float = field(default_factory=lambda: _float("MISTRAL_OCR_PER_PAGE_USD", 0.004))

    @property
    def trustedrouter_chat_url(self) -> str:
        return self.trustedrouter_base_url + "/chat/completions"
