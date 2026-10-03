"""Best-effort model prewarm for Docker builds.

If Paddle's model CDN is temporarily unavailable the image still builds; the application
will retry model download at first request and can use Mistral layout rescue if configured.
"""
from __future__ import annotations

import os

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

try:
    from paddleocr import LayoutDetection
    name = os.getenv("PADDLE_LAYOUT_MODEL", "PP-DocLayout-M")
    LayoutDetection(model_name=name, device="cpu", engine=os.getenv("PADDLE_ENGINE", "paddle_static"))
    print(f"Prewarmed {name}")
except Exception as exc:
    print(f"Paddle prewarm skipped: {exc}")
