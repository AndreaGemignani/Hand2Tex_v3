from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import BBox, LayoutBlock


class LayoutDetectorError(RuntimeError):
    pass


TEXT_LABELS = {
    "doc_title", "document_title", "paragraph_title", "text", "page_number", "abstract",
    "references", "footnotes", "header", "footer", "algorithm", "figure_caption",
    "table_caption", "figure_title", "chart_title", "sidebar_text", "list", "reference", "caption",
    "table_of_contents", "section_title", "section_header", "vertical_text", "reference_content",
}
MATH_LABELS = {"formula", "equation", "formula_number", "inline_formula", "display_formula"}
TABLE_LABELS = {"table"}
FIGURE_LABELS = {
    "image", "figure", "chart", "seal", "header_image", "footer_image", "picture", "diagram",
}


def category_for_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MATH_LABELS:
        return "math"
    if key in TABLE_LABELS:
        return "table"
    if key in FIGURE_LABELS:
        return "figure"
    if key in TEXT_LABELS:
        return "text"
    return "unknown"


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    for attr in ("json", "to_dict", "dict"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            try:
                value = value() if callable(value) else value
            except TypeError:
                continue
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
            if isinstance(value, dict):
                return value
    try:
        value = dict(result)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    raise LayoutDetectorError(f"Unsupported Paddle result object: {type(result)!r}")


class PaddleLayoutDetector:
    """Thin compatibility wrapper around PaddleOCR LayoutDetection.

    Keep construction deliberately minimal: PaddleOCR 3.x has changed optional
    constructor arguments between releases, while model_name + device are stable.
    Thresholding is applied at predict() time as documented by PaddleOCR.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self.active_model = settings.paddle_model

    def _make_model(self, model_name: str):
        try:
            from paddleocr import LayoutDetection
        except Exception as exc:
            raise LayoutDetectorError(
                f"PaddleOCR import failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Follow the documented Python API first. Only add engine when explicitly
        # configured, and retry without it for cross-version compatibility.
        kwargs = {"model_name": model_name, "device": self.settings.paddle_device}
        engine = (self.settings.paddle_engine or "").strip()
        if engine and engine.lower() not in {"auto", "default", "none"}:
            kwargs["engine"] = engine
        try:
            return LayoutDetection(**kwargs)
        except TypeError:
            kwargs.pop("engine", None)
            try:
                return LayoutDetection(**kwargs)
            except Exception as exc:
                raise LayoutDetectorError(
                    f"Could not initialize Paddle model {model_name}: {type(exc).__name__}: {exc}"
                ) from exc
        except Exception as exc:
            raise LayoutDetectorError(
                f"Could not initialize Paddle model {model_name}: {type(exc).__name__}: {exc}"
            ) from exc

    def _load(self):
        if self._model is not None:
            return self._model
        primary = self.settings.paddle_model
        try:
            self._model = self._make_model(primary)
            self.active_model = primary
            return self._model
        except LayoutDetectorError as primary_exc:
            # Low-memory emergency fallback for small CPU hosts such as Render Free.
            # It preserves the same 23-class taxonomy, only with a smaller model.
            if primary == "PP-DocLayout-S":
                raise
            try:
                self._model = self._make_model("PP-DocLayout-S")
                self.active_model = "PP-DocLayout-S"
                return self._model
            except LayoutDetectorError as small_exc:
                raise LayoutDetectorError(
                    f"Primary detector failed ({primary_exc}); PP-DocLayout-S fallback failed ({small_exc})"
                ) from small_exc

    def _predict(self, model, page_path: Path):
        # PaddleOCR versions differ slightly in accepted predict kwargs.
        attempts = [
            {"batch_size": 1, "threshold": self.settings.paddle_box_threshold, "layout_nms": True},
            {"batch_size": 1, "threshold": self.settings.paddle_box_threshold},
            {"batch_size": 1},
        ]
        last: Exception | None = None
        for kwargs in attempts:
            try:
                return model.predict(str(page_path), **kwargs)
            except TypeError as exc:
                last = exc
                continue
            except Exception as exc:
                raise LayoutDetectorError(
                    f"Paddle inference failed with {self.active_model}: {type(exc).__name__}: {exc}"
                ) from exc
        raise LayoutDetectorError(
            f"Paddle predict API mismatch with {self.active_model}: {type(last).__name__}: {last}"
        )

    def detect(self, page_path: Path, page_index: int) -> tuple[list[LayoutBlock], float]:
        model = self._load()
        try:
            output = self._predict(model, page_path)
            result = next(iter(output))
            data = _result_to_dict(result)
        except LayoutDetectorError:
            raise
        except Exception as exc:
            raise LayoutDetectorError(
                f"Paddle result parsing failed: {type(exc).__name__}: {exc}"
            ) from exc

        payload = data.get("res", data)
        boxes = payload.get("boxes") or []
        blocks: list[LayoutBlock] = []
        scores: list[float] = []
        for idx, item in enumerate(boxes):
            coordinate = item.get("coordinate") or item.get("bbox")
            if not coordinate or len(coordinate) != 4:
                continue
            try:
                score = float(item.get("score", 0.0))
                coords = list(map(float, coordinate))
            except (TypeError, ValueError):
                continue
            label = str(item.get("label", "unknown"))
            category = category_for_label(label)
            blocks.append(
                LayoutBlock(
                    id=f"p{page_index:04d}_b{idx:04d}",
                    page=page_index,
                    category=category,  # type: ignore[arg-type]
                    source_label=label,
                    score=score,
                    bbox=BBox(*coords),
                    provider=f"paddle:{self.active_model}",
                )
            )
            scores.append(score)
        quality = statistics.fmean(scores) if scores else 0.0
        return blocks, quality
