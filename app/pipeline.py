from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.models import BBox, DocumentLayout, LayoutBlock, PageLayout
from app.services.costs import estimate_cost
from app.services.crops import materialize_crops
from app.services.figure_extraction import background_color as _background_color, extract_figures, residual_image
from app.services.grouping import build_processing_units
from app.services.input_pages import render_inputs
from app.services.jev_router import JevRouter
from app.services.layout_diagnostics import DEBUG_VERSION, LayoutDiagnostics
from app.services.layout_detector import PaddleLayoutDetector
from app.services.mistral_layout import MistralLayoutError, MistralLayoutRescue
from app.services.qwen_ocr import QwenOCRClient
from app.services.qwen_rescue import QwenRescueClient
from app.services.renderer import build_tex, compile_pdf
from app.services.validators import validate


def _looks_math(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    # A grouped paragraph can contain a few formulas among substantial prose.
    # Keep that prose in the text path instead of replacing it with math-only OCR.
    plain = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", value)
    plain = re.sub(r"\\[A-Za-z]+", " ", plain)
    math_words = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "lim", "exp", "max", "min", "det", "mod"}
    prose_words = [word for word in re.findall(r"[^\W\d_]{3,}", plain, flags=re.UNICODE) if word.lower() not in math_words]
    if len(prose_words) > 3:
        return False
    math_chars = set("=+-×÷∫∑√∞≈≠≤≥^_()[]{}|<>∂∆λμπσθΩ")
    symbol_count = sum(ch in math_chars for ch in value)
    alpha_count = sum(ch.isalpha() for ch in value)
    digit_count = sum(ch.isdigit() for ch in value)
    return (
        ("=" in value and len(value) <= 80 and (alpha_count <= 16 or digit_count > 0))
        or symbol_count >= 3
        or (alpha_count <= 3 and digit_count >= 2 and symbol_count >= 1)
    )


def _qwen_words_to_blocks(words: list[dict[str, Any]], page_index: int) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for idx, item in enumerate(words):
        loc = item.get("location") or []
        if len(loc) != 8:
            continue
        try:
            xs = [float(loc[i]) for i in (0, 2, 4, 6)]
            ys = [float(loc[i]) for i in (1, 3, 5, 7)]
        except (TypeError, ValueError):
            continue
        polygon = item.get("polygon") or []
        if len(polygon) != 8 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in polygon):
            polygon = []
        text = str(item.get("text", "")).strip()
        raw_category = str(item.get("category", "")).strip().lower()
        if raw_category in {"math", "formula", "equation", "matrix"}:
            category = "math"
        elif raw_category in {"table", "tabular"}:
            category = "table"
        elif raw_category in {"text", "title", "heading", "paragraph", "label"}:
            category = "text"
        else:
            # Position-only advanced-recognition output has no text. Start as text;
            # after decoding, math-like crops are automatically re-run as formula OCR.
            category = "math" if text and _looks_math(text) else "text"
        blocks.append(
            LayoutBlock(
                id=f"p{page_index:04d}_q{idx:04d}",
                page=page_index,
                category=category,  # type: ignore[arg-type]
                source_label=f"qwen_layout_{category}",
                score=0.95,
                bbox=BBox(min(xs), min(ys), max(xs), max(ys)),
                provider="qwen-vl-ocr-layout",
                hint_text=text,
                polygon=list(polygon),
            )
        )
    return blocks


def _make_residual_page(page_path: Path, blocks: list[LayoutBlock], out_path: Path) -> None:
    """Create a diagnostic residual; it is never a rendered page background."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_path) as src:
        image = residual_image(src, blocks)
        image.save(out_path, format="PNG", optimize=True)


class PipelineError(RuntimeError):
    pass


class Hand2TeXPipeline:
    def __init__(
        self,
        settings: Settings,
        detector: Any | None = None,
        qwen: Any | None = None,
        rescue: Any | None = None,
        mistral: Any | None = None,
        jev: Any | None = None,
    ):
        self.settings = settings
        self._detector_injected = detector is not None
        self.detector = detector or PaddleLayoutDetector(settings)
        self.qwen = qwen or QwenOCRClient(settings)
        self.rescue = rescue or QwenRescueClient(settings)
        self.mistral = mistral or MistralLayoutRescue(settings)
        self.jev = jev or JevRouter(settings)

    async def _decode_unit(self, unit, semaphore: asyncio.Semaphore) -> None:
        if unit.category == "figure":
            unit.decoder = "preserve-original"
            unit.validation_score = 1.0
            return
        if unit.category == "unknown":
            unit.category = "figure"
            unit.decoder = "preserve-unknown"
            unit.validation_score = 1.0
            return
        if not unit.crop_path:
            unit.validation_score = 0.0
            return

        # Qwen-VL-OCR layout pass already gives text + coordinates.
        # Reuse that text to avoid paying a second OCR call for normal text.
        if unit.category == "text" and unit.hint_text.strip():
            unit.decoded = unit.hint_text.strip()
            unit.decoder = "qwen-layout:text"
            unit.validation_score = validate("text", unit.decoded)
            if unit.validation_score >= self.settings.qwen_rescue_threshold:
                return

        async with semaphore:
            try:
                result = await self.qwen.decode(Path(unit.crop_path), unit.category)
                unit.decoded = result.text
                unit.usage_events.append({"provider": "qwen_ocr", **(result.usage or {})})
                unit.decoder = f"qwen-ocr:{unit.category}"
                unit.validation_score = validate(unit.category, unit.decoded)

                # If a position-only layout box was provisionally treated as text
                # but its OCR clearly looks mathematical, run the formula decoder
                # on that crop and keep the better structured result.
                if unit.category == "text" and not unit.hint_text.strip() and _looks_math(unit.decoded):
                    math_result = await self.qwen.decode(Path(unit.crop_path), "math")
                    math_score = validate("math", math_result.text)
                    unit.usage_events.append({"provider": "qwen_ocr", **(math_result.usage or {})})
                    if math_score >= unit.validation_score:
                        unit.category = "math"
                        unit.decoded = math_result.text
                        unit.decoder = "qwen-ocr:math-auto"
                        unit.validation_score = math_score
            except Exception as exc:
                unit.decoded = unit.decoded or ""
                unit.decoder = f"qwen-ocr-error:{type(exc).__name__}"
                unit.validation_score = validate(unit.category, unit.decoded) if unit.decoded else 0.0

            deterministic_rescue = unit.validation_score < self.settings.qwen_rescue_threshold
            jev_state = {
                "category": unit.category,
                "detector_score": round(unit.detector_score, 4),
                "validation_score": round(unit.validation_score, 4),
                "decoded_length": len(unit.decoded),
                "member_count": len(unit.member_ids),
            }
            jev_decision = await self.jev.should_rescue(jev_state)
            should_rescue = deterministic_rescue if jev_decision is None else jev_decision

            if should_rescue and self.rescue.available:
                try:
                    rr = await self.rescue.decode(Path(unit.crop_path), unit.category)
                    rescue_score = validate(unit.category, rr.text)
                    if rescue_score >= unit.validation_score:
                        unit.decoded = rr.text
                        unit.usage_events.append({"provider": "qwen_rescue", **(rr.usage or {})})
                        unit.decoder = f"qwen-rescue:{unit.category}"
                        unit.validation_score = rescue_score
                        unit.rescued = True
                except Exception:
                    pass

    async def _layout_qwen(self, page, detector_errors, layout_usage_events, diagnostics=None):
        trace = diagnostics.for_page(page.index, page.width, page.height) if diagnostics else None
        try:
            if trace:
                located = await self.qwen.locate_text_lines(page.path, diagnostics=trace)
            else:
                located = await self.qwen.locate_text_lines(page.path)
            blocks = _qwen_words_to_blocks(located.words_info, page.index)
        except Exception as exc:
            if trace:
                trace.fail(exc)
            raise PipelineError(f"Qwen layout call failed on page {page.index + 1}: {type(exc).__name__}: {exc}") from exc

        layout_usage_events.append({"provider": "qwen_ocr", **(located.usage or {})})

        if not blocks:
            # Layout failure must not silently replace transcription with a scan.
            # Decode the page once as prose and attach the source separately.
            detector_errors.append({
                "page": page.index, "stage": "qwen-layout",
                "error": "0 usable boxes; full-page text OCR fallback, with original source attached separately",
            })
            blocks = [
                LayoutBlock(
                    id=f"p{page.index:04d}_fullpage",
                    page=page.index,
                    category="text",
                    source_label="whole_page_ocr_fallback",
                    score=0.0,
                    bbox=BBox(0, 0, page.width, page.height),
                    provider="qwen-vl-ocr-layout",
                )
            ]
            if trace:
                trace.record_parsed(located.words_info, blocks, 0.0, "whole_page_ocr_fallback")
            return blocks, 0.0, "qwen-vl-ocr-layout"

        if trace:
            trace.record_parsed(located.words_info, blocks, 0.95)
        return blocks, 0.95, "qwen-vl-ocr-layout"

    async def run(self, input_paths: list[Path], work_dir: Path, title: str, include_debug: bool = False) -> dict[str, Any]:
        pages_dir = work_dir / "pages"
        crops_dir = work_dir / "crops"
        result_dir = work_dir / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = LayoutDiagnostics(result_dir, self.settings.trustedrouter_api_key) if include_debug else None

        pages = render_inputs(input_paths, pages_dir, self.settings.render_dpi, self.settings.max_pages)
        if not pages:
            raise PipelineError("No pages found")
        if not self.qwen.available:
            raise PipelineError("TRUSTEDROUTER_API_KEY is required for decoding")

        layouts: list[PageLayout] = []
        raw_rescues: list[dict[str, Any]] = []
        detector_errors: list[dict[str, Any]] = []
        content_warnings: list[dict[str, Any]] = []
        layout_usage_events: list[dict[str, Any]] = []
        mistral_pages = 0

        for page in pages:
            detector_name = ""
            rescued_layout = False
            blocks: list[LayoutBlock] = []
            quality = 0.0

            use_qwen_layout = self.settings.layout_backend == "qwen" and not self._detector_injected

            if use_qwen_layout:
                blocks, quality, detector_name = await self._layout_qwen(page, detector_errors, layout_usage_events, diagnostics)
            else:
                try:
                    blocks, quality = self.detector.detect(page.path, page.index)
                    active = getattr(self.detector, "active_model", "")
                    detector_name = f"paddle:{active}" if active else "paddle"
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    detector_name = f"paddle-error:{type(exc).__name__}"
                    detector_errors.append({"page": page.index, "stage": "paddle", "error": detail})
                    blocks, quality = [], 0.0

                needs_rescue = quality < self.settings.layout_rescue_threshold or len(blocks) < self.settings.layout_min_blocks
                if needs_rescue and self.mistral.available:
                    try:
                        blocks, quality, raw = await self.mistral.detect(page.path, page.index, page.width, page.height)
                        detector_name = "mistral-layout-rescue"
                        rescued_layout = True
                        mistral_pages += 1
                        if include_debug:
                            raw_rescues.append({"page": page.index, "response": raw})
                    except MistralLayoutError as exc:
                        detector_errors.append({"page": page.index, "stage": "mistral-layout", "error": str(exc)})

                if not blocks:
                    blocks, quality, detector_name = await self._layout_qwen(page, detector_errors, layout_usage_events, diagnostics)
                    rescued_layout = True

            page_crop_dir = crops_dir / f"page_{page.index:04d}"
            figure_crops: dict[str, str] = {}
            if detector_name.startswith("qwen-vl-ocr-layout") and quality > 0:
                figures = extract_figures(page.path, blocks, page_crop_dir / "illustrations", page.index)
                blocks.extend(figures.blocks)
                figure_crops = figures.crops
                content_warnings.extend({"page": page.index, "stage": "source-fragments", "warning": warning} for warning in figures.warnings)
                if diagnostics:
                    trace = next((trace for trace in diagnostics.pages if trace.parsed["page"] == page.index), None)
                    if trace:
                        trace.record_parsed(trace.parsed["words_info"], blocks, quality, trace.parsed["fallback"])

            pl = PageLayout(
                index=page.index,
                width_px=page.width,
                height_px=page.height,
                image_path=str(page.path),
                blocks=blocks,
                detector=detector_name,
                detector_quality=quality,
                rescued_layout=rescued_layout,
            )
            pl.units = build_processing_units(blocks, page.width, page.height, page.index)
            materialize_crops(pl, page_crop_dir)
            for unit in pl.units:
                source_id = next((identifier for identifier in unit.member_ids if identifier in figure_crops), None)
                if source_id:
                    unit.crop_path = figure_crops[source_id]

            layouts.append(pl)

        layout = DocumentLayout(layouts)
        semaphore = asyncio.Semaphore(max(1, self.settings.qwen_concurrency))
        await asyncio.gather(*(self._decode_unit(unit, semaphore) for p in layout.pages for unit in p.units))

        all_units = [u for p in layout.pages for u in p.units]
        missing_units = [unit for unit in all_units if unit.category in {"text", "math", "table"} and not unit.decoded.strip()]
        content_warnings.extend({"page": unit.page, "stage": "ocr", "unit": unit.id, "warning": "This content region could not be transcribed; original source attached separately."} for unit in missing_units)
        source_pages = []
        warned_pages = {item["page"] for item in content_warnings} | {page.index for page in layout.pages if page.detector_quality == 0}
        for page in pages:
            if page.index in warned_pages:
                source_path = result_dir / "sources" / f"page_{page.index:04d}.png"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(page.path, source_path)
                source_pages.append({"page": page.index, "path": source_path.relative_to(result_dir).as_posix(), "purpose": "original source for review; excluded from typeset document"})

        tex_path = build_tex(layout, result_dir, title)
        manifest = {
            "version": DEBUG_VERSION,
            "status": "warning" if warned_pages else "ok",
            "title": title,
            "pages": len(layout.pages),
            "detectors": [
                {
                    "page": p.index,
                    "provider": p.detector,
                    "quality": p.detector_quality,
                    "rescued": p.rescued_layout,
                }
                for p in layout.pages
            ],
            "decoders": [
                {
                    "id": u.id,
                    "category": u.category,
                    "decoder": u.decoder,
                    "validation": u.validation_score,
                    "rescued": u.rescued,
                }
                for u in all_units
            ],
            "cost_estimate": estimate_cost(self.settings, all_units, mistral_pages, layout_usage_events),
            "detector_errors": detector_errors,
            "content_warnings": content_warnings,
            "source_pages": source_pages,
            "output_mode": "flow-document",
            "architecture": "TrustedRouter Qwen-VL-OCR layout -> compact grouped OCR crops and local illustration crops -> category OCR -> Qwen3.8 Max rescue -> reading order -> standard flowing LaTeX document",
        }
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / "layout.json").write_text(json.dumps(layout.to_jsonable(work_dir), ensure_ascii=False, indent=2), encoding="utf-8")
        (result_dir / "decoded.json").write_text(
            json.dumps(
                [
                    {
                        "id": u.id,
                        "page": u.page,
                        "category": u.category,
                        "bbox": u.bbox.__dict__,
                        "decoded": u.decoded,
                        "decoder": u.decoder,
                        "validation": u.validation_score,
                    }
                    for u in all_units
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if include_debug:
            (result_dir / "layout_rescue_raw.json").write_text(json.dumps(raw_rescues, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep decoded text, routing and costs available in a diagnostic ZIP even
        # if the external PDF compiler fails.
        try:
            compilation = compile_pdf(tex_path)
        except Exception as exc:
            manifest["status"] = "error"
            manifest["compilation"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        manifest["compilation"] = {"ok": bool(compilation.get("ok")), "reason": compilation.get("reason")}
        if not compilation.get("ok"):
            manifest["status"] = "error"
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not compilation.get("ok"):
            raise PipelineError(f"LaTeX compilation failed: {compilation.get('reason')}. See compile.log")
        return {"result_dir": result_dir, "manifest": manifest, "compilation": compilation}
