from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Category = Literal["text", "math", "table", "figure", "unknown"]


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, width: float, height: float) -> "BBox":
        return BBox(
            max(0.0, min(width, self.x1)),
            max(0.0, min(height, self.y1)),
            max(0.0, min(width, self.x2)),
            max(0.0, min(height, self.y2)),
        )

    def expand(self, px: float, width: float, height: float) -> "BBox":
        return BBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px).clamp(width, height)

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2),
        )

    def intersection_area(self, other: "BBox") -> float:
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contained_fraction(self, outer: "BBox") -> float:
        return 0.0 if self.area <= 0 else self.intersection_area(outer) / self.area

    def horizontal_overlap_ratio(self, other: "BBox") -> float:
        overlap = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        denom = min(self.width, other.width)
        return 0.0 if denom <= 0 else overlap / denom

    def normalize(self, width: float, height: float) -> list[float]:
        return [self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height]


@dataclass
class LayoutBlock:
    id: str
    page: int
    category: Category
    source_label: str
    score: float
    bbox: BBox
    provider: str = "paddle"
    hint_text: str = ""
    polygon: list[float] = field(default_factory=list)


@dataclass
class ProcessingUnit:
    id: str
    page: int
    category: Category
    bbox: BBox
    member_ids: list[str]
    detector_score: float
    crop_path: str = ""
    decoded: str = ""
    decoder: str = ""
    validation_score: float = 0.0
    rescued: bool = False
    usage_events: list[dict[str, Any]] = field(default_factory=list)
    hint_text: str = ""
    polygons: list[list[float]] = field(default_factory=list)


@dataclass
class PageLayout:
    index: int
    width_px: int
    height_px: int
    image_path: str
    blocks: list[LayoutBlock] = field(default_factory=list)
    units: list[ProcessingUnit] = field(default_factory=list)
    detector: str = ""
    detector_quality: float = 0.0
    rescued_layout: bool = False


@dataclass
class DocumentLayout:
    pages: list[PageLayout]

    def to_jsonable(self, root: Path | None = None) -> dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                value = str(obj)
                if root:
                    try:
                        return str(obj.relative_to(root))
                    except ValueError:
                        return value
                return value
            if isinstance(obj, BBox):
                return asdict(obj)
            if isinstance(obj, (LayoutBlock, ProcessingUnit, PageLayout)):
                d = asdict(obj)
                return {k: convert(v) for k, v in d.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        return {"pages": convert(self.pages)}
