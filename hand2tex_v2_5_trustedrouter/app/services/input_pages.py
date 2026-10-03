from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageOps


@dataclass(frozen=True)
class RenderedPage:
    index: int
    path: Path
    width: int
    height: int
    source_name: str
    source_page: int


ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def render_inputs(paths: list[Path], out_dir: Path, dpi: int, max_pages: int) -> list[RenderedPage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[RenderedPage] = []
    global_index = 0

    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported input: {path.name}")
        if suffix == ".pdf":
            doc = fitz.open(path)
            try:
                scale = dpi / 72.0
                matrix = fitz.Matrix(scale, scale)
                for source_page, page in enumerate(doc):
                    if len(pages) >= max_pages:
                        raise ValueError(f"Maximum {max_pages} pages per conversion")
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    target = out_dir / f"page_{global_index:04d}.png"
                    pix.save(str(target))
                    pages.append(RenderedPage(global_index, target, pix.width, pix.height, path.name, source_page))
                    global_index += 1
            finally:
                doc.close()
        else:
            if len(pages) >= max_pages:
                raise ValueError(f"Maximum {max_pages} pages per conversion")
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                target = out_dir / f"page_{global_index:04d}.png"
                image.save(target, format="PNG", optimize=True)
                pages.append(RenderedPage(global_index, target, image.width, image.height, path.name, 0))
                global_index += 1
    return pages
