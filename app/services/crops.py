from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.models import PageLayout


def materialize_crops(page: PageLayout, out_dir: Path, padding_px: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page.image_path) as image:
        image = image.convert("RGB")
        for unit in page.units:
            bbox = unit.bbox.expand(padding_px, page.width_px, page.height_px)
            left, top, right, bottom = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            if right <= left or bottom <= top:
                continue
            crop = image.crop((left, top, right, bottom))
            path = out_dir / f"{unit.id}.png"
            crop.save(path, format="PNG", optimize=True)
            unit.crop_path = str(path)
