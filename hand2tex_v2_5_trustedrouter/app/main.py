from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app.config import Settings
from app.pipeline import Hand2TeXPipeline, PipelineError

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
settings = Settings()
app = FastAPI(title=settings.app_name, version="0.2.4")
pipeline = Hand2TeXPipeline(settings)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _cleanup(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("{{APP_NAME}}", settings.app_name))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "qwen_configured": bool(settings.dashscope_api_key),
        "mistral_layout_rescue": bool(settings.mistral_api_key and settings.enable_mistral_layout_rescue),
        "jev": bool(settings.typesafe_api_key and settings.jev_enabled),
        "layout_backend": settings.layout_backend,
        "paddle_model": settings.paddle_model if settings.layout_backend == "paddle" else None,
    }


@app.get("/health/paddle")
def health_paddle() -> dict:
    try:
        model = pipeline.detector._load()
        return {
            "status": "ok",
            "model": getattr(pipeline.detector, "active_model", settings.paddle_model),
            "type": type(model).__name__,
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": settings.paddle_model,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.post("/api/convert")
async def convert(
    files: list[UploadFile] = File(...),
    title: str = Form("Converted Notes"),
    include_debug: bool = Form(False),
) -> FileResponse:
    if not files:
        raise HTTPException(400, "Upload at least one file")
    if len(files) > settings.max_files:
        raise HTTPException(400, f"Maximum {settings.max_files} files")

    temp_root = tempfile.mkdtemp(prefix="hand2tex_v2_")
    root = Path(temp_root)
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        for idx, upload in enumerate(files):
            filename = Path(upload.filename or f"upload_{idx}").name
            suffix = Path(filename).suffix.lower()
            if suffix not in allowed:
                raise HTTPException(415, f"Unsupported file: {filename}")
            data = await upload.read()
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"{filename} exceeds {settings.max_upload_mb} MB")
            target = upload_dir / f"{idx:03d}_{filename}"
            target.write_bytes(data)
            paths.append(target)

        result = await pipeline.run(paths, root / "work", title.strip() or "Converted Notes", include_debug=include_debug)
        result_dir: Path = result["result_dir"]
        zip_path = root / "hand2tex_result.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in result_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(result_dir))
        return FileResponse(
            zip_path, media_type="application/zip", filename="hand2tex_result.zip",
            background=BackgroundTask(_cleanup, temp_root),
        )
    except HTTPException:
        _cleanup(temp_root)
        raise
    except PipelineError as exc:
        _cleanup(temp_root)
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        _cleanup(temp_root)
        raise HTTPException(500, f"Conversion failed: {exc}") from exc
