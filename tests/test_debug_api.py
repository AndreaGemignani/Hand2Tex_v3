import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from app.pipeline import PipelineError
from app.services.layout_diagnostics import LayoutDiagnostics


API_KEY = "sk-tr-api-debug-secret"
DEBUG_FILES = {
    "qwen_layout_raw.json",
    "request_payload_sanitized.json",
    "parsed_layout.json",
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_main, "settings", Settings(
        trustedrouter_api_key=API_KEY, include_debug_by_default=True,
    ))
    with TestClient(app_main.app) as test_client:
        yield test_client


def upload(client, *, include_debug=None):
    data = {"title": "Debug API"}
    if include_debug is not None:
        data["include_debug"] = "true" if include_debug else "false"
    return client.post(
        "/api/convert", data=data,
        files=[("files", ("notes.png", b"test-image", "image/png"))],
    )


def zip_members(response):
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.mark.parametrize("failure", ["layout", "unexpected", "compilation"])
def test_debug_failures_return_downloadable_redacted_diagnostics(client, monkeypatch, failure):
    seen = {}

    async def failing_run(paths, work_dir, title, include_debug=False):
        seen["include_debug"] = include_debug
        seen["temp_root"] = work_dir.parent
        result_dir = work_dir / "result"
        diagnostics = LayoutDiagnostics(result_dir, API_KEY)
        diagnostics.for_page(0, 400, 600)
        if failure == "compilation":
            (result_dir / "compile.log").write_text("pdflatex compilation failed", encoding="utf-8")
            raise PipelineError(f"LaTeX compilation failed: {API_KEY}")
        if failure == "unexpected":
            raise RuntimeError(f"Unexpected conversion failure: {API_KEY}")
        raise PipelineError(f"Qwen layout call failed: {API_KEY}")

    monkeypatch.setattr(app_main.pipeline, "run", failing_run)
    response = upload(client, include_debug=True)
    assert response.status_code == 200
    assert response.headers["X-Hand2TeX-Status"] == "error"
    assert "hand2tex_debug.zip" in response.headers["content-disposition"]
    members = zip_members(response)
    assert DEBUG_FILES <= members.keys()
    error = json.loads(members["diagnostic_error.json"])
    assert error["status"] == "error"
    assert error["error"]
    assert API_KEY not in error["error"]
    assert all(API_KEY.encode() not in members[name] for name in DEBUG_FILES | {"diagnostic_error.json"})
    assert seen["include_debug"] is True
    assert not seen["temp_root"].exists()
    if failure == "compilation":
        assert members["compile.log"] == b"pdflatex compilation failed"


@pytest.mark.parametrize("error_type,status_code", [(PipelineError, 502), (RuntimeError, 500)])
def test_non_debug_failures_keep_http_errors(client, monkeypatch, error_type, status_code):
    seen = {}

    async def failing_run(paths, work_dir, title, include_debug=False):
        seen["include_debug"] = include_debug
        seen["temp_root"] = work_dir.parent
        raise error_type("conversion failed")

    monkeypatch.setattr(app_main.pipeline, "run", failing_run)
    response = upload(client, include_debug=False)
    assert response.status_code == status_code
    assert response.headers["content-type"] == "application/json"
    assert "conversion failed" in response.json()["detail"]
    assert seen["include_debug"] is False
    assert not seen["temp_root"].exists()


@pytest.mark.parametrize("status", ["ok", "warning"])
def test_success_zip_reports_layout_status_and_enables_debug_by_default(
    client, monkeypatch, status,
):
    seen = {}

    async def successful_run(paths, work_dir, title, include_debug=False):
        seen["include_debug"] = include_debug
        seen["temp_root"] = work_dir.parent
        result_dir = work_dir / "result"
        diagnostics = LayoutDiagnostics(result_dir, API_KEY)
        diagnostics.for_page(0, 400, 600)
        parsed_status = "ok" if status == "ok" else "no_usable_boxes"
        parsed = {
            "version": "2.9-debug",
            "pages": [{
                "page": 0, "width_px": 400, "height_px": 600,
                "status": parsed_status, "words_info": [], "blocks": [],
                "usable_box_count": 1 if status == "ok" else 0,
                "quality": .95 if status == "ok" else 0,
                "fallback": None if status == "ok" else "whole_page_no_text",
            }],
        }
        (result_dir / "parsed_layout.json").write_text(json.dumps(parsed), encoding="utf-8")
        (result_dir / "main.pdf").write_bytes(b"%PDF-1.4\n")
        manifest = {
            "status": status,
            "detectors": [{"page": 0, "quality": .95 if status == "ok" else 0}],
            "detector_errors": [] if status == "ok" else [{"page": 0, "error": "No usable boxes"}],
        }
        (result_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"result_dir": result_dir, "manifest": manifest}

    monkeypatch.setattr(app_main.pipeline, "run", successful_run)
    response = upload(client)
    assert response.status_code == 200
    assert response.headers["X-Hand2TeX-Status"] == status
    assert "hand2tex_result.zip" in response.headers["content-disposition"]
    assert DEBUG_FILES | {"main.pdf", "manifest.json"} <= zip_members(response).keys()
    assert seen["include_debug"] is True
    assert not seen["temp_root"].exists()
