import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.services.jev_router import JevRouter
from app.services.mistral_layout import MistralLayoutRescue


@pytest.mark.asyncio
async def test_jev_reads_noul(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("JEV_ENABLED", "true")
    s = Settings()
    async def handler(request):
        return httpx.Response(200, json={"answers": {"rescue": {"noul": 0.8}}})
    router = JevRouter(s, transport=httpx.MockTransport(handler))
    assert await router.should_rescue({"validation_score": .4}) is True


def test_mistral_available_flag(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    assert MistralLayoutRescue(Settings()).available is True
