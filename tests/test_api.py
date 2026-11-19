import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

from querentcrew import api  # noqa: E402
from querentcrew.service import QuerentCrew  # noqa: E402

from .fakes import FakeLLM  # noqa: E402
from .scenarios import pii_script  # noqa: E402


@pytest.fixture
def client(settings, monkeypatch):
    pilot = QuerentCrew(settings, llm=FakeLLM(pii_script()))
    monkeypatch.setattr(api, "get_pilot", lambda: pilot)
    return TestClient(api.app)


def test_health_and_schema(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "orders" in client.get("/schema").json()["tables"]


def test_ask_then_approve(client):
    first = client.post("/ask", json={"question": "emails of enterprise customers"}).json()
    assert first["status"] == "awaiting_approval"
    done = client.post("/approve", json={"thread_id": first["thread_id"], "approved": True}).json()
    assert done["status"] == "completed"
    history = client.get(f"/history/{first['thread_id']}").json()["history"]
    assert history[-1]["role"] == "assistant"
