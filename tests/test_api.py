"""API tests.

Warm-up is stubbed out so the test client does not load a multi-GB model, and
every endpoint exercised here is one that must answer without the LLM.
"""
import pytest
from fastapi.testclient import TestClient

from backend.llm import manager as M


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    M.warmup = lambda: {"stubbed": True}  # noqa: E731 - keep the test fast and offline
    from backend.api.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_system_reports_discovered_models(client):
    d = client.get("/api/system").json()
    assert d["cpu_logical"] > 0 and d["ram_total_gb"] > 0
    assert "models_found" in d and "db" in d
    assert d["gpu"] is False


def test_task_endpoint_routes_arithmetic_without_llm(client):
    d = client.post("/api/task", json={"task": "What is 25 * 40?"}).json()
    assert d["answer"].startswith("1000")
    assert d["llm_used"] is False
    assert d["status"] == "SUCCESS"


def test_task_endpoint_rejects_a_bad_mode(client):
    assert client.post("/api/task", json={"task": "hi", "mode": "nonsense"}).status_code == 400


def test_task_endpoint_validates_input(client):
    assert client.post("/api/task", json={"task": ""}).status_code == 422


def test_knowledge_stats(client):
    d = client.get("/api/knowledge/stats").json()
    assert "documents" in d and "retrieval" in d


def test_search_is_llm_free(client):
    d = client.post("/api/knowledge/search",
                    json={"q": "local ai", "mode": "keyword", "top_k": 3}).json()
    assert d["llm_used"] is False
    assert d["latency_ms"] >= 0
    assert isinstance(d["hits"], list)


def test_style_profile(client):
    r = client.get("/api/style/profile")
    if r.status_code == 404:
        pytest.skip("no profile built")
    d = r.json()
    assert d["profile"]["n_documents"] > 0
    assert d["constraints"]["target_words"]


def test_metrics_summary(client):
    d = client.get("/api/metrics/summary").json()
    assert "summary" in d and "process" in d


def test_settings_roundtrip(client):
    before = client.get("/api/settings").json()["llm"]["max_tokens"]
    d = client.post("/api/settings", json={"max_tokens": 321}).json()
    assert d["llm"]["max_tokens"] == 321
    client.post("/api/settings", json={"max_tokens": before})


def test_cors_is_localhost_only(client):
    from backend.api.app import app

    for mw in app.user_middleware:
        origins = mw.kwargs.get("allow_origins") if hasattr(mw, "kwargs") else None
        if origins:
            assert all("localhost" in o or "127.0.0.1" in o for o in origins)
