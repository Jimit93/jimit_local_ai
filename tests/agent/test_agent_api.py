"""Agent HTTP API through the real app, with the worker running a scripted model."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agent import governor as G
from backend.agent import worker as W
from backend.llm import manager as M

FIX = "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"


class Script:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt, **kw):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else "`add` is defined in `calc.py`."
        return type("R", (), {"text": text, "model": "scripted", "total_tokens": 20, "prompt_tokens": 15,
                              "completion_tokens": 5, "meta": {"tier": kw.get("tier")}})()


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.setattr(M, "warmup", lambda: {"stubbed": True})
    g = G.Governor(sampler=lambda: (10.0, 1.0, 8.0), interval_s=0.2, persist=False)
    g.tick()
    G.set_governor(g)
    w = W.AgentWorker(llm=Script(FIX))
    W.set_worker(w)
    from backend.api.app import app

    with TestClient(app) as c:
        c.workspace = workspace
        c.worker = w
        yield c
    W.set_worker(None)
    G.set_governor(None)


def _wait(c, task_id, statuses=("completed", "failed", "blocked", "cancelled", "awaiting_approval"), timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        t = c.get(f"/api/agent/tasks/{task_id}").json()
        if t["status"] in statuses:
            return t
        time.sleep(0.2)
    raise AssertionError(f"task {task_id} did not finish: {t['status']}")


def test_status_models_workspaces(client):
    s = client.get("/api/agent/status").json()
    assert s["governor"]["state"] == "NORMAL" and "worker" in s
    assert client.workspace in client.get("/api/agent/workspaces").json()["workspaces"]
    assert "installed" in client.get("/api/agent/models").json()


def test_run_fix_task_end_to_end(client):
    r = client.post("/api/agent/run", json={"request": "fix the failing test in tests/test_calc.py",
                                            "workspace": client.workspace, "allow_edits": True}).json()
    assert r["category"] == "coding" and r["plan"][0]["name"] == "reproduce"
    t = _wait(client, r["task_id"])
    assert t["status"] == "completed", t["last_output"]
    assert "return a + b" in Path(client.workspace, "calc.py").read_text(encoding="utf-8")
    ev = client.get(f"/api/agent/tasks/{r['task_id']}/events").json()["events"]
    assert ev[0]["kind"] == "understand" and ev[-1]["kind"] == "complete"
    cps = client.get("/api/agent/checkpoints", params={"task_id": r["task_id"]}).json()["checkpoints"]
    assert cps and client.get(f"/api/agent/checkpoints/{cps[0]['id']}/diff").json()["changed_files"] == ["calc.py"]
    rb = client.post(f"/api/agent/checkpoints/{cps[0]['id']}/rollback", json={}).json()
    assert rb["ok"] and "return a - b" in Path(client.workspace, "calc.py").read_text(encoding="utf-8")
    evaluation = client.get("/api/agent/evaluation").json()["tasks"]
    assert evaluation["n"] >= 1 and evaluation["statuses"].get("completed", 0) >= 1


def test_non_coding_request_goes_through_the_harness(client):
    r = client.post("/api/agent/run", json={"request": "Calculate 345 × 67", "workspace": client.workspace}).json()
    assert r["category"] == "deterministic"
    t = _wait(client, r["task_id"])
    assert t["status"] == "completed" and "23115" in t["last_output"] and t["counters"]["llm_calls"] == 0


def test_deterministic_workspace_request(client):
    r = client.post("/api/agent/run", json={"request": "git status", "workspace": client.workspace}).json()
    t = _wait(client, r["task_id"])
    assert t["status"] == "completed" and t["plan"][0]["method"] == "deterministic"


def test_approval_flow_and_denial(client):
    r = client.post("/api/agent/run", json={"request": "delete the file config.json", "workspace": client.workspace}).json()
    t = _wait(client, r["task_id"])
    assert t["status"] == "awaiting_approval"
    assert client.post(f"/api/agent/tasks/{r['task_id']}/approve", json={"approve": False}).json()["status"] == "cancelled"
    assert Path(client.workspace, "config.json").exists()
    r2 = client.post("/api/agent/run", json={"request": "delete the file config.json", "workspace": client.workspace}).json()
    _wait(client, r2["task_id"])
    client.post(f"/api/agent/tasks/{r2['task_id']}/approve", json={"approve": True})
    assert _wait(client, r2["task_id"], ("completed", "failed"))["status"] == "completed"
    assert not Path(client.workspace, "config.json").exists()
    assert any(a["decision"] == "approved" for a in client.get("/api/agent/audit").json()["audit"])


def test_files_memory_ram_endpoints(client):
    tree = client.get("/api/agent/files/tree", params={"workspace": client.workspace}).json()
    names = {c["name"] for c in tree["root"]["children"]}
    assert "calc.py" in names and "node_modules" not in names
    f = client.get("/api/agent/files/read", params={"workspace": client.workspace, "path": "calc.py"}).json()
    assert "def add" in f["content"]
    assert client.get("/api/agent/files/read", params={"workspace": client.workspace, "path": ".env"}).status_code == 403
    assert client.get("/api/agent/files/read", params={"workspace": client.workspace, "path": "../x"}).status_code == 403
    assert "recent" in client.get("/api/agent/files/activity", params={"workspace": client.workspace}).json()
    mem = client.get("/api/agent/memory", params={"workspace": client.workspace}).json()
    assert mem["project"]["test_command"] == "python -m pytest"
    ram = client.get("/api/agent/ram").json()
    assert ram["available_gb"] > 0 and all(p["category"] in ("protected", "candidate", "allowlisted", "agent_owned")
                                           for p in ram["processes"])


def test_bad_inputs(client):
    assert client.post("/api/agent/run", json={"request": ""}).status_code == 422
    assert client.post("/api/agent/run", json={"request": "git status", "workspace": "C:\\Windows"}).status_code == 403
    assert client.get("/api/agent/tasks/doesnotexist").status_code == 404
    assert client.post("/api/agent/workspaces", json={"path": "C:\\Windows"}).status_code == 403


def test_cancel_queued_task(client):
    client.worker.max_concurrent = 0  # nothing can start: the task stays queued
    r = client.post("/api/agent/run", json={"request": "explain how the Accumulator class works",
                                            "workspace": client.workspace}).json()
    assert client.post(f"/api/agent/tasks/{r['task_id']}/cancel").json()["was"] == "queued"
    assert client.get(f"/api/agent/tasks/{r['task_id']}").json()["status"] == "cancelled"
    client.worker.max_concurrent = 1
