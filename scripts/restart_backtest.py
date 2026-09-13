"""Restart / resume backtest with real process kills.

A: kill the server right after the agent checkpoints a fix -> restart -> it must recover
   the task, auto-resume (project unchanged), finish, and remember task/step/files/
   decisions/checkpoints.
B: kill after context gathering, edit an active file while the server is down -> restart
   -> it must NOT auto-resume; a manual resume must re-inspect first and still finish.
C: corrupt a saved task while the server is down -> restart -> server healthy, task blocked.

    python -m scripts.restart_backtest [--port 8079]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.benchmarks.agent_backtest import make_fixture, run_pytest  # noqa: E402

TERMINAL = {"completed", "failed", "blocked", "cancelled"}


class Server:
    def __init__(self, port: int, db: Path):
        self.port, self.db, self.proc = port, db, None

    def start(self, auto_resume: bool = True) -> float:
        env = {**os.environ, "JLA_DB_PATH": str(self.db), "JLA_SKIP_WARMUP": "1", "PYTHONPATH": str(ROOT),
               "PYTHONIOENCODING": "utf-8", "JLA_AGENT_AUTO_RESUME": "1" if auto_resume else "0"}
        t0 = time.time()
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.api.app:app", "--host", "127.0.0.1",
                                      "--port", str(self.port)], cwd=ROOT, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
        while time.time() - t0 < 90:
            try:
                self.get("/api/health", timeout=2)
                return round(time.time() - t0, 2)
            except Exception:
                time.sleep(0.4)
        raise RuntimeError("server did not start")

    def kill(self) -> None:
        """Hard kill of the whole tree - no graceful shutdown, like a crash or closed window."""
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
            self.proc.wait(timeout=20)

    def get(self, path: str, timeout: float = 30):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict, timeout: float = 60):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())


def wait_event(srv: Server, task_id: str, pred, timeout: float) -> dict | None:
    end = time.time() + timeout
    while time.time() < end:
        evs = srv.get(f"/api/agent/tasks/{task_id}/events")["events"]
        hit = next((e for e in evs if pred(e)), None)
        if hit:
            return hit
        if srv.get(f"/api/agent/tasks/{task_id}")["status"] in TERMINAL:
            return None
        time.sleep(0.3)
    return None


def wait_status(srv: Server, task_id: str, statuses: set, timeout: float) -> dict:
    end = time.time() + timeout
    t = {}
    while time.time() < end:
        t = srv.get(f"/api/agent/tasks/{task_id}")
        if t["status"] in statuses:
            return t
        time.sleep(0.5)
    return t


def db_row(db: Path, task_id: str) -> dict:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM agent_tasks WHERE id=?", (task_id,)).fetchone()
        return dict(r) if r else {}
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8079)
    args = ap.parse_args()
    base = Path(tempfile.mkdtemp(prefix="jla_restart_"))
    db = base / "restart.db"
    srv = Server(args.port, db)
    report: dict = {"ts": time.time(), "scenarios": {}}

    # ---------------------------------------------------------------- A
    ws = make_fixture(base / "a" / "shop", ("discount_sign",))
    srv.start()
    srv.post("/api/agent/workspaces", {"path": str(ws)})
    run = srv.post("/api/agent/run", {"request": "fix the failing test in tests/test_pricing.py", "workspace": str(ws),
                                      "allow_edits": True})
    tid = run["task_id"]
    ev = wait_event(srv, tid, lambda e: e["kind"] == "checkpoint", 900)
    before = srv.get(f"/api/agent/tasks/{tid}")
    srv.kill()
    t_kill = time.time()
    row_after_kill = db_row(db, tid)
    startup_s = srv.start(auto_resume=True)
    t_back = srv.get(f"/api/agent/tasks/{tid}")
    final = wait_status(srv, tid, TERMINAL, 1200)
    evs = srv.get(f"/api/agent/tasks/{tid}/events")["events"]
    tests = run_pytest(ws)
    interrupted = next((e for e in evs if e["kind"] == "interrupted"), None)
    resumed = next((e for e in evs if e["kind"] == "resume"), None)
    checks = {
        "killed_at_checkpoint": ev is not None,
        "db_status_running_after_kill": row_after_kill.get("status") == "running",
        "marked_interrupted_on_restart": interrupted is not None,
        "auto_resumed": resumed is not None,
        "completed_after_resume": final.get("status") == "completed",
        "tests_pass_after_resume": tests["exit"] == 0,
        "remembers_task": final.get("task") == before.get("task") and final.get("topic") == before.get("topic"),
        "remembers_completed_steps": set(before.get("completed_steps", [])) <= set(final.get("completed_steps", [])),
        "remembers_active_files": "pricing.py" in final.get("active_files", []),
        "remembers_checkpoints": bool(final.get("checkpoints")) and final.get("last_checkpoint") is not None,
        "records_resume_decision": any("resumed" in str(d.get("text", "")) for d in final.get("decisions", [])),
        "has_next_action": bool(final.get("next_action")),
    }
    report["scenarios"]["A_kill_after_checkpoint"] = {
        "checks": checks, "passed": all(checks.values()), "step_at_kill": before.get("current_step"),
        "resume_step": (resumed or {}).get("step"), "resume_reason": (resumed or {}).get("payload", {}).get("reason"),
        "server_restart_s": startup_s, "kill_to_ready_s": round(time.time() - t_kill, 1) if t_back else None,
        "final_status": final.get("status"), "tests": tests,
    }
    print("A", json.dumps(report["scenarios"]["A_kill_after_checkpoint"], indent=1, default=str)[:1500], flush=True)
    srv.kill()

    # ---------------------------------------------------------------- B
    ws_b = make_fixture(base / "b" / "shop", ("discount_sign",))
    srv.start()
    srv.post("/api/agent/workspaces", {"path": str(ws_b)})
    run = srv.post("/api/agent/run", {"request": "fix the failing test in tests/test_pricing.py", "workspace": str(ws_b),
                                      "allow_edits": True})
    tid_b = run["task_id"]
    wait_event(srv, tid_b, lambda e: e["kind"] == "step_end" and e["step"] == "gather_context", 600)
    srv.kill()
    p = ws_b / "pricing.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# edited by the user while the agent was stopped\n", encoding="utf-8")
    srv.start(auto_resume=True)
    time.sleep(3)
    t = srv.get(f"/api/agent/tasks/{tid_b}")
    analysis = srv.get(f"/api/agent/tasks/{tid_b}/resume")
    manual = srv.post(f"/api/agent/tasks/{tid_b}/resume", {})
    final_b = wait_status(srv, tid_b, TERMINAL, 1200)
    tests_b = run_pytest(ws_b)
    checks_b = {
        "not_auto_resumed": t.get("status") == "interrupted",
        "change_detected": bool(analysis.get("project_changed")) and "pricing.py" in analysis.get("changed_files", []),
        "resume_reinspects": manual.get("resume_step") in ("locate", "reproduce", "gather_context"),
        "completed_after_manual_resume": final_b.get("status") == "completed",
        "tests_pass": tests_b["exit"] == 0,
        "user_edit_preserved": "edited by the user" in p.read_text(encoding="utf-8"),
    }
    report["scenarios"]["B_project_changed_offline"] = {"checks": checks_b, "passed": all(checks_b.values()),
                                                         "analysis_reason": analysis.get("reason"),
                                                         "resume_step": manual.get("resume_step"), "final_status": final_b.get("status")}
    print("B", json.dumps(report["scenarios"]["B_project_changed_offline"], indent=1, default=str)[:1500], flush=True)

    # ---------------------------------------------------------------- C
    run = srv.post("/api/agent/run", {"request": "explain how the apply_discount function works", "workspace": str(ws_b),
                                      "allow_edits": False})
    tid_c = run["task_id"]
    time.sleep(1.0)
    srv.kill()
    con = sqlite3.connect(db)
    con.execute("UPDATE agent_tasks SET state_json='{corrupted', status='running', heartbeat_at=0 WHERE id=?", (tid_c,))
    con.commit()
    con.close()
    srv.start(auto_resume=True)
    healthy = srv.get("/api/health")["ok"]
    row_c = db_row(db, tid_c)
    others_ok = srv.get("/api/agent/tasks")["tasks"]
    checks_c = {"server_healthy": healthy is True, "corrupted_task_blocked": row_c.get("status") == "blocked",
                "task_list_still_works": len(others_ok) >= 3}
    report["scenarios"]["C_corrupted_state"] = {"checks": checks_c, "passed": all(checks_c.values())}
    print("C", json.dumps(report["scenarios"]["C_corrupted_state"], indent=1), flush=True)
    srv.kill()

    report["passed"] = all(s["passed"] for s in report["scenarios"].values())
    out = ROOT / "reports" / "agent" / "restart_backtest.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("ALL PASSED" if report["passed"] else "SOME CHECKS FAILED", "->", out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
