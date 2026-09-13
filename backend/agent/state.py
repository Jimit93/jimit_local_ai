"""Persistent task state, timeline events, and safe resume after an interruption.

The state JSON is saved after every step transition, so a crash, a closed window or a
killed server leaves a record of exactly where the task stopped. On startup:

    LOAD -> VALIDATE -> CHECK PROJECT CHANGES -> CHECK LAST CHECKPOINT
         -> CHECK LAST VALIDATION -> IDENTIFY SAFE RESUME POINT -> (auto) RESUME

A task never continues blindly: if a file it was working on changed while the app was
offline, it resumes from re-inspection, not from the next edit.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from backend.agent import checkpoints as CP
from backend.agent import policy
from backend.db import execute, query

QUEUED, RUNNING, AWAITING, INTERRUPTED = "queued", "running", "awaiting_approval", "interrupted"
COMPLETED, FAILED, BLOCKED, CANCELLED = "completed", "failed", "blocked", "cancelled"
TERMINAL = {COMPLETED, FAILED, BLOCKED, CANCELLED}
MODIFYING_KINDS = {"apply_edit", "write", "rename", "delete", "rollback", "format", "deterministic_fix"}
REQUIRED_KEYS = {"id", "workspace", "task", "status", "plan", "completed_steps", "pending_steps",
                 "failed_steps", "active_files", "decisions", "validation_results", "counters"}

_seq_lock = threading.Lock()


class StateCorrupt(ValueError):
    pass


def _now() -> float:
    return time.time()


def new_task(workspace: str, request: str, *, session_id: str | None = None, intent: str | None = None,
             topic: str | None = None, plan: list[dict] | None = None, options: dict | None = None) -> dict:
    ws = str(policy.resolve_workspace(workspace))
    red, _ = policy.redact(request)
    st = {
        "id": uuid.uuid4().hex[:12], "project": Path(ws).name, "workspace": ws, "task": red,
        "topic": topic or red[:80], "intent": intent, "session_id": session_id, "status": QUEUED,
        "current_step": None, "plan": plan or [], "completed_steps": [], "pending_steps": [],
        "failed_steps": [], "last_checkpoint": None, "checkpoints": [], "active_files": [],
        "decisions": [], "validation_results": [], "last_output": "", "next_action": "start",
        "fingerprint": {}, "error_signatures": {}, "resume_count": 0, "resume_plan": None,
        "options": options or {}, "data": {}, "predictions": [],
        "counters": {"iterations": 0, "repairs": 0, "llm_calls": 0, "tokens": 0, "tool_calls": 0,
                     "nlp_calls": 0, "ml_calls": 0, "deterministic_calls": 0, "cache_hits": 0},
        "created_at": _now(), "updated_at": _now(),
    }
    save(st)
    return st


def make_step(name: str, kind: str, *, modifies: bool | None = None, max_attempts: int = 1,
              timeout_s: float | None = None, critical: bool = True, predicted: dict | None = None) -> dict:
    return {"name": name, "kind": kind, "status": "pending", "attempts": 0, "max_attempts": max_attempts,
            "modifies": kind in MODIFYING_KINDS if modifies is None else modifies,
            "timeout_s": timeout_s, "critical": critical, "method": None, "reason": None,
            "started_at": None, "ended_at": None, "summary": "", "predicted": predicted or {},
            "actual": {}, "checkpoint_id": None, "after_hashes": {}}


def _derive(st: dict) -> None:
    st["completed_steps"] = [s["name"] for s in st["plan"] if s["status"] == "done"]
    st["pending_steps"] = [s["name"] for s in st["plan"] if s["status"] in ("pending", "running")]
    st["failed_steps"] = [s["name"] for s in st["plan"] if s["status"] == "failed"]


_ME: dict | None = None


def process_identity() -> dict:
    """pid + process start time: a pid alone can be reused by an unrelated process."""
    global _ME
    if _ME is None:
        import os

        import psutil

        _ME = {"pid": os.getpid(), "create_time": round(psutil.Process().create_time(), 3)}
    return _ME


def owner_alive(owner: dict | None) -> bool:
    if not owner:
        return False
    try:
        import psutil

        p = psutil.Process(int(owner["pid"]))
        return abs(p.create_time() - float(owner["create_time"])) < 1.0
    except Exception:
        return False


def save(st: dict, *, status: str | None = None) -> dict:
    if status:
        st["status"] = status
    if st["status"] in (RUNNING, QUEUED):
        # Several servers may share one database; each task records which process owns it.
        st["owner"] = process_identity()
    _derive(st)
    st["updated_at"] = _now()
    blob = json.dumps(st, ensure_ascii=False, default=str)
    execute(
        "INSERT INTO agent_tasks(id,session_id,workspace,request,topic,intent,status,current_step,state_json,"
        "created_at,updated_at,heartbeat_at,resume_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET status=excluded.status, current_step=excluded.current_step, "
        "state_json=excluded.state_json, intent=excluded.intent, topic=excluded.topic, "
        "updated_at=excluded.updated_at, heartbeat_at=excluded.heartbeat_at, resume_count=excluded.resume_count",
        (st["id"], st.get("session_id"), st["workspace"], st["task"], st.get("topic"), st.get("intent"),
         st["status"], st.get("current_step"), blob, st["created_at"], st["updated_at"], _now(),
         st.get("resume_count", 0)))
    return st


def heartbeat(task_id: str) -> None:
    execute("UPDATE agent_tasks SET heartbeat_at=? WHERE id=?", (_now(), task_id))


def load(task_id: str) -> dict:
    rows = query("SELECT * FROM agent_tasks WHERE id=?", (task_id,))
    if not rows:
        raise KeyError(f"unknown task {task_id}")
    row = dict(rows[0])
    try:
        st = json.loads(row["state_json"])
    except (json.JSONDecodeError, TypeError) as e:
        raise StateCorrupt(f"state JSON is unreadable: {e}") from e
    problems = validate_state(st)
    if problems:
        raise StateCorrupt("; ".join(problems))
    st["status"] = row["status"]  # the column is authoritative (written by recovery)
    st["_heartbeat_at"] = row["heartbeat_at"]
    return st


def validate_state(st) -> list[str]:
    if not isinstance(st, dict):
        return ["state is not an object"]
    problems = [f"missing key: {k}" for k in sorted(REQUIRED_KEYS - set(st))]
    if problems:
        return problems
    if not isinstance(st["plan"], list) or not all(isinstance(s, dict) and "name" in s and "status" in s
                                                   for s in st["plan"]):
        problems.append("plan is malformed")
    names = [s.get("name") for s in st["plan"]] if isinstance(st["plan"], list) else []
    if st.get("current_step") and st["current_step"] not in names:
        problems.append(f"current_step {st['current_step']!r} is not in the plan")
    if len(set(names)) != len(names):
        problems.append("duplicate step names")
    return problems


def list_tasks(status: str | None = None, limit: int = 50, session_id: str | None = None) -> list[dict]:
    sql = "SELECT id,session_id,workspace,request,topic,intent,status,current_step,created_at,updated_at," \
          "heartbeat_at,resume_count FROM agent_tasks"
    cond, params = [], []
    if status:
        cond.append("status=?")
        params.append(status)
    if session_id:
        cond.append("session_id=?")
        params.append(session_id)
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in query(sql, params)]


def set_status(task_id: str, status: str, **extra) -> None:
    try:
        st = load(task_id)
    except (KeyError, StateCorrupt):
        execute("UPDATE agent_tasks SET status=?, updated_at=? WHERE id=?", (status, _now(), task_id))
        return
    st.update(extra)
    save(st, status=status)


# ------------------------------------------------------------------ events

def event(task_id: str, kind: str, *, step: str | None = None, method: str | None = None, **payload) -> int:
    body = json.dumps(policy.redact_obj(payload), ensure_ascii=False, default=str)
    if len(body) > 20000:
        body = json.dumps({"truncated": True, "preview": body[:19000]})
    with _seq_lock:
        seq = query("SELECT COALESCE(MAX(seq),0)+1 s FROM agent_events WHERE task_id=?", (task_id,))[0]["s"]
        execute("INSERT INTO agent_events(task_id,seq,ts,kind,step,method,payload_json) VALUES(?,?,?,?,?,?,?)",
                (task_id, seq, _now(), kind, step, method, body))
    return seq


def events(task_id: str, after_seq: int = 0, limit: int = 500) -> list[dict]:
    rows = query("SELECT * FROM agent_events WHERE task_id=? AND seq>? ORDER BY seq LIMIT ?",
                 (task_id, after_seq, limit))
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d.pop("payload_json"))
        except json.JSONDecodeError:
            d["payload"] = {}
        out.append(d)
    return out


# ------------------------------------------------------------- fingerprint

def fingerprint(workspace: str, paths: list[str]) -> dict[str, str | None]:
    out = {}
    for rel in paths:
        try:
            p = policy.resolve_path(workspace, rel)
            out[rel] = hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else None
        except (OSError, policy.PolicyError):
            out[rel] = None
    return out


def refresh_fingerprint(st: dict) -> None:
    st["fingerprint"] = fingerprint(st["workspace"], st.get("active_files", []))


def add_active_files(st: dict, paths: list[str]) -> None:
    for p in paths:
        if p and p not in st["active_files"]:
            st["active_files"].append(p)
    st["active_files"] = st["active_files"][-40:]


# ------------------------------------------------------------------ resume

def analyze_resume(task_id: str) -> dict:
    """Decide whether and where a task can safely resume. Read-only."""
    report: dict = {"task_id": task_id, "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "ok": ok, "detail": detail})

    try:
        st = load(task_id)
        check("load_state", True)
        check("validate_state", True)
    except KeyError as e:
        check("load_state", False, str(e))
        return {**report, "resumable": False, "reason": "task not found"}
    except StateCorrupt as e:
        check("validate_state", False, str(e))
        return {**report, "resumable": False, "reason": f"state corrupted: {e}",
                "recommendation": "start the task again; the corrupted record is kept for inspection"}

    if st["status"] in TERMINAL:
        return {**report, "resumable": False, "reason": f"task already {st['status']}"}
    try:
        policy.resolve_workspace(st["workspace"])
        check("workspace_available", True)
    except policy.PolicyError as e:
        check("workspace_available", False, str(e))
        return {**report, "resumable": False, "reason": str(e)}

    now = fingerprint(st["workspace"], list(st.get("fingerprint", {}).keys()))
    changed = [p for p, h in st.get("fingerprint", {}).items() if now.get(p) != h]
    check("project_unchanged", not changed, ", ".join(changed[:10]))
    report["changed_files"] = changed

    cp_id = st.get("last_checkpoint")
    cp = CP.get(cp_id) if cp_id else None
    check("last_checkpoint", cp is not None or not any(s.get("modifies") for s in st["plan"]),
          cp_id or "no checkpoint needed yet")

    last_val = st["validation_results"][-1] if st["validation_results"] else None
    check("last_validation", last_val is None or bool(last_val.get("ok")),
          json.dumps(last_val)[:200] if last_val else "none recorded")

    plan = st["plan"]
    idx = next((i for i, s in enumerate(plan) if s["status"] in ("running", "pending", "failed")), None)
    if idx is None:
        return {**report, "resumable": True, "resume_step": None, "reset_steps": [],
                "reason": "all steps finished; only final validation/report remains", "rollback_checkpoint": None}

    reset_from, rollback, reason = idx, None, ""
    step = plan[idx]
    if changed:
        inspect = next((i for i, s in enumerate(plan) if s["kind"] in ("inspect", "context", "locate")), 0)
        reset_from = min(idx, inspect)
        reason = (f"{len(changed)} active file(s) changed while the task was stopped; resuming from "
                  f"'{plan[reset_from]['name']}' so the change is re-inspected before any edit")
    elif step["status"] == "running" and step.get("modifies"):
        after = step.get("after_hashes") or {}
        current = fingerprint(st["workspace"], list(after.keys())) if after else {}
        if after and current == after:
            step_done = True
            reason = f"'{step['name']}' had been applied before the interruption; continuing with verification"
            reset_from = idx + 1
        else:
            step_done = False
            rollback = step.get("checkpoint_id") if after else None
            reason = (f"'{step['name']}' was interrupted mid-change; restoring its checkpoint and re-running it"
                      if rollback else f"'{step['name']}' did not start writing; re-running it")
        report["interrupted_step_applied"] = step_done
    elif step["status"] == "running":
        reason = f"'{step['name']}' is read-only; re-running it"
    else:
        reason = f"continuing with the next pending step '{step['name']}'"

    return {**report, "resumable": True, "resume_step": plan[reset_from]["name"] if reset_from < len(plan) else None,
            "reset_steps": [s["name"] for s in plan[reset_from:]], "rollback_checkpoint": rollback,
            "reason": reason, "project_changed": bool(changed),
            "safe_to_auto_resume": not changed and st.get("resume_count", 0) < _max_auto_resumes()}


def _max_auto_resumes() -> int:
    from backend.config import SETTINGS

    return SETTINGS.agent.max_auto_resumes


def apply_resume(task_id: str, analysis: dict | None = None) -> dict:
    analysis = analysis or analyze_resume(task_id)
    if not analysis.get("resumable"):
        raise ValueError(analysis.get("reason", "not resumable"))
    st = load(task_id)
    if analysis.get("rollback_checkpoint"):
        rb = CP.rollback(analysis["rollback_checkpoint"], task_id=task_id)
        st["decisions"].append({"ts": _now(), "text": f"resume: restored checkpoint {analysis['rollback_checkpoint']}",
                                "ok": rb["ok"]})
        if not rb["ok"]:
            save(st, status=BLOCKED)
            raise ValueError(f"rollback conflicts: {rb['conflicts']}")
    reset = set(analysis.get("reset_steps") or [])
    if analysis.get("interrupted_step_applied"):
        for s in st["plan"]:
            if s["status"] == "running":
                s["status"], s["summary"] = "done", (s.get("summary") or "") + " (applied before interruption)"
    for s in st["plan"]:
        if s["name"] in reset:
            s.update(status="pending", started_at=None, ended_at=None)
    st["resume_count"] = st.get("resume_count", 0) + 1
    st["resume_plan"] = analysis
    st["current_step"] = analysis.get("resume_step")
    st["next_action"] = f"resume at {analysis.get('resume_step')}"
    st["decisions"].append({"ts": _now(), "text": f"resumed: {analysis.get('reason')}"})
    refresh_fingerprint(st)
    save(st, status=QUEUED)
    event(task_id, "resume", step=analysis.get("resume_step"), reason=analysis.get("reason"),
          checks=analysis.get("checks"))
    return st


def recover_on_startup(stale_after_s: float = 0.0) -> list[dict]:
    """Mark tasks orphaned by a dead process as interrupted and analyse each.

    A task is orphaned when the process that owned it no longer exists - not merely
    when its heartbeat is old (a fast restart would leave it "running" forever) and
    never when another live server sharing this database still owns it.
    """
    out = []
    me = process_identity()
    cutoff = _now() - stale_after_s
    for row in query("SELECT id,status,heartbeat_at,state_json FROM agent_tasks WHERE status IN (?,?)", (RUNNING, QUEUED)):
        try:
            owner = json.loads(row["state_json"]).get("owner")
        except (json.JSONDecodeError, TypeError, AttributeError):
            owner = None
        if owner and owner != me and owner_alive(owner):
            continue  # a live server owns this task
        if stale_after_s and (row["heartbeat_at"] or 0) > cutoff and not owner:
            continue
        execute("UPDATE agent_tasks SET status=?, updated_at=? WHERE id=?", (INTERRUPTED, _now(), row["id"]))
        analysis = analyze_resume(row["id"])
        try:
            st = load(row["id"])
            st["resume_plan"] = analysis
            st["next_action"] = ("resume at " + str(analysis.get("resume_step"))) if analysis.get("resumable") \
                else f"blocked: {analysis.get('reason')}"
            save(st, status=INTERRUPTED)
            event(row["id"], "interrupted", previous_status=row["status"], analysis=analysis)
        except (KeyError, StateCorrupt):
            execute("UPDATE agent_tasks SET status=? WHERE id=?", (BLOCKED, row["id"]))
        out.append(analysis)
    return out
