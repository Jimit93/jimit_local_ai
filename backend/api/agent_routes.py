"""Coding-agent HTTP API (mounted at /api/agent). Localhost only, like the rest of the app."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.agent import checkpoints as CP
from backend.agent import policy
from backend.agent import state as S
from backend.config import REPORTS_DIR

router = APIRouter(prefix="/api/agent", tags=["agent"])


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except policy.ApprovalRequired as e:
        raise HTTPException(409, f"approval required: {e.reason}")
    except policy.PolicyError as e:
        raise HTTPException(403, str(e))
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    except S.StateCorrupt as e:
        raise HTTPException(422, f"corrupted task state: {e}")
    except (ValueError, NotADirectoryError, LookupError) as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ models

class RunIn(BaseModel):
    request: str = Field(min_length=1, max_length=20000)
    workspace: str | None = None
    session_id: str | None = None
    allow_edits: bool = True
    deep_review: bool = False


class WorkspaceIn(BaseModel):
    path: str


class ApproveIn(BaseModel):
    approve: bool


class RollbackIn(BaseModel):
    force: bool = False


class MemoryIn(BaseModel):
    workspace: str | None = None
    text: str = Field(min_length=1, max_length=2000)


class ReclaimIn(BaseModel):
    required_gb: float = Field(gt=0, le=64)
    approved: list[dict] = Field(default_factory=list)


class AllowlistIn(BaseModel):
    names: list[str]


# --------------------------------------------------------------- overview

@router.get("/status")
def status() -> dict:
    from backend.agent.code_tools import CACHE_STATS
    from backend.agent.governor import governor
    from backend.agent.worker import worker
    from backend.llm import manager as LLM

    return {
        "governor": governor().status(history=120),
        "worker": worker().status(),
        "llm": {"general_loaded": LLM.is_loaded(), "general_model": (LLM.selected_model().name if LLM.selected_model() else None),
                "busy": LLM.is_busy(), "usage": LLM.USAGE, "code_usage": LLM.CODE_USAGE},
        "read_cache": CACHE_STATS,
        "workspaces": policy.workspace_roots(),
        "ts": time.time(),
    }


@router.get("/models")
def models() -> dict:
    from backend.agent.ram import ollama_loaded_models
    from backend.config import SETTINGS
    from backend.llm import manager as LLM

    return {"installed": LLM.code_models(), "loaded_in_ollama": [
        {"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 2)} for m in ollama_loaded_models()],
        "tiers": {"small": SETTINGS.agent.code_model_small, "capable": SETTINGS.agent.code_model_capable},
        "general": LLM.selected_model().to_dict() if LLM.selected_model() else None}


@router.get("/workspaces")
def workspaces() -> dict:
    return {"workspaces": policy.workspace_roots()}


@router.post("/workspaces")
def add_workspace(inp: WorkspaceIn) -> dict:
    return {"registered": _guard(policy.register_workspace, inp.path), "workspaces": policy.workspace_roots()}


# ------------------------------------------------------------------ tasks

@router.post("/run")
def run(inp: RunIn) -> dict:
    from backend.agent.service import submit_request

    return _guard(submit_request, inp.request, workspace=inp.workspace, session_id=inp.session_id,
                  options={"allow_edits": inp.allow_edits, "deep_review": inp.deep_review})


@router.get("/tasks")
def tasks(status: str | None = None, session_id: str | None = None, limit: int = 50) -> dict:
    return {"tasks": S.list_tasks(status=status, session_id=session_id, limit=min(limit, 200))}


@router.get("/tasks/{task_id}")
def task(task_id: str) -> dict:
    from backend.agent.service import task_view

    return _guard(task_view, task_id)


@router.get("/tasks/{task_id}/events")
def task_events(task_id: str, after: int = 0) -> dict:
    return {"events": S.events(task_id, after_seq=after)}


@router.post("/tasks/{task_id}/cancel")
def cancel(task_id: str) -> dict:
    from backend.agent.worker import worker

    return _guard(worker().cancel, task_id)


@router.post("/tasks/{task_id}/approve")
def approve(task_id: str, inp: ApproveIn) -> dict:
    from backend.agent.worker import worker

    return _guard(worker().approve, task_id, inp.approve)


@router.get("/tasks/{task_id}/resume")
def resume_analysis(task_id: str) -> dict:
    return S.analyze_resume(task_id)


@router.post("/tasks/{task_id}/resume")
def resume(task_id: str) -> dict:
    from backend.agent.worker import worker

    return _guard(worker().resume, task_id)


@router.get("/sessions")
def sessions() -> dict:
    from backend.agent.memory import sessions as list_sessions

    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
def session(session_id: str) -> dict:
    from backend.agent.memory import session_recent

    return {"messages": session_recent(session_id, 200)}


# ------------------------------------------------------------------ files

@router.get("/files/tree")
def files_tree(workspace: str | None = None) -> dict:
    from backend.agent.code_tools import tree

    ws = str(_guard(policy.resolve_workspace, workspace))
    return {"workspace": ws, **tree(ws)}


@router.get("/files/read")
def files_read(path: str, workspace: str | None = None, start: int | None = None, end: int | None = None) -> dict:
    from backend.agent.code_tools import read_file

    return _guard(read_file, workspace, path, start_line=start, end_line=end)


@router.get("/files/activity")
def files_activity(workspace: str | None = None) -> dict:
    """Recent, relevant (last task's active files) and modified (git) files."""
    from backend.agent.code_tools import git_status, is_git_repo, recent_files

    ws = str(_guard(policy.resolve_workspace, workspace))
    last = next((t for t in S.list_tasks(limit=20) if policy._norm(t["workspace"]) == policy._norm(ws)), None)
    relevant = []
    if last:
        try:
            st = S.load(last["id"])
            relevant = [{"path": p, "task_id": last["id"]} for p in st.get("active_files", [])]
            relevant += [{"path": f["path"], "score": f.get("score"), "signals": f.get("signals")}
                         for f in (st["data"].get("ranking") or {}).get("files", []) if f["path"] not in st.get("active_files", [])]
        except (KeyError, S.StateCorrupt):
            pass
    modified = git_status(ws)["files"] if is_git_repo(ws) else []
    return {"recent": recent_files(ws), "relevant": relevant[:20], "modified": modified[:200]}


@router.get("/git/diff")
def git_diff(workspace: str | None = None, path: str | None = None) -> dict:
    from backend.agent.code_tools import git_diff as gd

    return _guard(gd, str(_guard(policy.resolve_workspace, workspace)), path)


# ----------------------------------------------------- memory + checkpoints

@router.get("/memory")
def memory(workspace: str | None = None, q: str | None = None) -> dict:
    from backend.agent import memory as MEM

    ws = str(_guard(policy.resolve_workspace, workspace))
    out = {"project": MEM.project_memory(ws), "sessions": MEM.sessions(20),
           "unfinished": [t for t in S.list_tasks(limit=100) if t["status"] not in S.TERMINAL]}
    if not out["project"].get("summary"):
        out["project"] = {**MEM.learn_project(ws), **MEM.project_memory(ws)}
    if q:
        out["search"] = MEM.search(q, workspace=ws)
    return out


@router.post("/memory/decision")
def add_decision(inp: MemoryIn) -> dict:
    from backend.agent import memory as MEM

    ws = str(_guard(policy.resolve_workspace, inp.workspace))
    MEM.record_decision(ws, inp.text, source="user")
    return {"ok": True}


@router.get("/checkpoints")
def checkpoints(task_id: str | None = None, workspace: str | None = None) -> dict:
    cps = CP.list_checkpoints(task_id=task_id, limit=100)
    if workspace:
        ws = str(_guard(policy.resolve_workspace, workspace))
        cps = [c for c in cps if policy._norm(c["workspace"]) == policy._norm(ws)]
    return {"checkpoints": cps}


@router.get("/checkpoints/{cp_id}/diff")
def checkpoint_diff(cp_id: str) -> dict:
    return _guard(CP.diff_since, cp_id)


@router.post("/checkpoints/{cp_id}/rollback")
def checkpoint_rollback(cp_id: str, inp: RollbackIn) -> dict:
    return _guard(CP.rollback, cp_id, force=inp.force)


@router.get("/audit")
def audit(limit: int = 100) -> dict:
    return {"audit": policy.recent_audit(min(limit, 500))}


# -------------------------------------------------------------------- RAM

@router.get("/ram")
def ram() -> dict:
    from backend.agent import ram as RAM

    procs = RAM.classify_processes()
    return {"available_gb": RAM.available_gb(), "allowlist": sorted(RAM._allowlist()),
            "processes": procs[:40], "ollama_loaded": RAM.ollama_loaded_models()}


@router.post("/ram/reclaim")
def ram_reclaim(inp: ReclaimIn) -> dict:
    from backend.agent import ram as RAM

    return RAM.reclaim(inp.required_gb, approved=inp.approved)


class TerminateIn(BaseModel):
    pid: int
    create_time: float | None = None
    name: str | None = None


@router.post("/ram/terminate")
def ram_terminate(inp: TerminateIn) -> dict:
    """One explicitly approved process; no model unloading, no allowlist sweep."""
    from backend.agent import ram as RAM

    return RAM.terminate_approved(inp.pid, inp.create_time, inp.name)


@router.post("/ram/allowlist")
def ram_allowlist(inp: AllowlistIn) -> dict:
    from backend.agent import ram as RAM

    return {"allowlist": RAM.set_allowlist(inp.names)}


# -------------------------------------------------------------- evaluation

@router.get("/evaluation")
def evaluation(limit: int = 200) -> dict:
    from backend.benchmarks.agent_eval import aggregate_tasks

    out = {"tasks": aggregate_tasks(limit=limit), "reports": {}}
    d = Path(REPORTS_DIR) / "agent"
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                out["reports"][f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return out


@router.post("/index")
def build_index(inp: WorkspaceIn) -> dict:
    from backend.agent.project_index import build

    ws = str(_guard(policy.resolve_workspace, inp.path))
    return build(ws)
