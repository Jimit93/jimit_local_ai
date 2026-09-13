"""Layered memory for the coding agent, on the existing `agent_memory` table.

    session     conversation turns (agent_messages), used to resolve follow-ups
    project     derived facts per workspace: languages, manifests, test runner, conventions,
                important files, known issues, decisions
    task        the resumable state in agent_tasks (see state.py)
    persistent  facts / decisions / error patterns that outlive a task

Rules: only derived, compact facts are stored (never raw file contents); everything is
redacted before it is written; generated content is stored with source='llm' and is
never trusted as fact.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from backend.agent import policy
from backend.db import execute, query, remember

KINDS = ("project", "decision", "error_pattern", "fact", "preference")


def ws_key(workspace: str) -> str:
    return hashlib.sha1(str(policy.resolve_workspace(workspace)).lower().encode()).hexdigest()[:10]


def _put(kind: str, key: str, value, source: str, confidence: float = 1.0) -> None:
    value = policy.redact_obj(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    remember(kind, key, text[:4000], source=source, confidence=confidence)


def _rows(kind: str, prefix: str | None = None, limit: int = 200) -> list[dict]:
    if prefix:
        rows = query("SELECT * FROM agent_memory WHERE kind=? AND key LIKE ? ORDER BY created_at DESC LIMIT ?",
                     (kind, prefix + "%", limit))
    else:
        rows = query("SELECT * FROM agent_memory WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit))
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["value"] = json.loads(d["value"])
        except (json.JSONDecodeError, TypeError):
            pass
        out.append(d)
    return out


# ----------------------------------------------------------------- session

def session_add(session_id: str, role: str, text: str, task_id: str | None = None) -> None:
    red, _ = policy.redact((text or "")[:4000])
    execute("INSERT INTO agent_messages(session_id,ts,role,text,task_id) VALUES(?,?,?,?,?)",
            (session_id, time.time(), role, red, task_id))


def session_recent(session_id: str, n: int = 20) -> list[dict]:
    rows = query("SELECT * FROM agent_messages WHERE session_id=? ORDER BY ts DESC LIMIT ?", (session_id, n))
    return [dict(r) for r in reversed(rows)]


def sessions(limit: int = 30) -> list[dict]:
    rows = query("SELECT session_id, MIN(ts) started, MAX(ts) last, COUNT(*) n, "
                 "(SELECT text FROM agent_messages m2 WHERE m2.session_id=m.session_id AND role='user' "
                 " ORDER BY ts LIMIT 1) first_request FROM agent_messages m GROUP BY session_id "
                 "ORDER BY last DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


_FOLLOWUP = [
    (re.compile(r"^\s*(?:undo|revert|roll ?back)\b(?: (?:that|it|the (?:last )?(?:change|edit)s?))?\s*[.!]?\s*$", re.I), "rollback_last"),
    (re.compile(r"^\s*(?:again|retry|try again|do it again|repeat(?: that)?)\s*[.!]?\s*$", re.I), "repeat_last"),
    (re.compile(r"^\s*(?:continue|resume|keep going|carry on)\b", re.I), "resume_last"),
]


def resolve_followup(session_id: str, request: str) -> dict | None:
    """Deterministic follow-up resolution ('undo that', 'again', 'continue')."""
    for rx, action in _FOLLOWUP:
        if rx.search(request):
            last_task = next((m for m in reversed(session_recent(session_id, 30)) if m.get("task_id")), None)
            last_user = next((m for m in reversed(session_recent(session_id, 30))
                              if m["role"] == "user" and not any(r.search(m["text"]) for r, _ in _FOLLOWUP)), None)
            return {"action": action, "task_id": last_task["task_id"] if last_task else None,
                    "previous_request": last_user["text"] if last_user else None}
    return None


# ----------------------------------------------------------------- project

def learn_project(workspace: str) -> dict:
    """Derive project facts deterministically and store them. Cheap to re-run."""
    from backend.agent.code_tools import workspace_summary
    from backend.agent.project_index import import_graph

    ws = str(policy.resolve_workspace(workspace))
    k = ws_key(ws)
    summary = workspace_summary(ws)
    facts = {"summary": summary}
    if summary["test_runner"] == "pytest":
        facts["test_command"] = "python -m pytest"
    elif summary["test_runner"] == "npm":
        facts["test_command"] = "npm test"
    try:
        g = import_graph(ws)
        central = sorted(((len(v), p) for p, v in g.items() if v), reverse=True)[:10]
        facts["important_files"] = [p for _, p in central]
    except Exception:
        facts["important_files"] = []
    conv = _conventions(ws)
    if conv:
        facts["conventions"] = conv
    for name, val in facts.items():
        _put("project", f"{k}:{name}", val, source="derived")
    _put("project", f"{k}:workspace", ws, source="derived")
    return facts


def _conventions(ws: str) -> list[str]:
    """Headline rules from CLAUDE.md / CONTRIBUTING / README - extracted, not summarised."""
    from pathlib import Path

    out: list[str] = []
    for name in ("CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md"):
        p = Path(ws) / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")[:20000]
        for m in re.finditer(r"^\s*(?:\d+\.|-)\s+\*\*(.+?)\*\*", text, re.M):
            out.append(m.group(1).strip()[:160])
        for m in re.finditer(r"^(?:Run|Never|Always|Do not|Prefer)\b.{10,160}$", text, re.M):
            out.append(m.group(0).strip())
    return list(dict.fromkeys(out))[:20]


def project_memory(workspace: str) -> dict:
    k = ws_key(workspace)
    out = {}
    for r in _rows("project", prefix=f"{k}:"):
        out[r["key"].split(":", 1)[1]] = r["value"]
    out["decisions"] = [r["value"] for r in _rows("decision", prefix=f"{k}:", limit=30)]
    out["known_issues"] = [r["value"] for r in _rows("error_pattern", prefix=f"{k}:", limit=30)]
    return out


def record_decision(workspace: str, text: str, *, task_id: str | None = None, source: str = "derived") -> None:
    k = ws_key(workspace)
    _put("decision", f"{k}:{time.time():.6f}", {"text": text, "task_id": task_id, "ts": time.time()}, source=source)


def error_signature(text: str) -> str:
    """Stable fingerprint of an error: exception type + message with numbers/paths removed."""
    t = re.sub(r"(?:[A-Za-z]:)?[\\/][^\s\"':]+", "<path>", text or "")
    t = re.sub(r"0x[0-9a-f]+|\d+", "N", t)
    m = re.findall(r"\b(\w+(?:Error|Exception|Failure))\b:?\s*([^\n]{0,120})", t)
    core = " | ".join(f"{a}: {b.strip()}" for a, b in m[-2:]) if m else t.strip()[-160:]
    return hashlib.sha1(core.encode()).hexdigest()[:12] + " " + core[:160]


def record_error(workspace: str, error_text: str, *, resolution: str | None = None,
                 task_id: str | None = None) -> str:
    k = ws_key(workspace)
    sig = error_signature(error_text)
    existing = [r for r in _rows("error_pattern", prefix=f"{k}:{sig[:12]}")]
    count = (existing[0]["value"].get("count", 0) if existing and isinstance(existing[0]["value"], dict) else 0) + 1
    _put("error_pattern", f"{k}:{sig[:12]}", {"signature": sig, "count": count, "resolution": resolution,
                                              "last_task": task_id, "ts": time.time()}, source="derived")
    return sig


def known_resolution(workspace: str, error_text: str) -> dict | None:
    k = ws_key(workspace)
    sig = error_signature(error_text)
    rows = _rows("error_pattern", prefix=f"{k}:{sig[:12]}")
    if rows and isinstance(rows[0]["value"], dict) and rows[0]["value"].get("resolution"):
        return rows[0]["value"]
    return None


# ---------------------------------------------------------------- search

def search(q: str, *, workspace: str | None = None, limit: int = 8) -> list[dict]:
    """BM25 over memory values (no model). Scoped to a workspace when given."""
    from backend.agent.project_index import code_tokens
    from backend.nlp.keywords import BM25Index

    prefix = f"{ws_key(workspace)}:" if workspace else None
    rows = []
    for kind in KINDS:
        rows += _rows(kind, prefix=prefix if kind != "preference" else None, limit=300)
    if not rows:
        return []
    docs = [f"{r['kind']} {r['key']} {json.dumps(r['value'], ensure_ascii=False) if not isinstance(r['value'], str) else r['value']}"
            for r in rows]
    idx = BM25Index(docs, tokenizer=code_tokens)
    return [{**rows[i], "score": round(s, 3)} for i, s in idx.normalised_search(q, top_k=limit)]


def memory_items_for(workspace: str, request: str, limit: int = 8) -> list[str]:
    """Compact strings for the context package."""
    items = []
    pm = project_memory(workspace)
    if pm.get("test_command"):
        items.append(f"Test command: {pm['test_command']}")
    for c in (pm.get("conventions") or [])[:5]:
        items.append(f"Convention: {c}")
    for hit in search(request, workspace=workspace, limit=limit):
        v = hit["value"]
        if hit["kind"] == "decision" and isinstance(v, dict):
            items.append(f"Earlier decision: {v.get('text')}")
        elif hit["kind"] == "error_pattern" and isinstance(v, dict) and v.get("resolution"):
            items.append(f"Known error: {v.get('signature', '')[13:90]} -> fixed by: {v['resolution']}")
    return list(dict.fromkeys(items))[:limit]
