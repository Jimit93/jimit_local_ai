"""File checkpoints: snapshot before modifying, restore exactly on rollback.

A checkpoint records, for every file the agent is about to touch, whether it existed
and its exact bytes. Rollback writes those bytes back (or removes a file the agent
created). It refuses to overwrite a file someone else changed after the agent's
last write unless forced - the user's work is never silently clobbered.

No git stash/reset is used: those would touch changes the agent does not own.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from backend.agent import policy
from backend.config import DATA_DIR, SETTINGS
from backend.db import execute, query

MAX_SNAPSHOT_BYTES = 50_000_000  # per checkpoint
TRASH_DIR = DATA_DIR / "trash"


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _load(cp_id: str) -> dict | None:
    rows = query("SELECT * FROM agent_checkpoints WHERE id=?", (cp_id,))
    if not rows:
        return None
    d = dict(rows[0])
    d["files"] = json.loads(d.pop("files_json") or "[]")
    d["state"] = json.loads(d.pop("state_json") or "{}")
    return d


def _save_files(cp_id: str, files: list[dict]) -> None:
    execute("UPDATE agent_checkpoints SET files_json=? WHERE id=?", (json.dumps(files), cp_id))


def create(workspace: str, label: str, *, task_id: str | None = None, step: str | None = None,
           paths: list[str] | None = None, state: dict | None = None) -> str:
    ws = str(policy.resolve_workspace(workspace))
    cp_id = uuid.uuid4().hex[:12]
    execute(
        "INSERT INTO agent_checkpoints(id,task_id,ts,label,step,workspace,files_json,state_json) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (cp_id, task_id, time.time(), label[:200], step, ws, "[]", json.dumps(state or {})),
    )
    for p in paths or []:
        ensure_file(cp_id, p)
    return cp_id


def ensure_file(cp_id: str, rel: str) -> dict:
    """Snapshot `rel` into the checkpoint unless it is already captured."""
    cp = _load(cp_id)
    if cp is None:
        raise ValueError(f"unknown checkpoint {cp_id}")
    target = policy.resolve_path(cp["workspace"], rel, for_write=True)
    relp = target.relative_to(Path(cp["workspace"])).as_posix()
    for f in cp["files"]:
        if f["path"] == relp:
            return f
    entry = {"path": relp, "existed": target.is_file(), "sha1": None, "size": 0,
             "content_b64": None, "after_sha1": None}
    if entry["existed"]:
        data = target.read_bytes()
        if len(data) > SETTINGS.agent.max_edit_bytes:
            raise policy.PolicyError(f"file too large to checkpoint safely: {relp}")
        used = sum(f.get("size", 0) for f in cp["files"])
        if used + len(data) > MAX_SNAPSHOT_BYTES:
            raise policy.PolicyError("checkpoint size limit reached; start a new checkpoint")
        entry.update(sha1=sha1_bytes(data), size=len(data),
                     content_b64=base64.b64encode(data).decode("ascii"))
    cp["files"].append(entry)
    _save_files(cp_id, cp["files"])
    return entry


def record_after(cp_id: str, rel: str, after_sha1: str | None) -> None:
    """Remember what the agent wrote, so rollback can detect later external edits."""
    cp = _load(cp_id)
    if cp is None:
        return
    for f in cp["files"]:
        if f["path"] == rel:
            f["after_sha1"] = after_sha1
    _save_files(cp_id, cp["files"])


def get(cp_id: str, *, with_content: bool = False) -> dict | None:
    cp = _load(cp_id)
    if cp and not with_content:
        for f in cp["files"]:
            f.pop("content_b64", None)
    return cp


def list_checkpoints(task_id: str | None = None, limit: int = 50) -> list[dict]:
    if task_id:
        rows = query("SELECT * FROM agent_checkpoints WHERE task_id=? ORDER BY ts DESC LIMIT ?",
                     (task_id, limit))
    else:
        rows = query("SELECT * FROM agent_checkpoints ORDER BY ts DESC LIMIT ?", (limit,))
    out = []
    for r in rows:
        d = dict(r)
        files = json.loads(d.pop("files_json") or "[]")
        d.pop("state_json", None)
        d["files"] = [{"path": f["path"], "existed": f["existed"], "size": f["size"],
                       "modified": f.get("after_sha1") is not None} for f in files]
        out.append(d)
    return out


def _file_sha1(p: Path) -> str | None:
    return sha1_bytes(p.read_bytes()) if p.is_file() else None


def rollback(cp_id: str, *, force: bool = False, task_id: str | None = None) -> dict:
    cp = _load(cp_id)
    if cp is None:
        raise ValueError(f"unknown checkpoint {cp_id}")
    policy.enforce("rollback", cp_id, task_id=task_id)
    ws = Path(cp["workspace"])
    actions, conflicts = [], []
    for f in cp["files"]:
        target = policy.resolve_path(ws, f["path"], for_write=True)
        current = _file_sha1(target)
        expected_now = f.get("after_sha1")
        if expected_now is not None and current != expected_now and current != f["sha1"] and not force:
            conflicts.append({"path": f["path"], "reason": "changed outside the agent after its edit"})
            continue
        if f["existed"]:
            if current == f["sha1"]:
                actions.append({"path": f["path"], "action": "unchanged"})
                continue
            data = base64.b64decode(f["content_b64"])
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".jla_tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            actions.append({"path": f["path"], "action": "restored"})
        else:
            if current is None:
                actions.append({"path": f["path"], "action": "unchanged"})
                continue
            dest = move_to_trash(ws, target)
            actions.append({"path": f["path"], "action": "removed_created_file", "trash": str(dest)})
    if not conflicts:
        execute("UPDATE agent_checkpoints SET restored_at=? WHERE id=?", (time.time(), cp_id))
    return {"checkpoint": cp_id, "actions": actions, "conflicts": conflicts,
            "ok": not conflicts}


def move_to_trash(workspace: Path, target: Path) -> Path:
    rel = target.relative_to(workspace)
    dest = TRASH_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:6]}" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(dest))
    return dest


def diff_since(cp_id: str, max_chars: int = 20000) -> dict:
    """Unified diff of every checkpointed file: snapshot vs current content."""
    import difflib

    cp = _load(cp_id)
    if cp is None:
        raise ValueError(f"unknown checkpoint {cp_id}")
    ws = Path(cp["workspace"])
    chunks, changed = [], []
    for f in cp["files"]:
        target = ws / f["path"]
        before = base64.b64decode(f["content_b64"]).decode("utf-8", "replace") if f["existed"] else ""
        after = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        if before == after:
            continue
        changed.append(f["path"])
        chunks.append("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile=f"a/{f['path']}", tofile=f"b/{f['path']}", n=2)))
    text, _ = policy.redact("\n".join(chunks))
    return {"changed_files": changed, "diff": text[:max_chars], "truncated": len(text) > max_chars}
