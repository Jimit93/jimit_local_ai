"""Workspace tools for the coding agent. Deterministic; no model anywhere.

Every function takes the workspace root first, goes through `policy`, and returns a
plain dict. Modifying tools snapshot into a checkpoint before they write, so every
change is recoverable. Reads are cached by (mtime, size) so unchanged files are never
re-read - the cache hit counter is reported as a token-efficiency metric.
"""
from __future__ import annotations

import difflib
import fnmatch
import hashlib
import io
import json
import os
import re
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

from backend.agent import checkpoints as CP
from backend.agent import policy
from backend.agent.proc import run_process
from backend.agent.symbols import extract as extract_symbols
from backend.agent.symbols import language_of
from backend.config import SETTINGS

MAX_SEARCH_FILE_BYTES = 1_000_000

# ------------------------------------------------------------------ bookkeeping

_cache_lock = threading.Lock()
_READ_CACHE: "OrderedDict[str, tuple[int, int, str, str]]" = OrderedDict()
_CACHE_MAX = 256
CACHE_STATS = {"hits": 0, "misses": 0, "chars_saved": 0}
_RECENT: dict[str, deque] = {}


def _touch(ws: Path, rel: str, action: str) -> None:
    dq = _RECENT.setdefault(str(ws), deque(maxlen=60))
    dq.appendleft({"path": rel, "action": action, "ts": time.time()})


def recent_files(workspace: str, limit: int = 30) -> list[dict]:
    ws = policy.resolve_workspace(workspace)
    seen, out = set(), []
    for e in _RECENT.get(str(ws), []):
        if e["path"] not in seen:
            seen.add(e["path"])
            out.append(e)
        if len(out) >= limit:
            break
    return out


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _read_text(p: Path) -> tuple[str, str]:
    """Cached UTF-8 read keyed by mtime_ns + size. Returns (text, sha1)."""
    st = p.stat()
    key = os.path.normcase(str(p))
    with _cache_lock:
        hit = _READ_CACHE.get(key)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            _READ_CACHE.move_to_end(key)
            CACHE_STATS["hits"] += 1
            CACHE_STATS["chars_saved"] += len(hit[2])
            return hit[2], hit[3]
    data = p.read_bytes()
    if b"\x00" in data[:4096]:
        raise ValueError(f"binary file: {p.name}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"file is not valid UTF-8: {p.name}")
    sha = hashlib.sha1(data).hexdigest()
    with _cache_lock:
        CACHE_STATS["misses"] += 1
        _READ_CACHE[key] = (st.st_mtime_ns, st.st_size, text, sha)
        if len(_READ_CACHE) > _CACHE_MAX:
            _READ_CACHE.popitem(last=False)
    return text, sha


def _atomic_write(p: Path, text: str) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    tmp = p.with_name(p.name + ".jla_tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    with _cache_lock:
        _READ_CACHE.pop(os.path.normcase(str(p)), None)
    return hashlib.sha1(data).hexdigest()


# ---------------------------------------------------------------- file walking

def _gitignore_patterns(ws: Path) -> list[str]:
    f = ws / ".gitignore"
    if not f.is_file():
        return []
    pats = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith(("#", "!")):
            pats.append(s)
    return pats


def _ignored(rel: str, is_dir: bool, pats: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for pat in pats:
        dir_only = pat.endswith("/")
        pp = pat.rstrip("/").lstrip("/")
        if dir_only and not is_dir:
            # a file under an ignored directory is handled when the directory is pruned
            continue
        if fnmatch.fnmatch(name, pp) or fnmatch.fnmatch(rel, pp):
            return True
    return False


def iter_files(workspace: str | Path, *, max_files: int = 50000, respect_gitignore: bool = True):
    """Yield (relative posix path, Path) for workspace files, pruning heavy/ignored dirs."""
    ws = policy.resolve_workspace(workspace)
    pats = _gitignore_patterns(ws) if respect_gitignore else []
    stack = [ws]
    n = 0
    while stack:
        d = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            rel = Path(e.path).relative_to(ws).as_posix()
            try:
                is_dir = e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if e.name in policy.IGNORED_DIRS or _ignored(rel, True, pats):
                    continue
                stack.append(Path(e.path))
            elif e.is_file(follow_symlinks=False):
                if _ignored(rel, False, pats) or e.name.endswith(".jla_tmp"):
                    continue
                yield rel, Path(e.path)
                n += 1
                if n >= max_files:
                    return


# ------------------------------------------------------------------ read tools

def file_exists(workspace: str, path: str) -> dict:
    p = policy.resolve_path(workspace, path)
    return {"path": path, "exists": p.exists(), "is_file": p.is_file(), "is_dir": p.is_dir()}


def file_info(workspace: str, path: str) -> dict:
    p = policy.resolve_path(workspace, path)
    if not p.exists():
        raise FileNotFoundError(path)
    st = p.stat()
    out = {"path": path, "is_dir": p.is_dir(), "size_bytes": st.st_size,
           "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
           "language": language_of(p.name), "secret_file": policy.is_secret_file(p)}
    if p.is_file() and not out["secret_file"] and st.st_size <= MAX_SEARCH_FILE_BYTES:
        try:
            text, sha = _read_text(p)
            out.update(lines=text.count("\n") + (0 if text.endswith("\n") or not text else 1), sha1=sha)
        except ValueError as e:
            out["note"] = str(e)
    return out


def list_directory(workspace: str, path: str = ".", *, limit: int = 500) -> dict:
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path)
    if not p.is_dir():
        raise NotADirectoryError(path)
    entries = []
    for e in sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name.lower())):
        if len(entries) >= limit:
            break
        try:
            is_dir = e.is_dir(follow_symlinks=False)
            entries.append({"name": e.name, "is_dir": is_dir,
                            "size_bytes": 0 if is_dir else e.stat().st_size,
                            "ignored": e.name in policy.IGNORED_DIRS})
        except OSError:
            continue
    _touch(ws, Path(p).relative_to(ws).as_posix() or ".", "list")
    return {"path": path, "count": len(entries), "entries": entries}


def tree(workspace: str, *, max_entries: int = 3000) -> dict:
    """Nested tree for the file explorer (ignored dirs pruned)."""
    root: dict = {"name": ".", "path": "", "dirs": {}, "files": []}
    n = 0
    for rel, p in iter_files(workspace, max_files=max_entries):
        parts = rel.split("/")
        node = root
        for i, part in enumerate(parts[:-1]):
            node = node["dirs"].setdefault(part, {"name": part, "path": "/".join(parts[: i + 1]),
                                                  "dirs": {}, "files": []})
        node["files"].append({"name": parts[-1], "path": rel})
        n += 1

    def _fmt(node: dict) -> dict:
        return {"name": node["name"], "path": node["path"],
                "children": [_fmt(c) for c in sorted(node["dirs"].values(), key=lambda c: c["name"].lower())]
                + [{"name": f["name"], "path": f["path"], "file": True}
                   for f in sorted(node["files"], key=lambda f: f["name"].lower())]}

    return {"root": _fmt(root), "files": n, "truncated": n >= max_entries}


def read_file(workspace: str, path: str, *, start_line: int | None = None,
              end_line: int | None = None, max_chars: int | None = None) -> dict:
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path)
    if policy.is_secret_file(p):
        policy.audit("read", path, "deny", policy.RISK_HIGH, "secret file")
        raise policy.PolicyError(f"secret file is never read by the agent: {path}")
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.stat().st_size > SETTINGS.agent.max_edit_bytes * 5:
        raise ValueError(f"file too large to read ({p.stat().st_size} bytes)")
    hits_before = CACHE_STATS["hits"]
    text, sha = _read_text(p)
    lines = text.splitlines()
    s = max(1, start_line or 1)
    e = min(len(lines), end_line or len(lines))
    body = "\n".join(lines[s - 1:e])
    limit = max_chars or SETTINGS.agent.max_read_chars
    truncated = len(body) > limit
    body, redactions = policy.redact(body[:limit])
    _touch(ws, Path(p).relative_to(ws).as_posix(), "read")
    return {"path": Path(p).relative_to(ws).as_posix(), "content": body, "start_line": s,
            "end_line": e, "total_lines": len(lines), "sha1": sha, "truncated": truncated,
            "redactions": redactions, "cached": CACHE_STATS["hits"] > hits_before}


def search_files(workspace: str, pattern: str, *, limit: int = 200) -> dict:
    """Glob (`**/*.py`, `*test*`) or plain substring match on the path."""
    has_glob = any(ch in pattern for ch in "*?[")
    pat = pattern.lower()
    out = []
    for rel, _p in iter_files(workspace):
        low = rel.lower()
        name = low.rsplit("/", 1)[-1]
        if (has_glob and (fnmatch.fnmatch(low, pat) or fnmatch.fnmatch(name, pat))) or \
                (not has_glob and pat in low):
            out.append(rel)
            if len(out) >= limit:
                break
    return {"pattern": pattern, "count": len(out), "files": out, "truncated": len(out) >= limit}


def search_code(workspace: str, query: str, *, regex: bool = False, glob: str | None = None,
                case_sensitive: bool = False, limit: int = 200, context: int = 0) -> dict:
    if not query:
        raise ValueError("empty query")
    flags = 0 if case_sensitive else re.I
    rx = re.compile(query if regex else re.escape(query), flags)
    matches, files_scanned, files_matched = [], 0, 0
    for rel, p in iter_files(workspace):
        if glob and not fnmatch.fnmatch(rel.lower(), glob.lower()) and \
                not fnmatch.fnmatch(rel.rsplit("/", 1)[-1].lower(), glob.lower()):
            continue
        if policy.is_secret_file(p):
            continue
        try:
            if p.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text, _ = _read_text(p)
        except (OSError, ValueError):
            continue
        files_scanned += 1
        if not rx.search(text):
            continue
        files_matched += 1
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if rx.search(line):
                snippet = line.strip()[:300]
                item = {"path": rel, "line": i + 1, "text": policy.redact(snippet)[0]}
                if context:
                    lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                    item["context"] = policy.redact("\n".join(lines[lo:hi]))[0]
                matches.append(item)
                if len(matches) >= limit:
                    return {"query": query, "count": len(matches), "matches": matches,
                            "files_scanned": files_scanned, "files_matched": files_matched,
                            "truncated": True}
    return {"query": query, "count": len(matches), "matches": matches,
            "files_scanned": files_scanned, "files_matched": files_matched, "truncated": False}


def file_symbols(workspace: str, path: str) -> dict:
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path)
    text, _ = _read_text(p)
    info = extract_symbols(p.name, text)
    return {"path": path, "lang": info["lang"], "syntax_error": info["syntax_error"],
            "symbols": [s.to_dict() for s in info["symbols"]],
            "imports": [{"module": m, "line": ln} for m, ln in info["imports"]]}


def search_symbols(workspace: str, name: str, *, kind: str | None = None, exact: bool = False,
                   limit: int = 100) -> dict:
    """Uses the incremental project index when present; falls back to a live scan."""
    try:
        from backend.agent.project_index import find_symbols

        return find_symbols(workspace, name, kind=kind, exact=exact, limit=limit)
    except ImportError:
        pass
    low = name.lower()
    out = []
    for rel, p in iter_files(workspace):
        if language_of(rel) not in ("python", "typescript", "javascript"):
            continue
        try:
            text, _ = _read_text(p)
        except (OSError, ValueError):
            continue
        for s in extract_symbols(rel, text)["symbols"]:
            hit = s.name.lower() == low if exact else low in s.name.lower()
            if hit and (kind is None or s.kind == kind):
                out.append({"path": rel, **s.to_dict()})
                if len(out) >= limit:
                    return {"name": name, "count": len(out), "symbols": out, "source": "scan"}
    return {"name": name, "count": len(out), "symbols": out, "source": "scan"}


# ----------------------------------------------------------------- write tools

def _checkpoint_for(ws: Path, rel: str, checkpoint_id: str | None, task_id: str | None,
                    label: str) -> str:
    cp = checkpoint_id or CP.create(str(ws), label, task_id=task_id)
    CP.ensure_file(cp, rel)
    return cp


def _unified(before: str, after: str, rel: str, max_chars: int = 6000) -> str:
    d = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                     fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2))
    return policy.redact(d[:max_chars])[0]


def write_file(workspace: str, path: str, content: str, *, overwrite: bool = True,
               checkpoint_id: str | None = None, task_id: str | None = None) -> dict:
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path, for_write=True)
    if policy.is_secret_file(p):
        raise policy.PolicyError(f"the agent never writes secret files: {path}")
    if len(content.encode("utf-8")) > SETTINGS.agent.max_edit_bytes:
        raise ValueError("content exceeds the edit size limit")
    existed = p.is_file()
    if existed and not overwrite:
        raise FileExistsError(path)
    rel = p.relative_to(ws).as_posix()
    policy.enforce("write" if existed else "create", rel, task_id=task_id)
    before = _read_text(p)[0] if existed else ""
    cp = _checkpoint_for(ws, rel, checkpoint_id, task_id, f"before write {rel}")
    sha = _atomic_write(p, content)
    CP.record_after(cp, rel, sha)
    _touch(ws, rel, "write")
    return {"path": rel, "created": not existed, "bytes": len(content.encode("utf-8")),
            "sha1": sha, "checkpoint_id": cp, "diff": _unified(before, content, rel),
            "syntax": _quick_syntax(rel, content)}


def _closest_lines(text: str, needle: str, n: int = 3) -> list[str]:
    first = next((ln for ln in needle.splitlines() if ln.strip()), needle).strip()
    cands = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return difflib.get_close_matches(first, cands, n=n, cutoff=0.5)


def edit_file(workspace: str, path: str, old: str, new: str, *, replace_all: bool = False,
              checkpoint_id: str | None = None, task_id: str | None = None) -> dict:
    """Exact string replacement. The match must be unique unless `replace_all`."""
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path, for_write=True)
    if policy.is_secret_file(p):
        raise policy.PolicyError(f"the agent never edits secret files: {path}")
    if not p.is_file():
        raise FileNotFoundError(path)
    if old == new:
        raise ValueError("old and new text are identical")
    if not old:
        raise ValueError("old text is empty; use write_file to create content")
    text, _ = _read_text(p)
    crlf = "\r\n" in text
    o, n_ = old, new
    if crlf and "\r\n" not in old:
        o, n_ = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    count = text.count(o)
    if count == 0:
        hint = _closest_lines(text, old)
        raise LookupError("old text not found in file"
                          + (f"; closest lines: {hint}" if hint else ""))
    if count > 1 and not replace_all:
        raise LookupError(f"old text matches {count} times; include more surrounding context")
    rel = p.relative_to(ws).as_posix()
    policy.enforce("edit", rel, task_id=task_id)
    after = text.replace(o, n_) if replace_all else text.replace(o, n_, 1)
    cp = _checkpoint_for(ws, rel, checkpoint_id, task_id, f"before edit {rel}")
    sha = _atomic_write(p, after)
    CP.record_after(cp, rel, sha)
    _touch(ws, rel, "edit")
    return {"path": rel, "replacements": count if replace_all else 1, "sha1": sha,
            "checkpoint_id": cp, "diff": _unified(text, after, rel),
            "syntax": _quick_syntax(rel, after)}


def rename_file(workspace: str, src: str, dst: str, *, overwrite: bool = False,
                checkpoint_id: str | None = None, task_id: str | None = None) -> dict:
    ws = policy.resolve_workspace(workspace)
    ps = policy.resolve_path(ws, src, for_write=True)
    pd = policy.resolve_path(ws, dst, for_write=True)
    if not ps.is_file():
        raise FileNotFoundError(src)
    if pd.exists() and not overwrite:
        raise FileExistsError(dst)
    rs, rd = ps.relative_to(ws).as_posix(), pd.relative_to(ws).as_posix()
    policy.enforce("rename", f"{rs} -> {rd}", task_id=task_id)
    cp = _checkpoint_for(ws, rs, checkpoint_id, task_id, f"before rename {rs}")
    CP.ensure_file(cp, rd)
    pd.parent.mkdir(parents=True, exist_ok=True)
    os.replace(ps, pd)
    CP.record_after(cp, rs, None)
    CP.record_after(cp, rd, CP.sha1_bytes(pd.read_bytes()))
    with _cache_lock:
        _READ_CACHE.pop(os.path.normcase(str(ps)), None)
    _touch(ws, rd, "rename")
    return {"from": rs, "to": rd, "checkpoint_id": cp}


def delete_file(workspace: str, path: str, *, approved: bool = False,
                checkpoint_id: str | None = None, task_id: str | None = None) -> dict:
    """Soft delete: the file moves to the agent trash and stays in the checkpoint."""
    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path, for_write=True)
    if not p.is_file():
        raise FileNotFoundError(path)
    rel = p.relative_to(ws).as_posix()
    policy.enforce("delete", rel, approved=approved, task_id=task_id)
    cp = _checkpoint_for(ws, rel, checkpoint_id, task_id, f"before delete {rel}")
    dest = CP.move_to_trash(ws, p)
    CP.record_after(cp, rel, None)
    with _cache_lock:
        _READ_CACHE.pop(os.path.normcase(str(p)), None)
    _touch(ws, rel, "delete")
    return {"path": rel, "trash": str(dest), "checkpoint_id": cp}


# --------------------------------------------------------------- verification

def _quick_syntax(rel: str, text: str) -> dict:
    lang = language_of(rel)
    if lang == "python":
        import ast

        try:
            ast.parse(text)
            return {"checked": True, "ok": True, "lang": lang}
        except SyntaxError as e:
            return {"checked": True, "ok": False, "lang": lang, "line": e.lineno,
                    "column": e.offset, "error": f"SyntaxError: {e.msg}"}
    if lang == "json":
        try:
            json.loads(text)
            return {"checked": True, "ok": True, "lang": lang}
        except json.JSONDecodeError as e:
            return {"checked": True, "ok": False, "lang": lang, "line": e.lineno,
                    "column": e.colno, "error": f"JSONDecodeError: {e.msg}"}
    if lang == "toml":
        import tomllib

        try:
            tomllib.loads(text)
            return {"checked": True, "ok": True, "lang": lang}
        except tomllib.TOMLDecodeError as e:
            return {"checked": True, "ok": False, "lang": lang, "error": str(e)}
    return {"checked": False, "ok": None, "lang": lang,
            "note": "no in-process syntax checker for this language (use typecheck)"}


def syntax_check(workspace: str, path: str | None = None, *, glob: str = "*.py",
                 limit: int = 2000) -> dict:
    ws = policy.resolve_workspace(workspace)
    targets = [policy.resolve_path(ws, path)] if path else \
        [p for rel, p in iter_files(ws) if fnmatch.fnmatch(rel.rsplit("/", 1)[-1], glob)][:limit]
    results, errors = [], []
    for p in targets:
        rel = p.relative_to(ws).as_posix()
        try:
            text, _ = _read_text(p)
        except (OSError, ValueError) as e:
            results.append({"path": rel, "checked": False, "error": str(e)})
            continue
        r = {"path": rel, **_quick_syntax(rel, text)}
        results.append(r)
        if r.get("ok") is False:
            errors.append(r)
    checked = sum(1 for r in results if r.get("checked"))
    return {"checked": checked, "errors": errors, "ok": not errors,
            "results": results if path else results[:50]}


def lint(workspace: str, path: str | None = None) -> dict:
    """pyflakes, in-process (no subprocess, no model)."""
    from pyflakes import api as pf_api
    from pyflakes import reporter as pf_rep

    ws = policy.resolve_workspace(workspace)
    targets = [policy.resolve_path(ws, path)] if path else \
        [p for rel, p in iter_files(ws) if rel.endswith(".py")]
    issues = []
    for p in targets:
        rel = p.relative_to(ws).as_posix()
        try:
            text, _ = _read_text(p)
        except (OSError, ValueError):
            continue
        out, err = io.StringIO(), io.StringIO()
        pf_api.check(text, rel, pf_rep.Reporter(out, err))
        for line in (out.getvalue() + err.getvalue()).splitlines():
            m = re.match(r"^(.*?):(\d+):(?:(\d+):)?\s*(.*)$", line)
            if m:
                issues.append({"path": rel, "line": int(m.group(2)),
                               "column": int(m.group(3)) if m.group(3) else None,
                               "message": m.group(4)})
    return {"tool": "pyflakes", "files": len(targets), "count": len(issues), "issues": issues[:500]}


def format_file(workspace: str, path: str, *, checkpoint_id: str | None = None,
                task_id: str | None = None) -> dict:
    """Formats with black or ruff if installed. Neither is installed here - reported, not faked."""
    import importlib.util

    ws = policy.resolve_workspace(workspace)
    p = policy.resolve_path(ws, path, for_write=True)
    rel = p.relative_to(ws).as_posix()
    tool = next((m for m in ("ruff", "black") if importlib.util.find_spec(m)), None)
    if tool is None:
        raise RuntimeError("no formatter installed (black and ruff are both absent)")
    policy.enforce("format", rel, task_id=task_id)
    cp = _checkpoint_for(ws, rel, checkpoint_id, task_id, f"before format {rel}")
    argv = ["python", "-m", tool, "format", rel] if tool == "ruff" else ["python", "-m", "black", "-q", rel]
    r = run_command(ws, argv, timeout_s=60, task_id=task_id)
    CP.record_after(cp, rel, CP.sha1_bytes(p.read_bytes()))
    return {"path": rel, "tool": tool, "checkpoint_id": cp, **r}


# ----------------------------------------------------------------- execution

def child_cpu_cap() -> float:
    """CPU hard cap for child processes (percent of the whole machine)."""
    try:
        from backend.agent.governor import governor

        return governor().child_cpu_cap()
    except Exception:
        return max(10.0, SETTINGS.agent.cpu_cap_percent - 10.0)


def run_command(workspace: str | Path, argv: list[str], *, timeout_s: float | None = None,
                approved: bool = False, task_id: str | None = None, cancel_event=None,
                env_extra: dict | None = None) -> dict:
    ws = policy.resolve_workspace(workspace)
    d = policy.check_command(argv, ws)
    shown = " ".join(argv)[:300]
    if d.needs_approval and not approved:
        policy.audit("run_command", shown, "approval_required", d.risk, d.reason, task_id=task_id)
        raise policy.ApprovalRequired("run_command", shown, d.reason)
    if not d.allowed:
        policy.audit("run_command", shown, "deny", d.risk, d.reason, task_id=task_id)
        raise policy.PolicyError(d.reason)
    policy.audit("run_command", shown, "allow", d.risk, d.reason, task_id=task_id)
    cap = child_cpu_cap()
    r = run_process(policy.resolve_executable(argv), cwd=str(ws),
                    env=policy.scrubbed_env(env_extra),
                    timeout_s=timeout_s or SETTINGS.agent.command_timeout_s,
                    cpu_rate_percent=cap, cancel_event=cancel_event, label=shown)
    out = r.to_dict()
    out["argv"] = argv
    out["output"] = policy.redact(out["output"])[0]
    return out


_PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)")
_PYTEST_FAIL = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.M)
_TB_LOC = re.compile(r"^([\w./\\\-]+\.py):(\d+):\s*(.*)$", re.M)


def parse_pytest(output: str) -> dict:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    summary_line = ""
    for line in reversed(output.splitlines()):
        if _PYTEST_SUMMARY.search(line) and (" in " in line or "no tests ran" in line):
            summary_line = line.strip("= ").strip()
            break
    for n, what in _PYTEST_SUMMARY.findall(summary_line):
        key = "errors" if what.startswith("error") else what
        if key in counts:
            counts[key] = int(n)
    dur = re.search(r" in ([\d.]+)s", summary_line)
    failures = [{"kind": k.lower(), "nodeid": node, "message": (msg or "").strip()[:400]}
                for k, node, msg in _PYTEST_FAIL.findall(output)]
    if not summary_line:
        # Very quiet configurations omit the summary; the short test summary still lists
        # every failure and error, so those counts stay exact. Passed is then unknown.
        counts["failed"] = sum(1 for f in failures if f["kind"] == "failed")
        counts["errors"] = sum(1 for f in failures if f["kind"] == "error")
        counts["passed"] = None
    locations = []
    for f, ln, msg in _TB_LOC.findall(output):
        loc = {"path": f.replace("\\", "/"), "line": int(ln), "text": msg.strip()[:200]}
        if loc not in locations:
            locations.append(loc)
    return {**counts, "duration_s": float(dur.group(1)) if dur else None,
            "summary": summary_line, "failures": failures[:100], "locations": locations[:60]}


def detect_test_runner(workspace: str | Path) -> str | None:
    ws = policy.resolve_workspace(workspace)
    if any((ws / f).is_file() for f in ("pytest.ini", "conftest.py", "tox.ini")):
        return "pytest"
    for f in ("pyproject.toml", "setup.cfg"):
        fp = ws / f
        if fp.is_file() and "pytest" in fp.read_text(encoding="utf-8", errors="replace"):
            return "pytest"
    if (ws / "tests").is_dir() or any(ws.glob("test_*.py")):
        return "pytest"
    pkg = ws / "package.json"
    if pkg.is_file():
        try:
            if json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {}).get("test"):
                return "npm"
        except json.JSONDecodeError:
            return None
    return None


def run_tests(workspace: str, target: str | None = None, *, keyword: str | None = None,
              timeout_s: float | None = None, task_id: str | None = None, cancel_event=None) -> dict:
    ws = policy.resolve_workspace(workspace)
    runner = detect_test_runner(ws)
    if runner is None:
        raise RuntimeError("no test runner detected (no pytest config, tests/ dir or npm test script)")
    policy.enforce("run_tests", target or ".", task_id=task_id)
    if runner == "pytest":
        # No -q here: projects often set it in addopts, and -qq suppresses the summary line.
        argv = ["python", "-m", "pytest", "-rfE", "--tb=short", "--color=no",
                "-p", "no:cacheprovider"]
        if target:
            policy.resolve_path(ws, target.split("::")[0])  # must stay inside the workspace
            argv.append(target)
        if keyword:
            argv += ["-k", keyword]
    else:
        argv = ["npm", "test"]
    r = run_command(ws, argv, timeout_s=timeout_s or SETTINGS.agent.test_timeout_s,
                    task_id=task_id, cancel_event=cancel_event,
                    env_extra={"PYTHONPATH": str(ws)})
    parsed = parse_pytest(r["output"]) if runner == "pytest" else {}
    if runner == "pytest" and r["exit_code"] == 5:
        parsed["note"] = "no tests collected"
    # `ok` is the verdict; `passed`/`failed` (from the parser) are counts.
    return {"runner": runner, "target": target, **parsed, **r,
            "ok": r["exit_code"] == 0 and not r["timed_out"]}


def typecheck(workspace: str, project: str = ".", *, task_id: str | None = None) -> dict:
    """TypeScript `tsc --noEmit` using the workspace's own node_modules (no download)."""
    ws = policy.resolve_workspace(workspace)
    proj = policy.resolve_path(ws, project)
    tsc = proj / "node_modules" / "typescript" / "bin" / "tsc"
    if not tsc.is_file():
        raise RuntimeError("typescript is not installed in this project")
    argv = ["node", str(tsc), "--noEmit", "-p", str(proj)]
    r = run_command(ws, argv, timeout_s=300, task_id=task_id)
    errors = [{"path": m.group(1), "line": int(m.group(2)), "message": m.group(4)}
              for m in re.finditer(r"^(.+?)\((\d+),(\d+)\): error TS\d+: (.*)$", r["output"], re.M)]
    return {"ok": r["exit_code"] == 0, "error_count": len(errors), "errors": errors[:200], **r}


# ------------------------------------------------------------------------ git

def _git(ws: Path, args: list[str], timeout_s: float = 30) -> dict:
    return run_command(ws, ["git", *args], timeout_s=timeout_s)


def is_git_repo(workspace: str) -> bool:
    ws = policy.resolve_workspace(workspace)
    return (ws / ".git").exists() or any((p / ".git").exists() for p in ws.parents)


def git_status(workspace: str) -> dict:
    ws = policy.resolve_workspace(workspace)
    if not is_git_repo(str(ws)):
        return {"repo": False, "files": [], "branch": None}
    r = _git(ws, ["status", "--porcelain=v1", "-b", "--untracked-files=normal"])
    if r["exit_code"] != 0:
        raise RuntimeError(r["output"][:300])
    branch, files = None, []
    for line in r["output"].splitlines():
        if line.startswith("## "):
            branch = line[3:].split("...")[0].strip()
            continue
        if len(line) > 3:
            code, path = line[:2], line[3:].strip().strip('"')
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            files.append({"path": path, "index": code[0], "worktree": code[1],
                          "status": _git_label(code)})
    return {"repo": True, "branch": branch, "clean": not files, "count": len(files), "files": files}


def _git_label(code: str) -> str:
    if code == "??":
        return "untracked"
    for ch, label in (("A", "added"), ("D", "deleted"), ("R", "renamed"), ("M", "modified"), ("U", "conflict")):
        if ch in code:
            return label
    return "changed"


def git_diff(workspace: str, path: str | None = None, *, staged: bool = False,
             max_chars: int = 30000) -> dict:
    ws = policy.resolve_workspace(workspace)
    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--cached")
    if path:
        policy.resolve_path(ws, path)
    stat = _git(ws, args + ["--stat"] + (["--", path] if path else []))
    if path:
        args += ["--", path]
    r = _git(ws, args)
    text = r["output"]
    return {"diff": text[:max_chars], "truncated": len(text) > max_chars, "stat": stat["output"][-2000:],
            "files_changed": len(re.findall(r"^diff --git ", text, re.M))}


def git_log(workspace: str, n: int = 10) -> dict:
    ws = policy.resolve_workspace(workspace)
    r = _git(ws, ["log", f"-{max(1, min(int(n), 200))}", "--pretty=format:%h\x1f%an\x1f%ad\x1f%s",
                  "--date=short"])
    commits = []
    for line in r["output"].splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({"hash": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
    return {"count": len(commits), "commits": commits}


# ---------------------------------------------------------------- system

def process_monitor(top: int = 10) -> dict:
    import psutil

    procs = list(psutil.process_iter(["pid", "name", "memory_info"]))
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(0.3)
    ncpu = psutil.cpu_count() or 1
    rows = []
    for p in procs:
        try:
            rows.append({"pid": p.pid, "name": p.info["name"],
                         "cpu_percent": round(p.cpu_percent(None) / ncpu, 1),
                         "rss_mb": round((p.info["memory_info"].rss if p.info["memory_info"] else 0) / 1e6, 1)})
        except psutil.Error:
            continue
    vm = psutil.virtual_memory()
    return {
        "system_cpu_percent": psutil.cpu_percent(None), "ram_used_percent": vm.percent,
        "ram_available_gb": round(vm.available / 1e9, 2), "logical_cpus": ncpu,
        "top_cpu": sorted(rows, key=lambda r: -r["cpu_percent"])[:top],
        "top_ram": sorted(rows, key=lambda r: -r["rss_mb"])[:top],
    }


def workspace_summary(workspace: str) -> dict:
    """Deterministic project fingerprint: languages, manifests, test runner, size."""
    ws = policy.resolve_workspace(workspace)
    langs: dict[str, int] = {}
    n = 0
    for rel, _p in iter_files(ws, max_files=20000):
        n += 1
        lang = language_of(rel)
        if lang:
            langs[lang] = langs.get(lang, 0) + 1
    manifests = [f for f in ("pyproject.toml", "requirements.txt", "setup.py", "package.json",
                             "pytest.ini", "tsconfig.json", "CLAUDE.md", "README.md", "Cargo.toml",
                             "go.mod", "vite.config.ts") if (ws / f).exists()]
    for sub in ("frontend", "backend", "app", "src"):
        for f in ("package.json", "tsconfig.json", "pyproject.toml"):
            if (ws / sub / f).exists():
                manifests.append(f"{sub}/{f}")
    return {"workspace": str(ws), "files": n, "languages": dict(sorted(langs.items(), key=lambda kv: -kv[1])),
            "manifests": manifests, "test_runner": detect_test_runner(ws), "git": is_git_repo(str(ws))}
