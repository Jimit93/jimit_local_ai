"""Minimal context package for a coding request.

    request -> intent -> relevant files (ranked) -> relevant symbols -> code ranges
            -> memory -> history -> errors/tests -> budgeted, redacted package

Whole files are sent only when they are short. Long files contribute the symbols the
request or the traceback points at, not their full text. Every block is labelled and
added in priority order under a character budget (`harness.context_manager`), so the
least relevant material is what gets dropped.
"""
from __future__ import annotations

import time
from pathlib import Path

from backend.agent import policy
from backend.agent import project_index as PI
from backend.agent.code_tools import _read_text, git_status, is_git_repo
from backend.config import SETTINGS
from backend.harness.context_manager import build_context, estimate_tokens

WHOLE_FILE_MAX_LINES = 150
MAX_RANGE_LINES = 120


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 3:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def select_ranges(workspace: str, rel: str, request: str, error_lines: list[int]) -> tuple[list[tuple[int, int]], list[str]]:
    """Line ranges worth sending from one file, and the symbols they cover."""
    rec = PI.file_record(workspace, rel)
    total = (rec or {}).get("lines") or 0
    if total and total <= WHOLE_FILE_MAX_LINES:
        return [(1, total)], ["<whole file>"]
    syms = PI.symbols_in(workspace, rel)
    q = set(PI.code_tokens(request))
    chosen: list[dict] = []
    for s in syms:
        if s["kind"] == "constant":
            continue
        covers_error = any(s["line"] <= ln <= (s["end_line"] or s["line"]) for ln in error_lines)
        name_hit = bool(q & set(PI.code_tokens(s["name"])))
        if covers_error or name_hit:
            chosen.append({**s, "_err": covers_error})
    # Innermost symbols first (methods over their classes), error-covering first.
    chosen.sort(key=lambda s: (not s["_err"], (s["end_line"] or s["line"]) - s["line"]))
    ranges, names = [], []
    for s in chosen[:4]:
        a, b = s["line"], s["end_line"] or s["line"]
        if b - a > MAX_RANGE_LINES:
            centre = next((ln for ln in error_lines if a <= ln <= b), a)
            a, b = max(a, centre - MAX_RANGE_LINES // 2), min(b, centre + MAX_RANGE_LINES // 2)
        ranges.append((a, b))
        names.append(s["name"])
    for ln in error_lines:
        if not any(a <= ln <= b for a, b in ranges):
            ranges.append((max(1, ln - 15), ln + 15))
            names.append(f"<around line {ln}>")
    if not ranges:
        ranges = [(1, min(total or 40, 40))]
        names = ["<file head>"]
    return _merge(ranges), names


def build_code_context(
    workspace: str,
    request: str,
    *,
    error_text: str = "",
    memory_items: list[str] | None = None,
    history: list[str] | None = None,
    budget_chars: int | None = None,
    scale: float = 1.0,
    max_files: int = 4,
    include_repo_map: bool = True,
    pinned_files: list[str] | None = None,
) -> tuple[str, dict]:
    ws = str(policy.resolve_workspace(workspace))
    t0 = time.perf_counter()
    budget = int((budget_chars or SETTINGS.agent.context_max_chars) * max(0.3, min(1.0, scale)))
    modified: list[str] = []
    if is_git_repo(ws):
        try:
            modified = [f["path"] for f in git_status(ws)["files"]][:50]
        except Exception:
            modified = []
    ranking = PI.rank_files(ws, request, top_k=max(8, max_files * 2), error_text=error_text, modified=modified)
    ranked = ranking["files"]
    top_score = ranked[0]["score"] if ranked else 0.0
    picked: list[dict] = [{"path": p, "score": None, "signals": ["pinned"]} for p in (pinned_files or [])]
    for e in ranked:
        if len(picked) >= max_files:
            break
        if e["path"] in {p["path"] for p in picked}:
            continue
        if e["score"] >= 0.35 * top_score or e.get("signals"):
            picked.append(e)

    tb = PI.traceback_locations(error_text, ws)
    blocks: list[tuple[str, str]] = []
    files_acct, redactions = [], 0
    if error_text.strip():
        err, n = policy.redact(error_text.strip()[-1800:])
        redactions += n
        blocks.append(("Error output (most recent)", err))
    for e in picked:
        rel = e["path"]
        p = Path(ws) / rel
        if policy.is_secret_file(p) or not p.is_file():
            continue
        try:
            text, _ = _read_text(p)
        except (OSError, ValueError):
            continue
        lines = text.splitlines()
        err_lines = [loc["line"] for loc in tb if rel.lower().endswith(loc["path"].lower())]
        ranges, names = select_ranges(ws, rel, request, err_lines)
        parts = []
        for a, b in ranges:
            body = "\n".join(lines[a - 1:b])
            parts.append(body if (a, b) == (1, len(lines)) else f"# lines {a}-{b}\n{body}")
        code, n = policy.redact("\n# ...\n".join(parts))
        redactions += n
        label = f"File {rel}" + ("" if ranges == [(1, len(lines))] else f" (excerpts: {', '.join(names)})")
        blocks.append((label, code))
        files_acct.append({"path": rel, "ranges": ranges, "symbols": names, "score": e.get("score"),
                           "signals": e.get("signals", [])})
    if include_repo_map and ranked:
        rmap = PI.repo_map(ws, [e["path"] for e in ranked[:12]], max_chars=1200)
        if rmap:
            blocks.append(("Repository map (related files -> symbols)", rmap))
    if memory_items:
        blocks.append(("Project memory", "\n".join(f"- {m}" for m in memory_items[:12])))
    if history:
        blocks.append(("Task history", "\n".join(f"- {h}" for h in history[-8:])))

    text, acct = build_context(blocks, max_chars=budget)
    acct.update({
        "files": files_acct, "ranking_confidence": ranking["confidence"], "ranking_ms": ranking["ms"],
        "redactions": redactions, "budget_chars": budget, "tokens_est": estimate_tokens(text),
        "build_ms": round((time.perf_counter() - t0) * 1000, 1),
        "whole_repo_chars_avoided": _repo_chars(ws) - len(text),
    })
    return text, acct


_repo_size_cache: dict[str, tuple[float, int]] = {}


def _repo_chars(ws: str) -> int:
    """Total indexed source size - what 'send the whole repository' would have cost."""
    hit = _repo_size_cache.get(ws)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    from backend.db import query

    n = int(query("SELECT COALESCE(SUM(size),0) s FROM code_files WHERE workspace=?", (ws,))[0]["s"])
    _repo_size_cache[ws] = (time.time(), n)
    return n
