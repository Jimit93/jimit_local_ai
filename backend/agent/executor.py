"""Deterministic intent execution: one request -> one sandboxed tool call -> an answer.

No model is involved. When the request lacks what the tool needs (no path, no search
term) the executor returns confidence 0 with an error, and the harness escalates to the
agent loop instead of guessing.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time

from backend.agent import checkpoints as CP
from backend.agent import code_tools as C
from backend.agent import policy
from backend.agent.intents import IntentResult
from backend.common import ProcessorResult

HEAVY_TOOLS = {"run_tests": "tests", "typecheck": "build"}


def _res(result, answer: str, tool: str, t0: float, *, conf: float = 0.97, meta: dict | None = None) -> ProcessorResult:
    return ProcessorResult(result=result, confidence=conf, processor=f"tool:{tool}",
                           latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
                           meta={"answer": answer, "tool": tool, **(meta or {})})


def _fail(tool: str, t0: float, error: str, **meta) -> ProcessorResult:
    return ProcessorResult(result=None, confidence=0.0, processor=f"tool:{tool}",
                           latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
                           error=error, meta={"tool": tool, **meta})


_TERM_RX = [
    re.compile(r"\b(?:for|of|to)\s+[\"'`]?([^\"'`?]{2,60}?)[\"'`]?\s*(?:in\b.*)?[?.!]*$", re.I),
    re.compile(r"\bwhere (?:is|are)\s+[\"'`]?([\w.]+)[\"'`]?\s+(?:used|called|referenced|defined|declared|implemented)", re.I),
    re.compile(r"\bgrep\s+(?:for\s+)?[\"'`]?([^\"'`]+?)[\"'`]?\s*$", re.I),
    re.compile(r"\bwhich files (?:contain|mention|use)\s+[\"'`]?([^\"'`?]+?)[\"'`]?\s*[?.]*$", re.I),
]
_NOISE = re.compile(r"^(?:the|a|an)\s+(?:function|class|method|symbol|word|string|text)\s+", re.I)


def search_term(request: str, ir: IntentResult) -> str | None:
    if ir.params.get("quoted"):
        return ir.params["quoted"][0]
    for rx in _TERM_RX:
        m = rx.search(request.strip())
        if m:
            return _NOISE.sub("", m.group(1).strip()).strip() or None
    if ir.params.get("symbols"):
        return ir.params["symbols"][0]
    return None


def _resolve_file(ws: str, rel: str) -> str | None:
    """Exact path, else a unique match on the file name (e.g. 'router.py')."""
    try:
        if policy.resolve_path(ws, rel).is_file():
            return rel
    except policy.PolicyError:
        raise
    name = rel.replace("\\", "/").split("/")[-1].lower()
    matches = [r for r, _ in C.iter_files(ws) if r.lower().endswith("/" + rel.lower()) or r.lower().split("/")[-1] == name]
    return matches[0] if len(matches) == 1 else None


def _first_file(ws: str, ir: IntentResult, tool: str, t0: float):
    files = ir.params.get("files") or []
    if not files:
        return None, _fail(tool, t0, "no file path found in the request")
    rel = _resolve_file(ws, files[0])
    if rel is None:
        return None, _fail(tool, t0, f"file not found (or ambiguous): {files[0]}")
    return rel, None


def run_intent_tool(ir: IntentResult, request: str, workspace: str | None, *, task_id: str | None = None,
                    session_id: str | None = None, approved: bool = False, cancel_event=None) -> ProcessorResult:
    t0 = time.perf_counter()
    intent = ir.intent
    try:
        ws = str(policy.resolve_workspace(workspace))
    except policy.PolicyError as e:
        return _fail(intent, t0, str(e))
    try:
        return _dispatch(intent, ir, request, ws, t0, task_id=task_id, session_id=session_id,
                         approved=approved, cancel_event=cancel_event)
    except policy.ApprovalRequired as e:
        return _fail(intent, t0, f"approval_required: {e.reason}", approval={"action": e.action, "target": e.target})
    except policy.PolicyError as e:
        return _fail(intent, t0, f"denied: {e}")
    except Exception as e:
        return _fail(intent, t0, f"{type(e).__name__}: {e}")


def _dispatch(intent, ir, request, ws, t0, *, task_id, session_id, approved, cancel_event) -> ProcessorResult:
    p = ir.params
    if intent == "calculate":
        from backend.tools.deterministic import calculate

        v = calculate(request)
        return _res(v, f"{v['value']}", "calculate", t0)

    if intent == "file_exists":
        target = (p.get("files") or p.get("quoted") or [None])[0]
        if not target:
            return _fail("file_exists", t0, "no path found in the request")
        r = C.file_exists(ws, target)
        return _res(r, f"{target}: {'exists' if r['exists'] else 'does not exist'}"
                    + (" (directory)" if r["is_dir"] else ""), "file_exists", t0)

    if intent == "list_directory":
        path = p.get("directory") or (p.get("files") or ["."])[0]
        r = C.list_directory(ws, path)
        shown = [f"{e['name']}{'/' if e['is_dir'] else ''}" for e in r["entries"] if not e["ignored"]]
        return _res(r, f"{path} ({r['count']} entries):\n" + "\n".join(shown[:80]), "list_directory", t0)

    if intent == "read_file":
        rel, err = _first_file(ws, ir, "read_file", t0)
        if err:
            return err
        r = C.read_file(ws, rel, start_line=p.get("start_line"), end_line=p.get("end_line"))
        head = f"{r['path']} lines {r['start_line']}-{r['end_line']} of {r['total_lines']}" + \
               (" (cached)" if r["cached"] else "") + (" [truncated]" if r["truncated"] else "")
        return _res(r, head + "\n" + r["content"], "read_file", t0, meta={"files": [rel]})

    if intent == "file_info":
        rel, err = _first_file(ws, ir, "file_info", t0)
        if err:
            return err
        r = C.file_info(ws, rel)
        return _res(r, f"{rel}: {r.get('lines', '?')} lines, {r['size_bytes']} bytes, {r['language'] or 'unknown'}, "
                       f"modified {r['modified']}", "file_info", t0, meta={"files": [rel]})

    if intent == "file_outline":
        rel, err = _first_file(ws, ir, "file_symbols", t0)
        if err:
            return err
        r = C.file_symbols(ws, rel)
        lines = [f"{'  ' if s['parent'] else ''}{s['kind']} {s['name']} (line {s['line']})"
                 + (f" - {s['doc']}" if s.get("doc") else "") for s in r["symbols"]]
        return _res(r, f"{rel}: {len(r['symbols'])} symbols\n" + "\n".join(lines[:120]), "file_symbols", t0,
                    meta={"files": [rel]})

    if intent == "search_files":
        pat = p.get("glob") or search_term(request, ir)
        if not pat:
            return _fail("search_files", t0, "no file name or pattern found in the request")
        pat = re.sub(r"^(?:named|called|matching)\s+", "", pat)
        r = C.search_files(ws, pat)
        return _res(r, f"{r['count']} file(s) match '{pat}':\n" + "\n".join(r["files"][:100]), "search_files", t0)

    if intent == "search_code":
        term = search_term(request, ir)
        if not term:
            return _fail("search_code", t0, "no search term found in the request")
        r = C.search_code(ws, term, limit=200)
        lines = [f"{m['path']}:{m['line']}: {m['text']}" for m in r["matches"][:60]]
        return _res(r, f"{r['count']} match(es) for '{term}' in {r['files_matched']} file(s) "
                       f"({r['files_scanned']} scanned):\n" + "\n".join(lines), "search_code", t0,
                    meta={"files": sorted({m['path'] for m in r['matches']})[:20]})

    if intent == "search_symbols":
        name = (p.get("symbols") or [search_term(request, ir)])[0]
        if not name:
            return _fail("search_symbols", t0, "no symbol name found in the request")
        r = C.search_symbols(ws, name, exact=True)
        if not r["count"]:
            r = C.search_symbols(ws, name, exact=False)
        lines = [f"{s['path']}:{s['line']}  {s['kind']} {s['name']}" + (f"  ({s['parent']})" if s.get("parent") else "")
                 for s in r["symbols"][:40]]
        return _res(r, f"{r['count']} definition(s) of '{name}':\n" + "\n".join(lines), "search_symbols", t0,
                    conf=0.97 if r["count"] else 0.9, meta={"files": sorted({s['path'] for s in r['symbols']})[:20]})

    if intent == "git_status":
        r = C.git_status(ws)
        if not r["repo"]:
            return _res(r, "not a git repository", "git_status", t0)
        lines = [f"{f['status']:>9}  {f['path']}" for f in r["files"][:100]]
        return _res(r, f"branch {r['branch']}: " + ("clean" if r["clean"] else f"{r['count']} changed file(s)\n" + "\n".join(lines)),
                    "git_status", t0)

    if intent == "git_diff":
        path = (p.get("files") or [None])[0]
        r = C.git_diff(ws, path)
        return _res(r, (r["stat"].strip() + "\n\n" + r["diff"]).strip() or "no unstaged changes", "git_diff", t0)

    if intent == "git_log":
        r = C.git_log(ws, p.get("n") or 10)
        return _res(r, "\n".join(f"{c['hash']} {c['date']} {c['author']}: {c['subject']}" for c in r["commits"]),
                    "git_log", t0)

    if intent in ("run_tests", "typecheck"):
        from backend.agent.governor import governor

        with governor().slot(HEAVY_TOOLS[intent], timeout_s=120, cancel_event=cancel_event) as adm:
            if not adm.admitted:
                return _fail(intent, t0, adm.reason, governor=adm.to_dict())
            if intent == "typecheck":
                proj = "frontend" if (policy.resolve_workspace(ws) / "frontend" / "tsconfig.json").exists() else "."
                r = C.typecheck(ws, proj, task_id=task_id)
                ans = f"tsc: {'no errors' if r['ok'] else str(r['error_count']) + ' error(s)'}\n" + \
                      "\n".join(f"{e['path']}({e['line']}): {e['message']}" for e in r["errors"][:30])
                return _res(r, ans, "typecheck", t0, meta={"governor": adm.to_dict()})
            target = p.get("test_target") or next((f for f in p.get("files", []) if "test" in f.lower()), None)
            r = C.run_tests(ws, target, task_id=task_id, cancel_event=cancel_event)
        fails = "\n".join(f"  {f['kind'].upper()} {f['nodeid']} - {f['message']}" for f in r.get("failures", [])[:20])
        ans = (f"{'PASSED' if r['ok'] else 'FAILED'}: {r.get('summary') or ('exit code ' + str(r['exit_code']))}"
               + (f"\n{fails}" if fails else "") + (" (timed out)" if r["timed_out"] else ""))
        # A failing test run is still a correct, complete answer to "run the tests".
        return _res(r, ans, "run_tests", t0, meta={"governor": adm.to_dict()})

    if intent == "lint":
        path = (p.get("files") or [p.get("directory")] or [None])[0]
        r = C.lint(ws, path)
        return _res(r, f"pyflakes: {r['count']} issue(s) in {r['files']} file(s)\n" +
                    "\n".join(f"{i['path']}:{i['line']}: {i['message']}" for i in r["issues"][:60]), "lint", t0)

    if intent == "syntax_check":
        path = (p.get("files") or [None])[0]
        if path:
            path = _resolve_file(ws, path) or path
        r = C.syntax_check(ws, path)
        ans = (f"syntax OK ({r['checked']} file(s) checked)" if r["ok"] else
               f"{len(r['errors'])} syntax error(s):\n" + "\n".join(
                   f"{e['path']}:{e.get('line')}: {e.get('error')}" for e in r["errors"][:30]))
        return _res(r, ans, "syntax_check", t0)

    if intent == "parse_data_file":
        rel, err = _first_file(ws, ir, "parse_data_file", t0)
        if err:
            return err
        text = C.read_file(ws, rel, max_chars=2_000_000)["content"]
        if rel.lower().endswith(".json"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return _res({"valid": False, "line": e.lineno, "column": e.colno, "error": e.msg},
                            f"{rel}: INVALID JSON at line {e.lineno}, column {e.colno}: {e.msg}", "parse_json", t0)
            shape = (f"object with {len(data)} keys: {', '.join(list(data)[:20])}" if isinstance(data, dict)
                     else f"array of {len(data)} items" if isinstance(data, list) else type(data).__name__)
            return _res({"valid": True, "shape": shape}, f"{rel}: valid JSON, {shape}", "parse_json", t0)
        rows = list(csv.DictReader(io.StringIO(text)))
        cols = list(rows[0].keys()) if rows else []
        return _res({"rows": len(rows), "columns": cols}, f"{rel}: {len(rows)} rows, columns: {', '.join(cols)}",
                    "parse_csv", t0)

    if intent == "system_status":
        from backend.agent.governor import governor

        pm = C.process_monitor(8)
        g = governor().status(history=0)
        ans = (f"CPU {pm['system_cpu_percent']}% · RAM available {pm['ram_available_gb']} GB ({pm['ram_used_percent']}% used) "
               f"· governor {g['state']}\nTop RAM: " + ", ".join(f"{r['name']} {r['rss_mb']} MB" for r in pm["top_ram"][:6])
               + "\nTop CPU: " + ", ".join(f"{r['name']} {r['cpu_percent']}%" for r in pm["top_cpu"][:6]))
        return _res({"processes": pm, "governor": g}, ans, "process_monitor", t0)

    if intent == "memory_query":
        from backend.agent import memory as MEM
        from backend.agent import state as S

        unfinished = [t for t in S.list_tasks(limit=100) if t["status"] not in S.TERMINAL]
        hits = MEM.search(request, workspace=ws, limit=6)
        lines = [f"- [{t['status']}] {t['topic']} (step: {t['current_step'] or '-'})" for t in unfinished[:10]]
        dec = [f"- {h['kind']}: {json.dumps(h['value'], ensure_ascii=False)[:140]}" for h in hits]
        ans = (f"{len(unfinished)} unfinished task(s)" + (":\n" + "\n".join(lines) if lines else "")
               + ("\nRelevant memory:\n" + "\n".join(dec) if dec else ""))
        return _res({"unfinished": unfinished, "memory": hits}, ans, "memory_search", t0)

    if intent == "list_checkpoints":
        cps = [c for c in CP.list_checkpoints(limit=30) if policy._norm(c["workspace"]) == policy._norm(ws)]
        ans = "\n".join(f"{c['id']}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(c['ts']))}  {c['label']}"
                        + ("  (restored)" if c.get("restored_at") else "") for c in cps) or "no checkpoints"
        return _res(cps, ans, "list_checkpoints", t0)

    if intent == "rollback":
        from backend.agent import memory as MEM

        cp_id = None
        if session_id:
            f = MEM.resolve_followup(session_id, "undo")
            if f and f.get("task_id"):
                from backend.agent import state as S

                try:
                    cp_id = S.load(f["task_id"]).get("last_checkpoint")
                except Exception:
                    cp_id = None
        if cp_id is None:
            cps = [c for c in CP.list_checkpoints(limit=30)
                   if policy._norm(c["workspace"]) == policy._norm(ws) and not c.get("restored_at")
                   and any(f["modified"] for f in c["files"])]
            cp_id = cps[0]["id"] if cps else None
        if cp_id is None:
            return _fail("rollback", t0, "no agent change to undo in this workspace")
        r = CP.rollback(cp_id, task_id=task_id)
        ans = (f"rolled back checkpoint {cp_id}: " + ", ".join(f"{a['path']} {a['action']}" for a in r["actions"])
               + (f"\nconflicts (left untouched): {r['conflicts']}" if r["conflicts"] else ""))
        return _res(r, ans, "rollback", t0, conf=0.97 if r["ok"] else 0.5)

    if intent == "project_overview":
        from backend.agent import memory as MEM
        from backend.agent import project_index as PI

        PI.build(ws)
        facts = MEM.learn_project(ws)
        s = facts["summary"]
        ans = (f"{s['files']} files · languages: " + ", ".join(f"{k} {v}" for k, v in list(s["languages"].items())[:6])
               + f"\nmanifests: {', '.join(s['manifests'])}\ntests: {s['test_runner'] or 'none detected'}"
               + ("\nmost-connected files: " + ", ".join(facts.get("important_files", [])[:8]) if facts.get("important_files") else "")
               + "\n\n" + PI.repo_map(ws, facts.get("important_files") or None, max_chars=1800))
        return _res(facts, ans, "workspace_summary", t0)

    if intent == "rename_file":
        if not (p.get("rename_from") and p.get("rename_to")):
            return _fail("rename_file", t0, "could not read 'rename X to Y' from the request")
        r = C.rename_file(ws, p["rename_from"], p["rename_to"], task_id=task_id)
        return _res(r, f"renamed {r['from']} -> {r['to']} (checkpoint {r['checkpoint_id']})", "rename_file", t0)

    if intent == "delete_file":
        rel, err = _first_file(ws, ir, "delete_file", t0)
        if err:
            return err
        r = C.delete_file(ws, rel, approved=approved, task_id=task_id)
        return _res(r, f"moved {rel} to the agent trash (checkpoint {r['checkpoint_id']}; undo restores it)",
                    "delete_file", t0)

    return _fail(intent, t0, f"no deterministic executor for intent '{intent}'")
