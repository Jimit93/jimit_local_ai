"""Incremental index, symbol lookup, import graph, ranking signals."""
import os
import time
from pathlib import Path

from backend.agent import project_index as PI

EXTRA = {
    "app/__init__.py": "",
    "app/router.py": '''"""Request routing."""
from app.handlers import handle_login


def route(path):
    """Dispatch an incoming request path to its handler."""
    if path == "/login":
        return handle_login()
''',
    "app/handlers.py": '''"""HTTP handlers."""


def handle_login():
    """Authenticate the user with password and session cookie."""
    return "ok"
''',
    "web/src/api.ts": "export async function fetchUser(id: string) {\n  return id;\n}\n",
    "web/src/App.tsx": "import { fetchUser } from './api';\nexport default function App() {\n  return null;\n}\n",
}


def _add(ws, files):
    for rel, body in files.items():
        p = Path(ws) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def test_build_is_incremental(workspace):
    _add(workspace, EXTRA)
    first = PI.build(workspace)
    assert first["changed"] >= 7 and first["unchanged"] == 0
    second = PI.build(workspace)
    assert second["changed"] == 0 and second["unchanged"] == first["files"]
    p = Path(workspace, "calc.py")
    p.write_text(p.read_text(encoding="utf-8") + "\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")
    os.utime(p, (time.time() + 5, time.time() + 5))
    third = PI.build(workspace)
    assert third["changed"] == 1
    assert PI.find_symbols(workspace, "sub", exact=True)["count"] == 1
    Path(workspace, "notes/readme.md").unlink()
    assert PI.build(workspace)["removed"] == 1


def test_secret_and_ignored_files_are_not_indexed(workspace):
    PI.build(workspace)
    from backend.db import query

    paths = {r["path"] for r in query("SELECT path FROM code_files WHERE workspace=?", (workspace,))}
    assert ".env" not in paths and not any(p.startswith("node_modules") for p in paths)


def test_import_graph_python_and_ts(workspace):
    _add(workspace, EXTRA)
    PI.build(workspace)
    g = PI.import_graph(workspace)
    assert "app/handlers.py" in g["app/router.py"]
    assert "web/src/api.ts" in g["web/src/App.tsx"]
    assert "calc.py" in g["tests/test_calc.py"]


def test_ranking_uses_content_symbols_and_mentions(workspace):
    _add(workspace, EXTRA)
    PI.build(workspace)
    r = PI.rank_files(workspace, "where is the password session authentication handled")
    assert r["files"][0]["path"] == "app/handlers.py"
    r2 = PI.rank_files(workspace, "the bug is in router.py")
    assert r2["files"][0]["path"] == "app/router.py" and "file_mentioned" in r2["files"][0]["signals"]
    r3 = PI.rank_files(workspace, "change handle_login")
    assert r3["files"][0]["path"] == "app/handlers.py" and "symbol_match" in r3["files"][0]["signals"]


def test_traceback_locations_boost_the_failing_file(workspace):
    _add(workspace, EXTRA)
    PI.build(workspace)
    tb = f'Traceback (most recent call last):\n  File "{workspace}\\calc.py", line 6, in add\nAssertionError'
    locs = PI.traceback_locations(tb, workspace)
    assert locs[0]["path"] == "calc.py" and locs[0]["line"] == 6
    r = PI.rank_files(workspace, "fix the failing test", error_text=tb)
    assert r["files"][0]["path"] == "calc.py" and "traceback" in r["files"][0]["signals"]


def test_code_tokens_split_identifiers():
    toks = PI.code_tokens("def handleLoginRequest(user_id): return parse_JSON")
    assert {"handleloginrequest", "handle", "login", "request", "user_id", "user", "parse_json"} <= set(toks)
    assert "def" not in toks and "return" not in toks


def test_repo_map_is_compact(workspace):
    _add(workspace, EXTRA)
    m = PI.repo_map(workspace)
    assert "calc.py: add, mul, Accumulator" in m and len(m) < 2500


def test_evaluate_ranking_runs(workspace):
    _add(workspace, EXTRA)
    ev = PI.evaluate_ranking(workspace, sample=20)
    assert ev["cases"] >= 2 and 0 <= ev["fused"]["mrr"] <= 1
