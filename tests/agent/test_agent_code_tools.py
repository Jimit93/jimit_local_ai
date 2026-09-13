"""Workspace tools: read cache, edits, checkpoints/rollback, search, symbols, tests, git."""
from pathlib import Path

import pytest

from backend.agent import checkpoints as CP
from backend.agent import code_tools as C
from backend.agent import policy as P
from backend.tools import registry as R


def test_read_file_uses_cache_for_unchanged_files(workspace):
    first = C.read_file(workspace, "calc.py")
    second = C.read_file(workspace, "calc.py")
    assert first["cached"] is False and second["cached"] is True
    assert "def add" in second["content"] and second["total_lines"] > 5


def test_secret_file_is_never_read(workspace):
    with pytest.raises(P.PolicyError):
        C.read_file(workspace, ".env")


def test_search_skips_ignored_dirs_and_secrets(workspace):
    r = C.search_code(workspace, "module.exports")
    assert r["count"] == 0  # node_modules pruned
    assert C.search_code(workspace, "sk-live")["count"] == 0  # .env skipped
    hits = C.search_code(workspace, "add(")
    assert {m["path"] for m in hits["matches"]} >= {"calc.py", "tests/test_calc.py"}
    assert C.search_files(workspace, "*.py")["count"] == 2


def test_symbols_python(workspace):
    r = C.search_symbols(workspace, "add", exact=True)
    assert any(s["path"] == "calc.py" and s["kind"] == "function" for s in r["symbols"])
    fs = C.file_symbols(workspace, "calc.py")
    kinds = {(s["name"], s["kind"]) for s in fs["symbols"]}
    assert ("Accumulator", "class") in kinds and ("push", "method") in kinds


def test_edit_is_checkpointed_and_rolls_back(workspace):
    before = Path(workspace, "calc.py").read_text(encoding="utf-8")
    r = C.edit_file(workspace, "calc.py", "return a - b", "return a + b")
    assert r["syntax"]["ok"] is True and "+    return a + b" in r["diff"]
    assert "return a + b" in Path(workspace, "calc.py").read_text(encoding="utf-8")
    rb = CP.rollback(r["checkpoint_id"])
    assert rb["ok"] and rb["actions"][0]["action"] == "restored"
    assert Path(workspace, "calc.py").read_text(encoding="utf-8") == before


def test_edit_requires_unique_match_and_reports_closest_lines(workspace):
    with pytest.raises(LookupError, match="closest"):
        C.edit_file(workspace, "calc.py", "return a -- b", "x")
    Path(workspace, "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(LookupError, match="2 times"):
        C.edit_file(workspace, "dup.py", "x = 1", "x = 2")
    assert C.edit_file(workspace, "dup.py", "x = 1", "x = 2", replace_all=True)["replacements"] == 2


def test_rollback_refuses_to_clobber_external_changes(workspace):
    r = C.edit_file(workspace, "calc.py", "return a * b", "return b * a")
    Path(workspace, "calc.py").write_text("# user rewrote this\n", encoding="utf-8")
    rb = CP.rollback(r["checkpoint_id"])
    assert rb["ok"] is False and rb["conflicts"]
    assert Path(workspace, "calc.py").read_text(encoding="utf-8") == "# user rewrote this\n"


def test_created_file_is_removed_on_rollback(workspace):
    r = C.write_file(workspace, "new_mod.py", "def f():\n    return 1\n")
    assert r["created"] and Path(workspace, "new_mod.py").exists()
    CP.rollback(r["checkpoint_id"])
    assert not Path(workspace, "new_mod.py").exists()


def test_write_reports_syntax_errors(workspace):
    r = C.write_file(workspace, "broken.py", "def f(:\n")
    assert r["syntax"]["ok"] is False and r["syntax"]["line"] == 1


def test_delete_is_soft_and_needs_approval(workspace):
    with pytest.raises(P.ApprovalRequired):
        C.delete_file(workspace, "config.json")
    r = C.delete_file(workspace, "config.json", approved=True)
    assert not Path(workspace, "config.json").exists() and Path(r["trash"]).exists()
    CP.rollback(r["checkpoint_id"])
    assert Path(workspace, "config.json").exists()


def test_rename_and_rollback(workspace):
    r = C.rename_file(workspace, "notes/readme.md", "docs/README.md")
    assert Path(workspace, "docs/README.md").exists()
    CP.rollback(r["checkpoint_id"])
    assert Path(workspace, "notes/readme.md").exists() and not Path(workspace, "docs/README.md").exists()


def test_syntax_and_lint(workspace):
    Path(workspace, "bad.py").write_text("import os\ndef f(:\n", encoding="utf-8")
    s = C.syntax_check(workspace)
    assert not s["ok"] and s["errors"][0]["path"] == "bad.py"
    Path(workspace, "bad.py").write_text("import os\n\ndef f():\n    return undefined_name\n", encoding="utf-8")
    msgs = [i["message"] for i in C.lint(workspace, "bad.py")["issues"]]
    assert any("imported but unused" in m for m in msgs)
    assert any("undefined name" in m for m in msgs)


def test_run_tests_parses_failures(workspace):
    r = C.run_tests(workspace)
    assert r["runner"] == "pytest" and r["ok"] is False
    assert r["failed"] == 1 and r["passed"] == 1
    assert any("test_add" in f["nodeid"] for f in r["failures"])
    assert r["job_object"] is True and r["cpu_cap_percent"]


def test_parse_pytest_summary():
    out = "FAILED tests/t.py::test_a - AssertionError: 1\n3 failed, 10 passed, 2 skipped in 1.50s\n"
    p = C.parse_pytest(out)
    assert (p["failed"], p["skipped"], p["duration_s"]) == (3, 2, 1.5)
    assert p["failures"][0]["nodeid"] == "tests/t.py::test_a"


def test_format_is_honest_when_no_formatter(workspace):
    import importlib.util

    if importlib.util.find_spec("black") or importlib.util.find_spec("ruff"):
        pytest.skip("a formatter is installed")
    with pytest.raises(RuntimeError, match="no formatter"):
        C.format_file(workspace, "calc.py")


def test_disallowed_command_is_denied(workspace):
    with pytest.raises(P.PolicyError):
        C.run_command(workspace, ["git", "push"])


def test_git_tools_on_this_repo():
    from backend.config import ROOT

    s = C.git_status(str(ROOT))
    assert s["repo"] is True and s["branch"]
    assert C.git_log(str(ROOT), 3)["count"] >= 1


def test_registry_exposes_workspace_tools(workspace):
    names = {t.name for t in R.all_tools()}
    for n in ("read_file", "edit_file", "search_code", "run_tests", "git_status", "run_command"):
        assert n in names
    res = R.call("file_exists", workspace=workspace, path="calc.py")
    assert res.ok and res.result["exists"] is True
    denied = R.call("read_file", workspace=workspace, path="../../etc/passwd")
    assert not denied.ok and "escapes" in denied.error


def test_process_monitor():
    r = C.process_monitor(3)
    assert len(r["top_ram"]) == 3 and r["logical_cpus"] > 0
