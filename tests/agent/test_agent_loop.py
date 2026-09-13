"""The controlled loop end-to-end with a scripted model (no inference).

The scripted model stands in for Ollama so these tests exercise real tools, real
pytest runs, real checkpoints and real state - only the text a model would return is
fixed. Real-model runs are in the backtest.
"""
import json
from pathlib import Path

import pytest

from backend.agent import checkpoints as CP
from backend.agent import governor as G
from backend.agent import loop as L
from backend.agent import state as S
from backend.harness import router as R
from backend.harness.task_classifier import classify_task

FIX = """calc.py
<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE"""

WRONG = """calc.py
<<<<<<< SEARCH
    return a - b
=======
    return a - b + 0
>>>>>>> REPLACE"""


class Script:
    """Returns queued responses; records every call."""

    def __init__(self, *responses, tier="small"):
        self.responses = list(responses)
        self.calls = []
        self.tier = tier

    def __call__(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        text = self.responses.pop(0) if self.responses else ""
        tier = kw.get("tier", self.tier)
        return type("R", (), {"text": text, "model": f"scripted-{tier}", "total_tokens": 50,
                              "prompt_tokens": 40, "completion_tokens": 10, "meta": {"tier": tier}})()


@pytest.fixture(autouse=True)
def calm_governor():
    g = G.Governor(sampler=lambda: (10.0, 1.0, 8.0), interval_s=0.05, persist=False)
    g.tick()
    G.set_governor(g)
    yield g
    G.set_governor(None)


def _run(ws, request, llm, **opts):
    st = L.create_task(ws, request, session_id="sess", options={"allow_edits": True, **opts})
    return L.run_agent(st["id"], llm=llm)


def test_fix_bug_full_loop(workspace):
    llm = Script(FIX)
    st = _run(workspace, "fix the failing test in tests/test_calc.py", llm)
    assert st["status"] == S.COMPLETED, st["last_output"]
    assert "return a + b" in Path(workspace, "calc.py").read_text(encoding="utf-8")
    names = [s["name"] for s in st["plan"]]
    assert names[:3] == ["reproduce", "locate", "gather_context"] and st["completed_steps"][-1] == "report"
    assert len(llm.calls) == 1 and st["counters"]["llm_calls"] == 1
    ev = st["evaluation"]
    assert ev["actual"]["files_changed"] == ["calc.py"] and ev["predicted"]["files_changed"] == ["calc.py"]
    assert ev["checks"]["tests_match"] is True and ev["match_score"] == 1.0
    # tests and non-code files (pytest.ini) are never offered as editable in a bug fix;
    # the function under test is found from the failing test's body.
    assert st["data"]["allowed_files"] == ["calc.py"]
    assert st["data"]["functions_under_test"] == ["calc.py"]
    kinds = [e["kind"] for e in S.events(st["id"])]
    assert {"understand", "plan", "checkpoint", "step_end", "complete"} <= set(kinds)
    assert CP.get(st["last_checkpoint"])["files"][0]["path"] == "calc.py"


def test_repair_loop_escalates_then_succeeds(workspace):
    llm = Script(WRONG, FIX)
    st = _run(workspace, "fix the failing test in tests/test_calc.py", llm)
    assert st["status"] == S.COMPLETED, st["last_output"]
    assert st["counters"]["repairs"] == 1 and len(llm.calls) == 2
    assert llm.calls[1]["tier"] == "capable"  # escalated after the failed verification
    assert "Verification failure" in llm.calls[1]["prompt"]
    assert any(e["kind"] == "repair" for e in S.events(st["id"]))


def test_same_failure_twice_blocks_instead_of_looping(workspace):
    llm = Script(WRONG, WRONG, WRONG, WRONG)
    st = _run(workspace, "fix the failing test in tests/test_calc.py", llm)
    assert st["status"] in (S.BLOCKED, S.FAILED)
    assert len(llm.calls) <= 3
    # the failed attempt was rolled back: the file is exactly as before
    assert "return a - b\n" in Path(workspace, "calc.py").read_text(encoding="utf-8")


def test_inapplicable_edit_is_retried_then_fails_cleanly(workspace):
    junk = "calc.py\n<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE"
    llm = Script(junk, "no blocks at all")
    st = _run(workspace, "fix the failing test in tests/test_calc.py", llm)
    assert st["status"] == S.FAILED and "propose_edit" in st["failed_steps"], st["last_output"]
    assert Path(workspace, "calc.py").read_text(encoding="utf-8").count("return a - b") == 1


def test_read_only_mode_never_edits(workspace):
    llm = Script("`add` returns `a - b`; it should return `a + b`.")
    st = L.create_task(workspace, "fix the failing test in tests/test_calc.py", options={"allow_edits": False})
    st = L.run_agent(st["id"], llm=llm)
    assert st["status"] == S.COMPLETED and "answer" in st["completed_steps"]
    assert "return a - b" in Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert st["validation_results"][-1]["grounded_ratio"] == 1.0


def test_deterministic_intent_uses_no_model(workspace):
    llm = Script()
    st = _run(workspace, "where is the function add defined", llm)
    assert st["status"] == S.COMPLETED and not llm.calls
    assert "calc.py" in st["last_output"] and st["evaluation"]["checks"]["no_llm_when_deterministic"] is True


def test_rename_symbol_is_deterministic_and_verified(workspace):
    Path(workspace, "calc.py").write_text(Path(workspace, "calc.py").read_text(encoding="utf-8")
                                          .replace("return a - b", "return a + b"), encoding="utf-8")
    llm = Script()
    st = _run(workspace, "rename function mul to multiply", llm)
    assert st["status"] == S.COMPLETED, st["last_output"]
    assert not llm.calls
    assert "def multiply(a, b):" in Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert "from calc import add, multiply" in Path(workspace, "tests/test_calc.py").read_text(encoding="utf-8")


def test_delete_waits_for_approval(workspace):
    st = L.create_task(workspace, "delete the file config.json", options={"allow_edits": True})
    st = L.run_agent(st["id"], llm=Script())
    assert st["status"] == S.AWAITING and st["data"]["pending_approval"]["action"] == "delete_file"
    assert Path(workspace, "config.json").exists()
    st["options"]["approved_actions"] = ["delete_file"]
    S.save(st, status=S.QUEUED)
    st = L.run_agent(st["id"], llm=Script())
    assert st["status"] == S.COMPLETED and not Path(workspace, "config.json").exists()


def test_cancel_and_iteration_limit(workspace):
    import threading

    ev = threading.Event()
    ev.set()
    st = L.create_task(workspace, "explain how the Accumulator class works", options={})
    assert L.run_agent(st["id"], llm=Script("x"), cancel_event=ev)["status"] == S.CANCELLED
    st = L.create_task(workspace, "explain how the Accumulator class works", options={"max_iterations": 1})
    assert L.run_agent(st["id"], llm=Script("x"))["status"] == S.BLOCKED


def test_governor_deferral_blocks_heavy_steps_until_deadline(workspace, calm_governor):
    for _ in range(3):
        calm_governor.tick((97.0, 1.0, 8.0))
    st = L.create_task(workspace, "explain how the Accumulator class works", options={"timeout_s": 3})
    st = L.run_agent(st["id"], llm=Script("x"))
    assert st["status"] == S.BLOCKED and "deadline" in st["last_output"]
    assert any(e["kind"] == "governor" for e in S.events(st["id"]))


def test_model_unavailable_fails_safely(workspace):
    def broken(prompt, **kw):
        raise RuntimeError("no coding model is installed in Ollama")

    st = L.create_task(workspace, "explain how the Accumulator class works", options={})
    st = L.run_agent(st["id"], llm=broken)
    assert st["status"] == S.FAILED and "no model could answer" in st["last_output"]


def test_llm_cache_avoids_repeat_calls(workspace):
    llm = Script("`Accumulator` keeps a running total using `add`.")
    st1 = L.run_agent(L.create_task(workspace, "explain how the Accumulator class works")["id"], llm=llm)
    st2 = L.run_agent(L.create_task(workspace, "explain how the Accumulator class works")["id"], llm=llm)
    assert len(llm.calls) == 1 and st2["counters"]["cache_hits"] == 1 and st2["counters"]["llm_calls"] == 0
    assert st1["last_output"] == st2["last_output"]


def test_memory_replay_fixes_a_recurring_error_without_the_model(workspace):
    st = _run(workspace, "fix the failing test in tests/test_calc.py", Script(FIX))
    assert st["status"] == S.COMPLETED
    Path(workspace, "calc.py").write_text(Path(workspace, "calc.py").read_text(encoding="utf-8")
                                          .replace("return a + b", "return a - b"), encoding="utf-8")
    llm = Script()
    st2 = _run(workspace, "fix the failing test in tests/test_calc.py", llm)
    assert st2["status"] == S.COMPLETED and not llm.calls
    assert any(s["method"] == "memory" for s in st2["plan"])


def test_requirements_catch_a_passing_but_wrong_implementation(workspace):
    # Measured on qwen2.5-coder:3b: asked to add a function, it edited an existing one; the
    # existing tests still passed. The requirements check must reject that and repair.
    Path(workspace, "calc.py").write_text(Path(workspace, "calc.py").read_text(encoding="utf-8")
                                          .replace("return a - b", "return a + b"), encoding="utf-8")
    wrong = "calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return b * a\n>>>>>>> REPLACE"
    right = "calc.py\n<<<<<<< SEARCH\n=======\ndef sub(a, b):\n    return a - b\n>>>>>>> REPLACE"
    llm = Script(wrong, right)
    st = _run(workspace, "add a function sub(a, b) to calc.py that subtracts b from a", llm)
    assert st["status"] == S.COMPLETED, st["last_output"]
    assert st["counters"]["repairs"] == 1 and len(llm.calls) == 2
    assert "`sub` is not defined" in llm.calls[1]["prompt"]
    assert "sub(2 parameters)" in llm.calls[0]["prompt"]
    text = Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert "def sub(a, b):" in text and "return a * b" in text  # the wrong edit was rolled back


MEASURED_BEFORE = '''"""String helpers."""
import re


def slugify(text):
    """Lower-case, hyphen-separated slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-")
'''
# Exactly what qwen2.5-coder:3b produced three times in the system backtest.
MEASURED_AFTER = '''"""String helpers."""
import re


def slugify(text):
    """Lower-case, hyphen-separated slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-").replace(" ", "").lower()

    def is_palindrome(text):
        """Check if a string is a palindrome, ignoring case and spaces."""
        cleaned_text = text.replace(" ", "").lower()
        return cleaned_text == cleaned_text[::-1]
'''


def test_hoist_and_revert_on_the_measured_model_output():
    import ast

    hoisted, moved = L._hoist_dead_nested(MEASURED_AFTER, {"is_palindrome"})
    assert moved == ["is_palindrome"]
    tree = ast.parse(hoisted)
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ["slugify", "is_palindrome"]
    fixed, reverted = L._revert_unrequested(MEASURED_BEFORE, hoisted, "utils/strings.py", {"is_palindrome"})
    assert reverted == ["slugify"]
    ns: dict = {}
    exec(compile(fixed, "strings.py", "exec"), ns)
    assert ns["slugify"]("Hello, World!") == "hello-world"
    assert ns["is_palindrome"]("never odd or even") and not ns["is_palindrome"]("python")
    assert 'return text.strip("-")\n' in fixed
    # nothing requested -> nothing moved; code before a return is never moved
    assert L._hoist_dead_nested(MEASURED_AFTER, {"other"})[1] == []


def test_nested_or_unrelated_changes_do_not_meet_requirements(workspace):
    Path(workspace, "calc.py").write_text(Path(workspace, "calc.py").read_text(encoding="utf-8")
                                          .replace("return a - b", "return a + b"), encoding="utf-8")
    nested = ("calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return a * b\n\n    def sub(a, b):\n"
              "        return a - b\n>>>>>>> REPLACE")
    unrelated = ("calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return b * a\n>>>>>>> REPLACE\n\n"
                 "calc.py\n<<<<<<< SEARCH\n=======\ndef sub(a, b):\n    return a - b\n>>>>>>> REPLACE")
    right = "calc.py\n<<<<<<< SEARCH\n=======\ndef sub(a, b):\n    return a - b\n>>>>>>> REPLACE"
    # Both habits are normalised deterministically, so each completes with a single model call.
    for edit in (nested, unrelated):
        make = Path(workspace, "calc.py").read_text(encoding="utf-8")
        llm = Script(edit)
        st = _run(workspace, "add a function sub(a, b) to calc.py", llm)
        assert st["status"] == S.COMPLETED, st["last_output"]
        assert len(llm.calls) == 1 and any(e["kind"] == "normalize" for e in S.events(st["id"]))
        text = Path(workspace, "calc.py").read_text(encoding="utf-8")
        assert "    return a * b" in text and text.count("def sub(") == 1 and "\ndef sub(a, b):" in text.replace("\r\n", "\n")
        Path(workspace, "calc.py").write_text(make, encoding="utf-8")
    assert right  # the plain append path is covered by the test above


def test_refine_follow_up_continues_the_previous_change(workspace):
    from backend.agent import memory as MEM

    st = L.run_agent(L.create_task(workspace, "fix the failing test in tests/test_calc.py", session_id="sr",
                                   options={"allow_edits": True})["id"], llm=Script(FIX))
    assert st["status"] == S.COMPLETED
    ref = MEM.refinement_of("sr", "now also make it reject non-numbers", workspace)
    assert ref and ref["task_id"] == st["id"] and ref["files"][0] == "calc.py" and "return a + b" in ref["diff"]
    assert MEM.refinement_of("sr", "git status", workspace) is None             # a new command
    assert MEM.refinement_of("sr", "now open notes/readme.md", workspace) is None  # names another file
    assert MEM.refinement_of("other-session", "now also do X", workspace) is None

    guard = ("calc.py\n<<<<<<< SEARCH\ndef add(a, b):\n    \"\"\"Return the sum.\"\"\"\n=======\ndef add(a, b):\n"
             "    \"\"\"Return the sum.\"\"\"\n    if not all(isinstance(v, (int, float)) for v in (a, b)):\n"
             "        raise TypeError(\"numbers only\")\n>>>>>>> REPLACE")
    llm = Script(guard)
    t2 = L.create_task(workspace, "now also make it reject non-numbers", session_id="sr",
                       options={"allow_edits": True, "refine_of": ref})
    assert t2["intent"] in ("fix_bug", "implement") and "calc.py" in t2["data"]["intent"]["params"]["files"]
    assert any(e["kind"] == "refine" for e in S.events(t2["id"]))
    t2 = L.run_agent(t2["id"], llm=llm)
    assert t2["status"] == S.COMPLETED, t2["last_output"]
    assert "refines the previous request" in llm.calls[0]["prompt"] and "Change already made for it" in llm.calls[0]["prompt"]
    assert 'raise TypeError("numbers only")' in Path(workspace, "calc.py").read_text(encoding="utf-8")


def test_functions_under_test_reads_the_failing_test_body(workspace):
    from backend.agent import project_index as PI

    PI.build(workspace)
    Path(workspace, "shop.py").write_text("def total(xs):\n    return sum(xs) - 1\n", encoding="utf-8")
    Path(workspace, "tests/test_shop.py").write_text("from shop import total\n\n\ndef test_total():\n    assert total([1, 2]) == 3\n",
                                                   encoding="utf-8")
    PI.build(workspace)
    got = L.functions_under_test(workspace, [{"nodeid": "tests/test_shop.py::test_total"},
                                             {"nodeid": "tests/test_calc.py::test_add"}])
    assert got == ["shop.py", "calc.py"]


def test_model_degrades_to_what_fits_in_ram(monkeypatch):
    from backend.llm import manager as M
    from backend.llm.base import LLMResult

    models = [{"tier": "small", "name": "big-coder", "size_gb": 1.9, "preferred": True},
              {"tier": "small", "name": "tiny-coder", "size_gb": 0.4, "preferred": False}]
    monkeypatch.setattr(M, "code_models", lambda: models)

    def ensure(model, task_id=None):
        if model["name"] == "big-coder":
            raise M.InsufficientRAM("big-coder", {"shortfall_gb": 1.2, "end_available_gb": 2.0, "candidates": []})
        return {"already_loaded": False}

    monkeypatch.setattr(M, "_ensure_ram_for", ensure)

    class FakeAdapter:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, **kw):
            return LLMResult("ok.", 3, 1, 5.0, self.name, "ollama")

    monkeypatch.setattr("backend.llm.ollama_adapter.QwenOllamaAdapter", FakeAdapter)
    M._code_adapters.clear()
    r = M.generate_code("x", tier="small")
    assert r.model == "tiny-coder" and r.meta["degraded_from"] == "big-coder"


def test_general_model_load_checks_ram_first(monkeypatch):
    from backend.agent import ram as RAM
    from backend.llm import manager as M

    calls = {}
    monkeypatch.setattr(M, "is_loaded", lambda: False)
    monkeypatch.setattr(RAM, "reclaim", lambda need, **kw: calls.setdefault("reclaim", (need, kw)) and {"ok": False})
    monkeypatch.setattr(RAM, "available_gb", lambda: 0.2)
    monkeypatch.setattr(M, "llm", lambda force_reload=False: calls.setdefault("reload", force_reload))
    import pytest as _pt

    with _pt.raises(M.InsufficientRAM):
        M._ensure_ram_general()
    assert calls["reclaim"][1]["keep_models"] == {"llama.cpp"} and calls["reclaim"][1]["use_allowlist"] is False


def test_placeholder_path_is_retargeted_only_with_a_unique_match(workspace):
    from backend.agent import edits as E

    ok = "path/to/file.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return b * a\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(ok), allowed_files=["calc.py"])
    assert rep.ok and rep.files_changed == ["calc.py"] and rep.applied[0].get("retargeted_from") == "path/to/file.py"
    bad = "path/to/file.py\n<<<<<<< SEARCH\nnot in calc\n=======\nx\n>>>>>>> REPLACE"
    assert not E.apply_blocks(workspace, E.parse_blocks(bad), allowed_files=["calc.py"]).ok
    two = E.apply_blocks(workspace, E.parse_blocks(ok.replace("b * a", "a * b")), allowed_files=["calc.py", "other.py"])
    assert not two.ok  # ambiguous target: never guessed


def test_secret_and_plural_requests_route_to_the_right_path(workspace):
    from backend.agent import intents as I
    from backend.harness.task_classifier import classify_task

    r = I.classify_intent("show me .env")
    assert r.intent == "read_file" and r.params["files"] == [".env"]
    assert classify_task("add three functions to utils/strings.py: title_case(text) and count_vowels(text)").result == "coding"
    st = L.run_agent(L.create_task(workspace, "show me .env", options={})["id"], llm=Script())
    assert st["status"] == S.FAILED and st["counters"]["llm_calls"] == 0 and "secret" in st["last_output"]


# ------------------------------------------------------------- harness integration

def test_router_routes_workspace_requests_to_tools(workspace):
    out = R.run_task("where is the function add defined", options={"workspace": workspace}, record=False)
    assert out["task_type"] == "workspace" and out["llm_used"] is False
    assert out["processor"] == "tool:search_symbols" and "calc.py" in out["answer"]


def test_router_runs_the_agent_for_coding_requests(workspace):
    out = R.run_task("fix the failing test in tests/test_calc.py",
                     options={"workspace": workspace, "allow_edits": True, "llm": Script(FIX)}, record=False)
    assert out["task_type"] == "coding" and out["processor"] == "coding_agent(loop)"
    assert out["result"]["status"] == "completed" and out["llm_calls"] == 1
    steps = [s["processor"] for s in out["steps"]]
    assert steps == ["tool:fix_bug", "coding_agent(loop)"] or steps[-1] == "coding_agent(loop)"


def test_refine_must_change_the_referenced_function(workspace):
    from backend.agent import memory as MEM

    Path(workspace, "calc.py").write_text(Path(workspace, "calc.py").read_text(encoding="utf-8")
                                          .replace("return a - b", "return a + b"), encoding="utf-8")
    add = "calc.py\n<<<<<<< SEARCH\n=======\ndef sub(a, b):\n    return a - b\n>>>>>>> REPLACE"
    st = L.run_agent(L.create_task(workspace, "add a function sub(a, b) to calc.py", session_id="rt",
                                   options={"allow_edits": True})["id"], llm=Script(add))
    assert st["status"] == S.COMPLETED, st["last_output"]
    ref = MEM.refinement_of("rt", "now also make it return 0 when both are None", workspace)
    assert ref["targets"] == ["sub"]
    # Measured habit: the follow-up edits a different function. It must be undone and repaired.
    wrong = "calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return (a or 0) * (b or 0)\n>>>>>>> REPLACE"
    right = ("calc.py\n<<<<<<< SEARCH\ndef sub(a, b):\n    return a - b\n=======\ndef sub(a, b):\n"
             "    if a is None and b is None:\n        return 0\n    return a - b\n>>>>>>> REPLACE")
    llm = Script(wrong, right)
    t = L.create_task(workspace, "now also make it return 0 when both are None", session_id="rt",
                      options={"allow_edits": True, "refine_of": ref})
    t = L.run_agent(t["id"], llm=llm)
    assert t["status"] == S.COMPLETED, t["last_output"]
    assert "`sub` from the previous request" in llm.calls[0]["prompt"] and len(llm.calls) == 2
    text = Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert "return a * b" in text and "if a is None and b is None" in text


def test_forced_category_is_not_reclassified(workspace):
    # Found by the refine before/after run: the worker re-classified an intake-created agent task
    # ("now also make it return False for empty strings" -> reasoning) and it never executed.
    st = L.create_task(workspace, "explain how the Accumulator class works", options={})
    out = R.run_task("now also make it return False for empty strings",
                     options={"workspace": workspace, "task_id": st["id"], "llm": Script("`Accumulator` sums values."),
                              "force_category": "coding"}, record=False)
    assert out["task_type"] == "coding" and out["classifier"]["processor"] == "forced:coding"
    assert S.load(st["id"])["status"] == S.COMPLETED


def test_existing_classifications_are_unchanged():
    base = json.loads(Path(__file__).with_name("classifier_baseline.json").read_text(encoding="utf-8"))
    changed = {t: (want, classify_task(t).result) for t, want in base.items() if classify_task(t).result != want}
    assert not changed, changed
