"""Coding intents: rule coverage/collisions, parameter extraction, ML fallback, complexity."""
import pytest

from backend.agent import intents as I


@pytest.mark.parametrize("text,intent", [
    ("Calculate 345 × 67", "calculate"),
    ("does backend/config.py exist?", "file_exists"),
    ("list the backend folder", "list_directory"),
    ("show me backend/agent/policy.py", "read_file"),
    ("show lines 10-30 of backend/db.py", "read_file"),
    ("how many lines in backend/harness/router.py", "file_info"),
    ("find all *.tsx files", "search_files"),
    ("find files named test_router", "search_files"),
    ("where is run_task used", "search_code"),
    ("grep for TODO", "search_code"),
    ("find usages of build_context", "search_code"),
    ("where is the function run_task defined", "search_symbols"),
    ("find the definition of Governor", "search_symbols"),
    ("git status", "git_status"),
    ("which files are modified?", "git_status"),
    ("show me the diff", "git_diff"),
    ("show the last 5 commits", "git_log"),
    ("run the tests", "run_tests"),
    ("run tests/agent/test_agent_policy.py", "run_tests"),
    ("lint backend/agent", "lint"),
    ("check syntax of calc.py", "syntax_check"),
    ("typecheck the frontend", "typecheck"),
    ("parse config.json", "parse_data_file"),
    ("what's using the RAM right now", "system_status"),
    ("what was I working on?", "memory_query"),
    ("list checkpoints", "list_checkpoints"),
    ("undo the last change", "rollback"),
    ("summarize the architecture of this project", "project_overview"),
    ("outline backend/agent/state.py", "file_outline"),
    ("rename the file notes/a.md to docs/a.md", "rename_file"),
    ("delete the file old_script.py", "delete_file"),
    ("rename function add to add_numbers", "rename_symbol"),
    ("fix the failing test in tests/test_calc.py", "fix_bug"),
    ("the tests are failing", "fix_bug"),
    ("line_total returns the wrong sum, fix it", "fix_bug"),
    ("the export button doesn't work, please fix", "fix_bug"),
    ("why does this raise a KeyError", "analyze_error"),
    ("refactor the planner module", "refactor"),
    ("write unit tests for backend/agent/edits.py", "write_tests"),
    ("review my changes", "review"),
    ("add a function that parses dates to utils.py", "implement"),
    ("create a new file utils/strings.py", "create_file"),
    ("explain how the select_ranges function works", "explain_code"),
])
def test_rules(text, intent):
    r = I.classify_intent(text)
    assert r.intent == intent, (text, r.intent, r.processor)


@pytest.mark.parametrize("text", [
    "What is 25 * 40?", "How many posts do I have?", "Find all LinkedIn posts mentioning RAG",
    "Write a LinkedIn post about local AI", "What have I written about risk management",
])
def test_non_coding_requests_are_not_deterministic_code_tools(text):
    r = I.classify_intent(text)
    assert r.intent in ("calculate", "general_coding", "search_code", "explain_code", "implement") or r.stage != "rules"


def test_params():
    p = I.extract_params("show lines 10-20 of `backend/db.py` and the class Governor in governor.py")
    assert p["files"] == ["backend/db.py", "governor.py"]
    assert (p["start_line"], p["end_line"]) == (10, 20)
    assert "Governor" in p["symbols"]
    assert I.extract_params("rename function add to add_numbers")["rename_to"] == "add_numbers"
    assert I.extract_params("find all *.tsx files")["glob"] == "*.tsx"
    assert I.extract_params("delete the file config.json")["files"] == ["config.json"]
    rc = I.extract_params("add title_case(text), count_vowels(text) and truncate(text, n) to utils/strings.py "
                          "(it capitalizes each word (like this))")["required_calls"]
    assert rc == [{"name": "title_case", "arity": 1}, {"name": "count_vowels", "arity": 1}, {"name": "truncate", "arity": 2}]
    # Found by the system backtest: the name before "folder" was ignored and "." was listed.
    assert I.extract_params("list the utils folder")["directory"] == "utils"
    assert I.extract_params("what is in the backend/agent directory")["directory"] == "backend/agent"
    assert I.extract_params("list files in src")["directory"] == "src"
    assert "directory" not in I.extract_params("list the current directory")
    assert I.extract_params("open src/App.tsx and api.ts")["files"] == ["src/App.tsx", "api.ts"]
    assert I.extract_params("run tests/agent/test_x.py::test_y")["test_target"] == "tests/agent/test_x.py::test_y"
    assert I.extract_params('Traceback (most recent call last):\n  File "a.py", line 3\nKeyError: 1')["has_traceback"]


def test_ml_fallback_and_low_confidence_is_safe():
    r = I.classify_intent("could you tidy up the duplicated helpers a bit")
    assert r.stage in ("ml", "ml_low_confidence")
    assert r.intent in I.INTENTS
    meta = I.model().meta
    assert meta["n_train"] > 40 and 0 < meta["cv_accuracy_mean"] <= 1


def test_complexity_levels():
    low = I.classify_intent("explain what select_ranges does")
    high = I.classify_intent("implement a thread-safe cache across all modules and migrate every caller")
    assert low.complexity["level"] == "low" and high.complexity["level"] == "high"
