"""Context pipeline: relevant files only, symbol excerpts for long files, budget, redaction."""
from pathlib import Path

from backend.agent import project_index as PI
from backend.agent.context import build_code_context, select_ranges


def _long_module(ws: str) -> None:
    funcs = "\n\n".join(f"def helper_{i}(x):\n    \"\"\"Helper number {i}.\"\"\"\n" + "    y = x\n" * 8 + "    return y"
                        for i in range(30))
    Path(ws, "big.py").write_text(
        funcs + '\n\n\ndef parse_invoice_total(lines):\n    """Sum invoice line amounts."""\n'
        "    total = 0\n    for ln in lines:\n        total += ln\n    return total\n", encoding="utf-8")


def test_context_contains_relevant_file_and_not_irrelevant_ones(workspace):
    text, acct = build_code_context(workspace, "fix add in calc.py returning the wrong sum")
    assert acct["files"][0]["path"] == "calc.py"
    assert "def add" in text
    assert "module.exports" not in text  # node_modules never reaches the model


def test_long_files_send_symbol_excerpts_not_whole_file(workspace):
    _long_module(workspace)
    PI.build(workspace, force=True)
    ranges, names = select_ranges(workspace, "big.py", "the parse_invoice_total function is wrong", [])
    assert names == ["parse_invoice_total"]
    text, acct = build_code_context(workspace, "the parse_invoice_total function is wrong")
    assert "def parse_invoice_total" in text and "def helper_3(" not in text
    big = next(f for f in acct["files"] if f["path"] == "big.py")
    assert big["ranges"] != [(1, sum(1 for _ in open(Path(workspace, "big.py"), encoding="utf-8")))]


def test_traceback_line_selects_the_enclosing_symbol(workspace):
    _long_module(workspace)
    PI.build(workspace, force=True)
    line = next(i for i, ln in enumerate(Path(workspace, "big.py").read_text(encoding="utf-8").splitlines(), 1)
                if "total += ln" in ln)
    ranges, names = select_ranges(workspace, "big.py", "fix the crash", [line])
    assert "parse_invoice_total" in names


def test_budget_is_enforced_and_secrets_are_redacted(workspace):
    Path(workspace, "settings.py").write_text(
        'API_TOKEN = "tok_9f8a7b6c5d4e3f2a1b0c"\nTOKENIZER = "bert-base-uncased"\nDEBUG = True\n',
        encoding="utf-8")
    PI.build(workspace, force=True)
    text, acct = build_code_context(workspace, "what does settings.py configure", budget_chars=1500)
    assert len(text) <= 1500 + 50
    assert "tok_9f8a7b6c5d4e3f2a1b0c" not in text and acct["redactions"] >= 1
    assert "bert-base-uncased" in text  # not credential-shaped: left for the model to see


def test_scale_shrinks_budget(workspace):
    _, full = build_code_context(workspace, "calc add", budget_chars=4000)
    _, half = build_code_context(workspace, "calc add", budget_chars=4000, scale=0.5)
    assert half["budget_chars"] == 2000 and full["budget_chars"] == 4000
