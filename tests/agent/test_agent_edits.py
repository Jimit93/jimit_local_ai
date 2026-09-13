"""Edit-block parsing and deterministic application / refusal."""
from pathlib import Path

from backend.agent import checkpoints as CP
from backend.agent import edits as E

GOOD = """Here is the fix:

calc.py
```python
<<<<<<< SEARCH
def add(a, b):
    \"\"\"Return the sum.\"\"\"
    return a - b
=======
def add(a, b):
    \"\"\"Return the sum.\"\"\"
    return a + b
>>>>>>> REPLACE
```
"""


def test_parse_tolerates_fences_and_prose():
    blocks = E.parse_blocks(GOOD)
    assert len(blocks) == 1 and blocks[0].path == "calc.py" and "a + b" in blocks[0].replace
    b2 = E.parse_blocks("**File: `src/x.py`**\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
    assert b2[0].path == "src/x.py"


def test_apply_exact_and_checkpointed(workspace):
    cp = CP.create(workspace, "t")
    rep = E.apply_blocks(workspace, E.parse_blocks(GOOD), allowed_files=["calc.py"], checkpoint_id=cp)
    assert rep.ok and rep.files_changed == ["calc.py"] and "+    return a + b" in rep.diff
    assert "return a + b" in Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert CP.rollback(cp)["ok"]


def test_whitespace_and_indentation_fallbacks(workspace):
    txt = "calc.py\n<<<<<<< SEARCH\ndef mul(a, b):   \n    return a * b\n=======\ndef mul(a, b):\n    return b * a\n>>>>>>> REPLACE"
    assert E.apply_blocks(workspace, E.parse_blocks(txt)).ok
    dedented = ("calc.py\n<<<<<<< SEARCH\nself.total = add(self.total, x)\nreturn self.total\n=======\n"
                "self.total = add(self.total, x)\nreturn int(self.total)\n>>>>>>> REPLACE")
    rep = E.apply_blocks(workspace, E.parse_blocks(dedented))
    assert rep.ok and rep.applied[0]["exact"] is False
    assert "        return int(self.total)" in Path(workspace, "calc.py").read_text(encoding="utf-8")


def test_over_indented_blocks_are_dedented_to_match(workspace):
    # Measured on qwen2.5-coder:3b: SEARCH and REPLACE both carried four extra spaces.
    txt = ("calc.py\n<<<<<<< SEARCH\n    def mul(a, b):\n        return a * b\n=======\n"
           "    def mul(a, b):\n        return a * b\n\n    def div(a, b):\n        return a / b\n>>>>>>> REPLACE")
    rep = E.apply_blocks(workspace, E.parse_blocks(txt), allow_new_files=True)
    assert rep.ok, rep.errors
    text = Path(workspace, "calc.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "\ndef div(a, b):\n    return a / b" in text and "\ndef mul(a, b):" in text


def test_reindent_keeps_crlf_line_endings(workspace):
    p = Path(workspace, "calc.py")
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    before = p.read_bytes()
    txt = ("calc.py\n<<<<<<< SEARCH\n    def mul(a, b):\n        return a * b\n=======\n"
           "    def mul(a, b):\n        return a * b\n\n    def div(a, b):\n        return a / b\n>>>>>>> REPLACE")
    assert E.apply_blocks(workspace, E.parse_blocks(txt), allow_new_files=True).ok
    after = p.read_bytes()
    assert b"\n" not in after.replace(b"\r\n", b"")  # no bare LF introduced
    assert after.startswith(before.split(b"def mul")[0])  # untouched lines byte-identical


def test_refusals(workspace):
    missing = "calc.py\n<<<<<<< SEARCH\nreturn a / b\n=======\nreturn 0\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(missing))
    assert not rep.ok and "not found" in rep.errors[0]
    outside = "tests/test_calc.py\n<<<<<<< SEARCH\nassert add(2, 3) == 5\n=======\nassert True\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(outside), allowed_files=["calc.py"])
    assert not rep.ok and "not one of the files" in rep.errors[0]
    new_file = "brand_new.py\n<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE"
    assert not E.apply_blocks(workspace, E.parse_blocks(new_file)).ok
    assert E.apply_blocks(workspace, E.parse_blocks(new_file), allow_new_files=True).ok
    escape = "../evil.py\n<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE"
    assert not E.apply_blocks(workspace, E.parse_blocks(escape), allow_new_files=True).ok
    assert not E.apply_blocks(workspace, []).ok


def test_empty_search_appends_only_when_new_code_is_allowed(workspace):
    add = "calc.py\n<<<<<<< SEARCH\n=======\ndef sub(a, b):\n    return a - b\n>>>>>>> REPLACE"
    assert not E.apply_blocks(workspace, E.parse_blocks(add)).ok  # a bug fix may not append
    rep = E.apply_blocks(workspace, E.parse_blocks(add), allow_new_files=True)
    assert rep.ok and rep.applied[0]["append"] is True
    text = Path(workspace, "calc.py").read_text(encoding="utf-8")
    assert text.rstrip().endswith("def sub(a, b):\n    return a - b") and "\n\n\ndef sub" in text


def test_syntax_error_feedback_shows_the_resulting_lines(workspace):
    bad = "calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return a * (b\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(bad))
    assert not rep.ok and "the file would become:" in rep.errors[0] and "| " in rep.errors[0]


def test_syntax_errors_are_never_written(workspace):
    before = Path(workspace, "calc.py").read_text(encoding="utf-8")
    bad = "calc.py\n<<<<<<< SEARCH\n    return a * b\n=======\n    return a * (b\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(bad))
    assert not rep.ok and rep.syntax_errors
    assert Path(workspace, "calc.py").read_text(encoding="utf-8") == before


def test_destructive_rewrite_is_refused(workspace):
    body = "\n".join(f"def f{i}():\n    return {i}" for i in range(40)) + "\n"
    Path(workspace, "many.py").write_text(body, encoding="utf-8")
    search = "\n".join(f"def f{i}():\n    return {i}" for i in range(30))
    txt = f"many.py\n<<<<<<< SEARCH\n{search}\n=======\npass\n>>>>>>> REPLACE"
    rep = E.apply_blocks(workspace, E.parse_blocks(txt))
    assert not rep.ok and "destructive" in rep.errors[0]


def test_deterministic_rename_skips_strings_and_comments(workspace):
    Path(workspace, "use.py").write_text('from calc import add\n# add is great\nprint(add(1, 2), "add")\n', encoding="utf-8")
    rep = E.rename_symbol(workspace, "add", "add_numbers")
    assert rep.ok and set(rep.files_changed) >= {"calc.py", "use.py", "tests/test_calc.py"}
    use = Path(workspace, "use.py").read_text(encoding="utf-8")
    assert "from calc import add_numbers" in use and "# add is great" in use and '"add"' in use
    assert "def add_numbers(a, b):" in Path(workspace, "calc.py").read_text(encoding="utf-8")
