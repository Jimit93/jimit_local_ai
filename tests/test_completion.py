"""Replies must end cleanly inside the token limit (no mid-sentence cut, no half edit block)."""
from backend.harness import completion as C
from backend.llm import manager as M
from backend.llm.base import LLMResult


def test_untruncated_output_is_untouched():
    text = "The router tries tools first. Then the model"
    assert C.finish(text, "stop")[0] == text


def test_trims_to_last_complete_sentence():
    text = "The governor samples CPU every second. It defers heavy work above the cap. When load falls it"
    out, meta = C.finish(text, "length")
    assert out == "The governor samples CPU every second. It defers heavy work above the cap."
    assert meta["truncated"] and meta["action"] == "trimmed_to_last_sentence" and meta["trimmed_chars"] > 0


def test_list_items_and_quotes():
    text = "Steps:\n1. Reproduce the failure.\n2. Locate the file.\n3. Apply the"
    out, meta = C.finish(text, "length")
    assert out.endswith("2. Locate the file.") and "3. Apply" not in out
    out2, _ = C.finish('He said "use tools first." Then the', "length")
    assert out2 == 'He said "use tools first."'


def test_dangling_heading_is_dropped():
    # Measured on qwen2.5-coder: the trim point landed right after a markdown heading.
    text = "Here is a guide to fixing tests.\n\n### 1. Understand the Test Case\n- **Identify the"
    out, meta = C.finish(text, "length")
    assert out == "Here is a guide to fixing tests." and "dropped_dangling_leadin" in meta["action"]


def test_dangling_leadins_are_dropped():
    # Both measured on qwen2.5-coder:3b at a 60-token limit.
    a = "BM25 scores term overlap. Here are some reasons why BM25 works well for this purpose:\n\n1. Exact"
    assert C.finish(a, "length")[0] == "BM25 scores term overlap."
    b = "Fix it carefully. Here is a structured approach to follow:\n\n1. **Understand the Test Failure**:\n   - Read"
    assert C.finish(b, "length")[0] == "Fix it carefully."


def test_code_fences_are_closed_or_dropped():
    text = "Use this:\n```python\ndef f():\n    return 1\n    x ="
    out, meta = C.finish(text, "length")
    assert out.count("```") == 2 and "return 1" in out and meta["action"] == "closed_code_fence"
    out2, meta2 = C.finish("Here is the fix. ```py\nx", "length")
    assert "```" not in out2 and out2 == "Here is the fix."


def test_no_complete_sentence_gets_an_explicit_ellipsis():
    out, meta = C.finish("The main reason the loop stops is that the same", "length")
    assert out.endswith("…") and meta["action"] == "ellipsis_no_complete_sentence"


def test_unterminated_edit_block_is_dropped():
    good = "calc.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE"
    text = good + "\n\ncalc.py\n<<<<<<< SEARCH\nc\n=======\nd"
    out, meta = C.finish(text, "length", mode="edits")
    assert out == good and meta["action"] == "dropped_unterminated_edit_block"
    assert C.finish(good, "length", mode="edits")[0] == good


def test_budget_instruction_scales_with_tokens():
    assert C.word_budget(600) > C.word_budget(100) >= 8
    assert "complete sentence" in C.budget_instruction(200)


def test_manager_applies_finish(monkeypatch):
    class Fake:
        name, runtime, loaded = "fake", "fake", True

        def generate(self, prompt, **kw):
            return LLMResult("First point is done. Second point is not", 5, 9, 1.0, "fake", "fake", stop_reason="length")

    monkeypatch.setattr(M, "llm", lambda force_reload=False: Fake())
    monkeypatch.setattr(M, "_ensure_ram_general", lambda: {})  # never unload real models in a unit test
    r = M.generate("x")
    assert r.text == "First point is done." and r.meta["completion"]["truncated"] is True
    raw = M.generate("x", finish="none")
    assert raw.text.endswith("is not")
