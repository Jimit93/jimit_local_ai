"""Harness behaviour: routing, escalation, budgeting and validation.

None of these tests may call the LLM - if one does, the assertion on `llm_used`
fails, which is exactly the property being tested.
"""
import pytest

from backend.harness import router as R
from backend.harness.context_manager import build_context, compress, estimate_tokens, extract_payload
from backend.harness.validator import validate_numeric


def test_arithmetic_never_reaches_the_model():
    r = R.run_task("What is 25 * 40?", record=False)
    assert r["answer"].startswith("1000")
    assert r["llm_used"] is False and r["llm_calls"] == 0
    assert r["layer"] == "deterministic"
    assert r["latency_ms"] < 1000


def test_corpus_counting_uses_sql_not_the_model():
    r = R.run_task("How many posts do I have?", record=False)
    assert r["llm_used"] is False
    assert r["processor"] == "sqlite_query"


def test_exact_search_reports_zero_without_escalating():
    r = R.run_task("Find posts containing the word zzzznotpresentzzz", record=False)
    assert r["llm_used"] is False
    assert r["processor"] == "exact_keyword_search"
    assert r["answer"].startswith("0 document")
    assert r["escalations"] == 0


def test_escalation_happens_when_a_processor_cannot_answer():
    """Text stats with no payload must hand off rather than guess."""
    res = R.p_text_stats("count the words", {})
    assert not res.ok and res.confidence == 0.0


def test_every_step_is_recorded_in_the_trace():
    r = R.run_task("top 5 hashtags i use", record=False)
    assert r["steps"] and all("processor" in s for s in r["steps"])
    assert r["classifier"]["confidence"] > 0


def test_numeric_validation_catches_a_wrong_answer():
    good = validate_numeric("What is 25 * 40?", "The answer is 1000")
    bad = validate_numeric("What is 25 * 40?", "The answer is 65")
    assert good["status"] == "SUCCESS"
    assert bad["status"] == "FAILURE"


def test_context_budget_is_enforced():
    blocks = [("A", "x" * 5000), ("B", "y" * 5000), ("C", "z" * 5000)]
    text, acct = build_context(blocks, max_chars=1000)
    assert len(text) <= 1100
    assert acct["blocks_included"] and acct["blocks_dropped"]


def test_compress_cuts_on_a_boundary():
    out = compress("First sentence here. Second sentence here. Third one.", 30)
    assert out.endswith("…") and len(out) <= 32


def test_token_estimate_is_monotonic():
    assert estimate_tokens("a" * 380) > estimate_tokens("a" * 38) > 0


def test_payload_extraction():
    assert extract_payload('Count the words in "hello world this is a longer sample text '
                           'that exceeds forty characters"')
    assert extract_payload("Summarise this:\n" + "body text " * 10)
    assert extract_payload("what is the date today") is None


@pytest.mark.parametrize("task,expected_layer", [
    ("What is 15% of 2400?", "deterministic"),
    ("How many comments were ingested?", "deterministic"),
    ("Find posts semantically related to local AI", "retrieval"),
    ("What are the top 3 hashtags I use?", "nlp"),
])
def test_routing_layers(task, expected_layer):
    r = R.run_task(task, record=False)
    assert r["layer"] == expected_layer, f"{task} -> {r['processor']} ({r['layer']})"
    assert r["llm_used"] is False
