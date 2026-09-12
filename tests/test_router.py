"""Router internals: chain definitions, thresholds and answer extraction."""
import pytest

from backend.common import ProcessorResult
from backend.harness import router as R
from backend.harness.task_classifier import CATEGORIES


def test_every_category_has_a_chain():
    for cat in CATEGORIES:
        assert cat in R.CHAINS, f"no processor chain for {cat}"
        assert R.CHAINS[cat], f"empty chain for {cat}"


def test_cheap_layers_come_before_the_model():
    """The LLM may only ever appear last in a chain."""
    for cat, chain in R.CHAINS.items():
        if R.p_llm in chain:
            assert chain[-1] is R.p_llm, f"{cat} calls the LLM before cheaper processors"


def test_thresholds_are_stricter_for_cheaper_layers():
    t = R.LAYER_THRESHOLD
    assert t["deterministic"] > t["nlp"] > t["ml"] > t["retrieval"] >= t["llm"]


def test_acceptance_requires_confidence_and_success():
    ok = ProcessorResult("x", 0.99, "p", 1.0, "deterministic")
    weak = ProcessorResult("x", 0.10, "p", 1.0, "deterministic")
    broken = ProcessorResult(None, 1.0, "p", 1.0, "deterministic", error="boom")
    assert R._accepted(ok) is True
    assert R._accepted(weak) is False
    assert R._accepted(broken) is False


@pytest.mark.parametrize("task,expected", [
    ("find posts mentioning RAG", "RAG"),
    ('search my posts for "supply zone"', "supply zone"),
    ("show posts containing Qwen", "Qwen"),
])
def test_search_term_extraction(task, expected):
    assert R._extract_search_term(task) == expected


def test_write_verb_stripping_keeps_the_topic():
    assert R._strip_write_verbs("Write a LinkedIn post about local inference") == \
        "local inference"
    assert R._strip_write_verbs("draft a short post on CPU benchmarks") == "CPU benchmarks"


def test_corpus_detection():
    assert R._mentions_corpus("what makes my writing distinctive") is True
    assert R._mentions_corpus("why is the sky blue") is False


def test_answer_extraction_prefers_the_prepared_answer():
    res = ProcessorResult({"raw": 1}, 1.0, "p", 1.0, "deterministic")
    res.meta["answer"] = "42"
    assert R._answer_of(res) == "42"
    assert R._answer_of(ProcessorResult("plain", 1.0, "p", 1.0)) == "plain"
    assert R._answer_of(None) == ""


def test_baseline_mode_bypasses_classification():
    """Baseline must not consult the router - that is the point of the comparison."""
    calls = []
    original = R.p_llm

    def spy(task, ctx):
        calls.append(task)
        return ProcessorResult("stub", 0.6, "qwen_llm(stub)", 1.0, "llm",
                               llm_used=True, llm_calls=1, tokens=10,
                               meta={"answer": "stub"})

    R.p_llm = spy
    try:
        out = R.run_task("What is 25 * 40?", mode="baseline", record=False)
    finally:
        R.p_llm = original
    assert calls == ["What is 25 * 40?"]
    assert out["task_type"] == "baseline"
    assert out["llm_used"] is True
