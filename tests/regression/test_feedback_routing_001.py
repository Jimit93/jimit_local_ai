"""Regression: user feedback must be routed to the layer that owns the fix.

The rule that matters is negative: feedback must NOT be turned into extra prompt
text by default, and it must not cost an LLM call to classify.
"""
import pytest

from backend.services.feedback import classify_feedback

CASES = [
    ("It keeps using generic AI buzzwords.", "VALIDATION"),
    ("Not technical enough.", "VALIDATION"),
    ("It ignored my numbers.", "VALIDATION"),
    ("Too long.", "HARNESS"),
    ("Too many hashtags.", "HARNESS"),
    ("The CTA is wrong, I'm not hiring.", "HARNESS"),
    ("It selected the wrong audience.", "CLASSIFICATION"),
    ("This doesn't sound like me.", "RETRIEVAL"),
    ("The hook is weak.", "STYLE"),
    ("Too generic.", "STYLE"),
    ("It repeats itself and is garbled.", "MODEL"),
]


@pytest.mark.parametrize("text,layer", CASES)
def test_feedback_routes_to_the_owning_layer(text, layer):
    r = classify_feedback(text)
    assert r.ok
    assert r.result["primary_layer"] == layer, (
        f"{text!r} -> {r.result['primary_layer']}, expected {layer}")


@pytest.mark.parametrize("text,_layer", CASES)
def test_classification_never_calls_the_model(text, _layer):
    r = classify_feedback(text)
    assert r.llm_used is False
    assert r.llm_calls == 0
    assert r.layer == "nlp"


def test_most_feedback_has_a_deterministic_fix():
    """Only decoding problems should require the model to be touched."""
    deterministic = [t for t, _ in CASES
                     if classify_feedback(t).result["deterministic_fix_available"]]
    assert len(deterministic) >= len(CASES) - 1


def test_unknown_feedback_falls_back_to_prompt_with_low_confidence():
    r = classify_feedback("the third paragraph feels slightly off somehow")
    assert r.result["primary_layer"] == "PROMPT"
    assert r.confidence <= 0.5, "an unmatched guess must not look confident"


def test_empty_feedback_is_an_error_not_a_guess():
    assert not classify_feedback("   ").ok
