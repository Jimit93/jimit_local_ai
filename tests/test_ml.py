import pytest

from backend.harness import task_classifier as TC
from backend.ml import topics as T


def test_rule_label_confident_cases():
    lab, conf = T.rule_label(
        "break of structure below the supply zone, waiting for the retest before entry "
        "with proper risk management")
    assert lab == "trading" and conf > 0.6
    lab, conf = T.rule_label(
        "built a RAG pipeline with local embeddings and a quantized model for offline "
        "inference in python")
    assert lab == "ai_engineering" and conf > 0.6


def test_rule_label_abstains_when_unsure():
    lab, conf = T.rule_label("Had a good day today, thanks everyone")
    assert lab is None and conf == 0.0


def test_classifier_returns_envelope_with_confidence():
    r = T.classify("the four hour chart shows a clear break of structure")
    assert r.ok
    assert 0.0 <= r.confidence <= 1.0
    assert r.layer in ("nlp", "ml")
    assert r.llm_used is False
    assert r.latency_ms >= 0


def test_trained_model_generalises_beyond_the_rules():
    """A phrasing the keyword rules abstain on must still be classified by the model."""
    clf = T.classifier()
    if clf.pipe is None and not clf.load():
        pytest.skip("topic model not trained yet")
    text = "my EURUSD short from the zone hit target"
    assert T.rule_label(text)[0] is None
    label, conf = clf.predict(text)
    assert label in T.LABELS and 0.0 < conf <= 1.0


def test_intent_rules_do_not_collide():
    cases = {
        "What is 25 * 40?": "deterministic",
        "Classify this LinkedIn post as AI or trading": "classification",
        "Write a LinkedIn post about local inference": "creative_generation",
        "Find all LinkedIn posts mentioning RAG": "retrieval",
        "How many posts do I have?": "structured_data",
        "top 5 hashtags i use": "ranking",
        'Extract the URLs from: "see https://example.com/post here"': "text_processing",
        "How many words are in this text: hello world": "text_processing",
    }
    for text, expected in cases.items():
        r = TC.classify_task(text)
        assert r.result == expected, f"{text!r} -> {r.result} (expected {expected})"


def test_intent_classifier_reports_honest_confidence():
    r = TC.classify_task("something entirely unrelated to any seeded example xyzzy")
    assert 0.0 <= r.confidence <= 1.0
    assert r.result in TC.CATEGORIES
