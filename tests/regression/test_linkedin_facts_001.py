"""Regression: supplied evidence must be used, and its absence must be detectable.

Baseline defect (measured, iteration 0): 18 measured numbers were supplied, the
post cited 0 of them, and the validator returned style_match 1.0 / SUCCESS. The
`numbers_grounded` check passes vacuously for a post containing no numbers, so on
its own it cannot see this failure at all.
"""
from backend.harness import prompt_builder as PB
from backend.harness.validator import CRITICAL_CHECKS, STATUS_OK, validate_post

FACTS = ("LLM calls 23 to 4 (-82.6%). Tokens 8364 to 2214 (-73.5%). "
         "Accuracy 39.1% to 100%. 19 of the 23 tasks used no LLM call.")

CONSTRAINTS = {
    "target_words": [40, 110],
    "max_sentence_words": 40,
    "hashtags": [0, 5],
    "emoji_allowed": True,
    "min_technical_density": 0.02,
    "banned_phrases": [],
}

EVIDENCE_FREE = (
    "I built a routing harness that sends every task to the cheapest processor that "
    "can answer it reliably, making the language model the last resort. It cuts "
    "calls and latency on a CPU-only laptop and keeps the accuracy high."
)
EVIDENCE_CITED = (
    "I built a routing harness that keeps the model out of work cheaper layers can "
    "do. Across 23 benchmark tasks it cut LLM calls by 82.6% and tokens by 73.5%. "
    "It runs offline on a CPU-only laptop."
)


def test_evidence_free_post_is_caught():
    v = validate_post(EVIDENCE_FREE, CONSTRAINTS, examples=[], facts=FACTS)
    assert v["checks"]["facts_used"] is False
    assert v["status"] != STATUS_OK, "an evidence-free proof post must not be SUCCESS"
    assert any("evidence unused" in i for i in v["issues"])


def test_numbers_grounded_alone_would_have_passed_it():
    """Documents exactly why facts_used had to be added."""
    v = validate_post(EVIDENCE_FREE, CONSTRAINTS, examples=[], facts=FACTS)
    assert v["checks"]["numbers_grounded"] is True


def test_post_that_cites_the_evidence_passes():
    v = validate_post(EVIDENCE_CITED, CONSTRAINTS, examples=[], facts=FACTS)
    assert v["checks"]["facts_used"] is True
    assert v["status"] == STATUS_OK
    assert v["issues"] == []


def test_check_is_skipped_when_no_facts_were_supplied():
    v = validate_post(EVIDENCE_FREE, CONSTRAINTS, examples=[], facts="")
    assert "facts_used" not in v["checks"]


def test_facts_used_is_critical_so_the_repair_pass_can_run():
    assert "facts_used" in CRITICAL_CHECKS


def test_prompt_obligates_citing_evidence_only_when_facts_exist():
    with_facts = PB.writer_prompt(topic="t", style_rules=[], examples=[], facts=FACTS,
                                  target_words=(40, 110))
    without = PB.writer_prompt(topic="t", style_rules=[], examples=[], facts="",
                               target_words=(40, 110))
    assert "Include at least two of the measured numbers" in with_facts
    assert "Include at least two of the measured numbers" not in without


def test_repair_prompt_carries_the_facts():
    """A repair told to cite numbers must be shown the numbers."""
    fix = PB.build_prompt(
        objective="Fix specific problems in a LinkedIn post without rewriting its message.",
        context=f"MEASURED FACTS (the only numbers allowed):\n{FACTS}",
        task="Post:\nsome text",
        constraints=["Fix: supplied evidence unused"],
    )
    assert "82.6" in fix and "8364" in fix
