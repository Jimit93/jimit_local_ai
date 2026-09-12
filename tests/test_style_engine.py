import pytest

from backend.harness.validator import validate_post
from backend.services import style_engine as SE


@pytest.fixture(scope="module")
def profile():
    p = SE.load_profile()
    if not p or p.get("n_documents", 0) == 0:
        pytest.skip("no style profile on disk - run ingestion first")
    return p


def test_profile_has_measured_distributions(profile):
    for key in ("length", "structure", "punctuation", "hooks", "cta", "technical",
                "vocabulary", "recurring_phrases"):
        assert key in profile
    w = profile["length"]["words"]
    assert w["n"] > 0 and w["p25"] <= w["median"] <= w["p75"]


def test_hook_distribution_sums_to_one(profile):
    total = sum(profile["hooks"]["distribution"].values())
    assert 0.99 <= total <= 1.01


def test_summary_is_compact_enough_for_a_prompt(profile):
    s = SE.summarize_profile(profile)
    assert 80 < len(s) < 700, "style brief must stay cheap to send"
    assert "words" in s


def test_constraints_are_machine_readable(profile):
    c = SE.style_constraints(profile)
    lo, hi = c["target_words"]
    assert 0 < lo < hi
    assert isinstance(c["banned_phrases"], list) and c["banned_phrases"]
    assert 0.0 <= c["min_technical_density"] <= 1.0


def test_validator_rejects_generic_filler(profile):
    c = SE.style_constraints(profile)
    bad = ("In today's fast-paced world, we must delve into synergy and unlock the power "
           "of paradigm shifts to revolutionize the game-changer landscape forever.")
    v = validate_post(bad, c)
    assert v["checks"]["no_generic_phrases"] is False
    assert v["style_match"] < 1.0
    assert any("generic" in i for i in v["issues"])


def test_validator_rejects_assistant_preamble(profile):
    v = validate_post("Sure! Here's a post about local AI that runs on CPU.",
                      SE.style_constraints(profile))
    assert v["checks"]["no_meta_text"] is False


def test_validator_flags_copied_text(profile):
    src = ("We measured 14 tokens per second on a quantized model running locally on the "
           "CPU with no GPU at all in the machine")
    v = validate_post(src, SE.style_constraints(profile), examples=[src])
    assert v["checks"].get("not_copied") is False


def test_validator_accepts_an_in_style_post(profile):
    c = SE.style_constraints(profile)
    lo, hi = c["target_words"]
    body = ("Most teams assume local inference needs a GPU. We measured 14 tok/s on a "
            "quantized 1.8B model running on CPU alone, with retrieval answering in 25 ms. "
            "The routing layer keeps the model out of arithmetic and database work entirely. "
            "Happy to share the benchmark if useful.")
    words = body.split()
    while len(words) < lo:
        words += words[: max(1, lo - len(words))]
    v = validate_post(" ".join(words[: max(lo, min(hi, len(words)))]), c)
    assert v["style_match"] >= 0.6, v["issues"]


def test_validator_rejects_markdown(profile):
    v = validate_post("**Bold title** then some body text about local inference on CPU.",
                      SE.style_constraints(profile))
    assert v["checks"]["no_markdown"] is False
    assert any("markdown" in i for i in v["issues"])


def test_validator_grounds_numbers_in_supplied_facts(profile):
    c = SE.style_constraints(profile)
    facts = "harness used 4 LLM calls, 73.5% fewer tokens, 14.7 tok/s on CPU"
    ok = validate_post("We measured 4 calls, 73.5% fewer tokens and 14.7 tok/s.", c, facts=facts)
    assert ok["checks"]["numbers_grounded"] is True

    invented = validate_post("We hit 99 calls and a 12x speedup last quarter.", c, facts=facts)
    assert invented["checks"]["numbers_grounded"] is False
    assert any("not present in the supplied facts" in i for i in invented["issues"])


def test_number_grounding_is_skipped_when_no_facts_given(profile):
    v = validate_post("We measured 42 things today.", SE.style_constraints(profile))
    assert "numbers_grounded" not in v["checks"]


def test_fabricated_number_never_passes_as_success(profile):
    """A post can look perfect and still be unusable if it invents a metric."""
    c = SE.style_constraints(profile)
    body = ("We routed most work away from the model this month. The harness handled "
            "the arithmetic, the database counts and the exact search without touching "
            "it at all, which cut the bill by 50 percent across the whole workload and "
            "kept every answer verifiable end to end for the team reviewing it later.")
    v = validate_post(body, c, facts="the harness used 4 LLM calls instead of 23")
    assert v["checks"]["numbers_grounded"] is False
    assert v["status"] != "SUCCESS", "a fabricated number must never be reported as success"
