"""Regression: the Describe -> Generate -> Test -> Refine loop, with a scripted model.

The real model is replaced by a fake so the loop's mechanics - cheapest fixer first,
drift guard, LLM budget, iteration cap, complete endings - are tested exactly and fast.
"""
import pytest

from backend.llm.base import LLMResult
from backend.services import refine_loop as RL
from backend.services.style_engine import load_profile

pytestmark = pytest.mark.skipif(load_profile() is None, reason="no style profile - run ingestion")

HYPE_AND_CUT = (
    "I built a local agent that routes each task to the cheapest processor that can answer it. "
    "It keeps the language model as the last resort instead of the default path. "
    "Deterministic tools, NLP and retrieval answer most requests before the model loads.\n\n"
    "By harnessing the power of AI, it will revolutionize every workflow you have.\n\n"
    "Here is what the harness checks before any model call runs in the pipeline:"
)


def _base(text, truncated=False, tags=None):
    return {
        "text": text, "validation": {}, "plan": {}, "style_brief": "", "examples": [],
        "hashtags": tags or [],
        "metrics": {"llm_calls": 1, "tokens": 500, "prompt_tokens": 400, "prompt_chars": 1,
                    "latency_ms": 1.0, "model": "fake", "runtime": "fake", "repaired": False,
                    "truncated": truncated, "source_count": 0},
    }


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, prompt, **kw):
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "Plain sentence."
        return LLMResult(text=text, prompt_tokens=50, completion_tokens=20, latency_ms=1.0,
                         model="fake", runtime="fake", stop_reason="stop")


@pytest.fixture
def run(monkeypatch):
    def _run(base, replies=(), **kw):
        fake = FakeLLM(replies)
        monkeypatch.setattr(RL, "generate", lambda *a, **k: base)
        monkeypatch.setattr(RL.LLM, "generate", fake)
        out = RL.refine("routing harness for a local AI agent", audience="AI engineers", **kw)
        return out, fake
    return _run


def test_deterministic_fixers_clear_hype_and_cut_ending_with_no_model_call(run):
    out, fake = run(_base(HYPE_AND_CUT, truncated=True))
    r = out["refine"]
    assert fake.calls == 0, "both defects have free fixers - the model must not be called"
    assert out["validation"]["checks"]["complete_ending"] is True
    assert out["validation"]["checks"]["no_generic_phrases"] is True
    assert "harnessing the power" not in out["text"]
    assert not out["text"].rstrip().endswith(":")
    assert r["initial"]["status"] != "SUCCESS"
    assert out["metrics"]["llm_calls"] == 1


def test_free_fixer_refuses_to_drop_below_the_word_floor_and_model_rewrite_runs(run):
    # 44 words after the cut; dropping the 13-word hype sentence would leave 31, below
    # the 38-word floor - so the deterministic fixer must decline and the rewrite runs.
    short_cut = HYPE_AND_CUT.replace(
        " Deterministic tools, NLP and retrieval answer most requests before the model loads.", "")
    out, fake = run(_base(short_cut, truncated=True), replies=["It works on every workflow."])
    steps = {t["fixer"]: t for t in out["refine"]["trace"][1:]}
    assert steps["drop_hype_sentence"]["accepted"] is False
    assert "too short" in steps["drop_hype_sentence"]["note"]
    assert fake.calls == 1


def test_every_refine_step_is_recorded_as_a_micro_iteration(run):
    out, _ = run(_base(HYPE_AND_CUT, truncated=True))
    trace = out["refine"]["trace"]
    assert trace[0]["stage"] == "generate"
    assert all(t["stage"] == "refine" for t in trace[1:])
    assert len(out["refine"]["score_curve"]) == len(trace)
    assert out["refine"]["stages"] == ["describe", "generate", "test", "refine"]


def test_drift_guard_rolls_back_a_fix_that_makes_things_worse(run):
    # Too short to drop the hype sentence, so the model is asked to rewrite it - and the
    # scripted rewrite introduces a NEW banned phrase plus an invented number.
    short = "By harnessing the power of AI I built a local agent that routes tasks cheaply."
    out, fake = run(_base(short), replies=["It is a game-changer that saves 97% of costs."])
    steps = [t for t in out["refine"]["trace"] if t["fixer"] == "rewrite_hype_sentence"]
    assert steps and steps[0]["accepted"] is False
    assert "97%" not in out["text"]


def test_llm_budget_is_never_exceeded(run):
    short = "By harnessing the power of AI I built a local agent that routes tasks cheaply."
    bad = ["Still harnessing the power of AI here."] * 10
    out, fake = run(_base(short), replies=bad, max_llm_fixes=2)
    assert out["refine"]["llm_fixes_used"] <= 2
    assert fake.calls <= 2


def test_iteration_cap_is_a_hard_ceiling(run):
    out, _ = run(_base(HYPE_AND_CUT, truncated=True), max_iterations=1)
    assert out["refine"]["iterations"] <= 1


def test_incomplete_model_rewrite_is_rejected(run):
    short = "By harnessing the power of AI I built a local agent that routes tasks cheaply."
    out, _ = run(_base(short), replies=["I built a local agent that routes tasks and"])
    assert "routes tasks and" not in out["text"].replace("routes tasks cheaply", "")


def test_missing_evidence_is_fixed_by_quoting_the_facts_verbatim_first(run):
    facts = "LLM calls fell from 23 to 4. Tokens fell from 8364 to 2214."
    text = ("I built a local agent that routes each task to the cheapest processor first. "
            "The language model only runs when nothing cheaper can answer the request.")
    out, fake = run(_base(text), facts=facts)
    assert out["validation"]["checks"]["facts_used"] is True
    assert "LLM calls fell from 23 to 4." in out["text"]
    assert fake.calls == 0


def test_audience_hashtags_survive_refinement(run):
    out, _ = run(_base(HYPE_AND_CUT + "\n\n#AIEngineers", truncated=True, tags=["#AIEngineers"]))
    assert out["text"].rstrip().endswith("#AIEngineers")
