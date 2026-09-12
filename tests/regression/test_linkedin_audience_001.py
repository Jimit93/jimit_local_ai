"""Regression: the CTA must not invert the author's intent.

Baseline defect (measured, iteration 0): the plan forced CTA `hiring` even though
`cta_expected` is False at 20.8% usage, and the raw detector label was passed to
the model, which wrote "Hiring for more engineers" to an audience of hiring
managers. Measured evidence: of the 9 corpus posts whose CTA matches the `hiring`
pattern, 0 announce a vacancy.
"""
import re

import pytest

from backend.harness import prompt_builder as PB
from backend.services.linkedin_writer import build_plan
from backend.services.style_engine import CTA_GUIDANCE, cta_guidance

RECRUITING = re.compile(
    r"\b(?:we(?:'re| are) hiring|i(?:'m| am) hiring|now hiring|hiring for|"
    r"join (?:my|our) team)\b", re.I)


def test_plan_does_not_force_a_cta_the_author_rarely_uses():
    constraints = {"preferred_ctas": ["hiring", "dm"], "cta_expected": False}
    plan = build_plan("a topic", constraints, [], "")
    assert plan["cta"] is None


def test_plan_uses_the_preferred_cta_when_the_author_does_use_one():
    constraints = {"preferred_ctas": ["dm"], "cta_expected": True}
    assert build_plan("a topic", constraints, [], "")["cta"] == "dm"


def test_an_explicit_user_choice_always_wins():
    constraints = {"preferred_ctas": ["dm"], "cta_expected": False}
    assert build_plan("a topic", constraints, [], "hiring")["cta"] == "hiring"


def test_hiring_label_is_translated_to_seeking_not_recruiting():
    g = cta_guidance("hiring")
    assert g != "hiring", "the raw detector label must never reach the model"
    assert not RECRUITING.search(g), f"guidance still reads as recruiting: {g}"
    assert "open to" in g.lower()


@pytest.mark.parametrize("label", sorted(CTA_GUIDANCE))
def test_every_cta_label_has_guidance_that_is_not_the_bare_label(label):
    assert cta_guidance(label) != label


def test_prompt_renders_guidance_text_not_the_internal_label():
    prompt = PB.writer_prompt(
        topic="t", style_rules=["Length: 40-100 words."], examples=[],
        cta=cta_guidance("hiring"), target_words=(40, 100),
    )
    assert "call to action of type: hiring" not in prompt
    assert "Close the post with a line saying you are open to opportunities" in prompt


def test_no_cta_constraint_is_emitted_when_the_plan_has_no_cta():
    prompt = PB.writer_prompt(
        topic="t", style_rules=["Length: 40-100 words."], examples=[],
        cta=cta_guidance(None), target_words=(40, 100),
    )
    assert "Close the post with" not in prompt
