"""Regression: the pasted draft that came back as SUCCESS.

It opened "Today, I am thrilled to announce a game-changing breakthrough that could
revolutionize...", invented a CEO backstory, and stopped at "...solving these
challenges:" - cut off by the token cap. The validator flagged the buzzwords but the
check was not critical, so 8/9 = 0.889 cleared the 0.85 bar and nothing was repaired.
"""
from backend.harness.validator import (
    CRITICAL_CHECKS, STATUS_OK, dangling_ending, validate_post)
from backend.nlp import features as F
from backend.services.linkedin_writer import _finish, _unfinished
from backend.services.style_engine import style_instructions

C = {"target_words": [40, 170], "max_sentence_words": 60, "hashtags": [0, 5],
     "emoji_allowed": True, "min_technical_density": 0.0, "banned_phrases": F.GENERIC_PHRASES}

DRAFT = (
    "Today, I am thrilled to announce a breakthrough that could revolutionize the way "
    "businesses operate. By harnessing the power of AI, organizations can streamline "
    "their processes.\n\n"
    "However, with the advent of AI, this has transformed. I developed an AI agent that "
    "provided insights into my business operations. By asking the AI about common "
    "problems, I discovered that AI could help in solving these challenges:"
)


def test_the_pasted_draft_no_longer_passes():
    v = validate_post(DRAFT, C, examples=[], facts="")
    assert v["status"] != STATUS_OK
    assert v["checks"]["no_generic_phrases"] is False
    assert v["checks"]["complete_ending"] is False


def test_generic_phrases_and_complete_ending_are_critical():
    assert "no_generic_phrases" in CRITICAL_CHECKS
    assert "complete_ending" in CRITICAL_CHECKS


def test_phrases_added_from_the_draft_are_banned():
    for p in ("thrilled to announce", "unprecedented", "with the advent of",
              "play a crucial role", "countless hours", "daunting"):
        assert p in F.GENERIC_PHRASES


def test_phrases_the_author_actually_uses_are_not_banned():
    # Measured in the author's 77 posts: each of these appears at least once.
    for p in ("game-changing", "streamline", "robust", "landscape"):
        assert p not in F.GENERIC_PHRASES


def test_capped_post_is_cut_back_to_its_last_complete_sentence():
    out = _finish(DRAFT, truncated=True)
    assert out.endswith("insights into my business operations.")
    assert dangling_ending(out) is None


def test_hashtag_tail_survives_the_trim():
    out = _finish("One point.\n\nAnother one that got cut off mid\n\n#AI #ML", truncated=True)
    assert out == "One point.\n\n#AI #ML"


def test_author_style_endings_are_left_alone():
    # 56 of the author's 77 posts end without terminal punctuation.
    for ending in ("Check the logs ⬇️", "Full write-up: https://lnkd.in/abc",
                   "Done.\n​Hashtags:\n#AI #ML"):
        assert _finish(ending, truncated=False) == ending
        assert validate_post(ending, C, examples=[], facts="")["checks"]["complete_ending"]


def test_one_word_label_before_hashtags_is_not_a_dangling_colon():
    assert dangling_ending("Shipped it.\nHashtags:\n#AI") is None


def test_capped_post_with_no_complete_sentence_fails():
    text = "I built an agent that routes every task to the cheapest processor and"
    assert _unfinished(_finish(text, truncated=True), truncated=True)
    v = validate_post(text, C, examples=[], facts="", unfinished=True)
    assert v["checks"]["complete_ending"] is False
    assert v["status"] != STATUS_OK


def test_ellipsis_added_by_the_runtime_trim_is_not_an_ending():
    # LLM.generate's completion trim returns "...half a sentence…" when nothing
    # complete exists. On a capped reply that must still count as unfinished.
    cut = "Yes. I built an agent that routes every task to the cheapest processor…"
    assert _finish(cut, truncated=True) == "Yes."
    assert _unfinished("I built an agent that routes every task to the cheapest…", truncated=True)


def test_uncapped_output_is_never_marked_unfinished():
    assert not _unfinished("No terminal punctuation here", truncated=False)


def test_model_is_told_to_finish_inside_its_limit():
    rules = style_instructions(target_words=(40, 120))
    assert any("never stop mid-sentence" in r for r in rules)
