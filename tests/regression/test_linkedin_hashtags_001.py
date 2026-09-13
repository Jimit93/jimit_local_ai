"""Regression: hashtags are generated from the target audience, deterministically.

User requirement: posts must carry hashtags built from the audience parameter. The
prompt only allowed "at most N hashtags", so the 1.8B model wrote none, and the 7B
model wrote generic ones (#Innovation #TechSolution) that match no audience.
"""
from backend.nlp.hashtags import apply_hashtags, audience_hashtags

PROFILE = {"hashtags": {"top": [["#AI", 5], ["#Automation", 5], ["#GenerativeAI", 5],
                                ["#MachineLearning", 5], ["#OpenToWork", 6],
                                ["#WealthManagement", 3], ["#ArtificialIntelligence", 4]]}}

AUDIENCE = "AI engineering hiring managers and companies that want AI automation built"


def _tokens(tag):
    import re
    return {t.lower() for t in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+", tag)}


def test_every_tag_traces_back_to_an_audience_word():
    r = audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)
    assert r["hashtags"], "an audience was given, so tags must be produced"
    aud = {"ai", "engineering", "hiring", "managers", "automation"}
    for tag in r["hashtags"]:
        toks = _tokens(tag) | ({"ai"} if "artificialintelligence" in tag.lower() else set())
        assert toks & aud, f"{tag} does not come from the audience"


def test_audience_phrases_become_tags():
    tags = audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)["hashtags"]
    assert "#HiringManagers" in tags
    assert "#AIEngineering" in tags


def test_author_tags_unrelated_to_the_audience_are_excluded():
    tags = audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)["hashtags"]
    assert "#WealthManagement" not in tags and "#MachineLearning" not in tags


def test_filler_words_never_become_tags():
    tags = audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)["hashtags"]
    assert not {"#Want", "#Built", "#Companies"} & set(tags)


def test_cap_is_respected():
    assert len(audience_hashtags(AUDIENCE, profile=PROFILE, cap=3)["hashtags"]) <= 3


def test_acronyms_keep_their_case():
    tags = audience_hashtags("RAG developers and CEOs", profile={}, cap=5)["hashtags"]
    assert "#RAGDevelopers" in tags and all("Rag" not in t and "Ceo" not in t for t in tags)


def test_no_audience_means_no_generated_tags():
    assert audience_hashtags("", topic="anything", profile=PROFILE)["hashtags"] == []


def test_deterministic():
    a = audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)["hashtags"]
    assert a == audience_hashtags(AUDIENCE, profile=PROFILE, cap=5)["hashtags"]


def test_model_tags_on_the_last_line_are_replaced():
    out = apply_hashtags("Even on low-end CPUs. #AI #Engineering #TechInnovation", ["#AIEngineering"])
    assert out == "Even on low-end CPUs.\n\n#AIEngineering"


def test_model_tags_on_their_own_lines_are_replaced():
    assert apply_hashtags("Body.\n\n#Innovation\n#Tech", ["#AI"]) == "Body.\n\n#AI"


def test_numbered_references_are_not_mistaken_for_tags():
    assert apply_hashtags("Ranked #1 in the test.", ["#AI"]) == "Ranked #1 in the test.\n\n#AI"


def test_writer_only_generates_tags_when_an_audience_is_given():
    import inspect

    from backend.services import linkedin_writer

    src = inspect.getsource(linkedin_writer.generate)
    assert "if (previous or not audience)" in src
