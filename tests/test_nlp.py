from backend.nlp import features as F
from backend.nlp import keywords as K


def test_hook_detection():
    assert F.detect_hook("What if you could run this offline?")["hook_type"] == "what_if"
    assert F.detect_hook("Is local inference actually viable?")["hook_type"] == "question"
    assert F.detect_hook("I built an AI accountant.")["hook_type"] == "announcement"
    assert F.detect_hook("Most people think local models are toys.")["hook_type"] == "contrarian"
    assert F.detect_hook("Hi everyone! I am looking for a role.")["hook_type"] == "greeting"
    assert F.detect_hook("Thank you, #SkillUp 🙌")["hook_type"] == "gratitude"
    assert F.detect_hook("Headline: Building agents got interesting.")["hook_type"] == "label"
    assert F.detect_hook("5 lessons from shipping local AI")["hook_type"] == "listicle"


def test_leading_emoji_does_not_hide_the_hook():
    h = F.detect_hook("🚀 AI is changing how brokers talk to clients")
    assert h["hook_type"] == "claim"
    assert h["starts_with_emoji"] is True


def test_cta_detection():
    assert F.detect_cta("Great stuff.\nDrop a comment if useful.")["cta_types"] == ["comment"]
    assert "dm" in F.detect_cta("Body text here.\nDM me for the repo.")["cta_types"]
    assert F.detect_cta("Just a plain statement about work.")["has_cta"] is False


def test_technical_density_orders_correctly():
    tech = "Quantized GGUF inference on CPU at 14 tok/s with FAISS retrieval and BM25."
    plain = "I had a really nice weekend with my family and the weather was lovely."
    assert F.technical_density(tech) > F.technical_density(plain)
    assert 0.0 <= F.technical_density(plain) <= 1.0


def test_generic_phrase_detection():
    hits = F.generic_phrase_hits("In today's fast-paced world we must delve into synergy.")
    assert "in today's fast-paced world" in hits and "delve into" in hits
    assert F.generic_phrase_hits("We measured 14 tok/s on CPU.") == []


def test_readability_and_repetition():
    assert F.flesch_reading_ease("The cat sat on the mat.") > 60
    assert F.repetition_score("test test test test test test test test test") > 0.5


def test_structure_signature():
    s = F.structure_signature("Line one.\n\n- a\n- b\n- c\n\n#ai #local")
    assert s["has_bullets"] and s["bullet_count"] == 3
    assert s["ends_with_hashtags"] is True


def test_bm25_ranks_relevant_document_first():
    docs = [
        "supply zone and demand zone trading setups on the four hour chart",
        "local llm inference with quantized models on cpu",
        "a post about hiring and open roles",
    ]
    idx = K.BM25Index(docs)
    assert idx.search("quantized cpu inference", top_k=1)[0][0] == 1
    assert idx.search("supply zone chart", top_k=1)[0][0] == 0


def test_bm25_normalised_scores_bounded():
    idx = K.BM25Index(["alpha beta gamma", "beta gamma delta"])
    hits = idx.normalised_search("beta", top_k=2)
    assert hits and all(0.0 <= s <= 1.0 for _, s in hits)


def test_keywords_and_overlap():
    kws = K.keywords_simple("local quantized inference on cpu with local models", top=4)
    assert "local" in kws
    assert K.lexical_overlap("local quantized inference", "local quantized inference") == 1.0
    assert K.lexical_overlap("local quantized inference", "weekend family weather") == 0.0
    # single characters are intentionally filtered out as noise
    assert K.tokenize("a b c") == []


def test_phrase_frequencies_prefers_longer_phrases():
    texts = ["risk management matters"] * 4 + ["other text"]
    phrases = [p["phrase"] for p in K.phrase_frequencies(texts, min_count=3)]
    assert "risk management matters" in phrases
    assert "risk management" not in phrases  # subsumed by the longer phrase


def test_vocabulary_profile():
    vp = K.vocabulary_profile(["building local agents", "building local tools"])
    assert vp["total_tokens"] > 0 and vp["vocabulary_size"] > 0
    assert any(t[0] == "building" for t in vp["top_terms"])
