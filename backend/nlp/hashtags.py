"""Hashtags derived from the target audience. Deterministic NLP, no model call.

Why not leave them to the model: the prompt only allows "at most N hashtags", so the
model is free to write none (the 1.8B model did exactly that) or generic tags that
match no one (#Innovation #TechSolution on a post aimed at AI hiring managers).
Hashtags decide who sees a post, so they are built from the audience the user named.

Candidates come from two places:
  1. tags composed from the audience text - content words and adjacent word pairs,
     e.g. "AI engineering hiring managers" -> #AIEngineering, #HiringManagers;
  2. the author's own historical hashtags from the style profile - their voice.
Scoring: 2 per distinct audience term a tag covers, 1 per extra topic/pillar term,
plus a small prior when the author has used the tag before. A tag must cover at
least one audience term to be eligible, so every tag traces back to the audience.
"""
from __future__ import annotations

import re

from backend.nlp.features import STOPWORDS

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+")
_TRAILING_TAGS_RE = re.compile(r"(?:[ \t]*\n*[ \t]*#[A-Za-z]\w*)+\s*$")

# Filler that appears in audience descriptions but makes a poor hashtag.
_WEAK = {
    "want", "wants", "wanting", "built", "build", "building", "looking", "need",
    "needs", "people", "company", "companies", "someone", "anyone", "everyone",
    "like", "get", "getting", "using", "use", "make", "new", "etc", "also",
}

_SPECIAL_CASE = {
    "ai": "AI", "ml": "ML", "llm": "LLM", "llms": "LLMs", "rag": "RAG", "nlp": "NLP",
    "cpu": "CPU", "gpu": "GPU", "api": "API", "apis": "APIs", "saas": "SaaS",
    "b2b": "B2B", "b2c": "B2C", "ceo": "CEO", "ceos": "CEOs", "cto": "CTO",
    "cfo": "CFO", "hr": "HR", "it": "IT", "ui": "UI", "ux": "UX", "qa": "QA",
    "genai": "GenAI", "devops": "DevOps", "mlops": "MLOps", "iot": "IoT",
}

# Multi-word spellings folded to the short form, so #ArtificialIntelligence matches an
# audience that says "AI" - without counting as two separate hits.
_FOLD = [(("artificial", "intelligence"), "ai"), (("machine", "learning"), "ml"),
         (("large", "language", "models"), "llms"), (("large", "language", "model"), "llm"),
         (("natural", "language", "processing"), "nlp")]

AUTHOR_PRIOR = 0.5
AUDIENCE_WEIGHT = 2
TOPIC_WEIGHT = 1


def _fold(tokens: list[str]) -> set[str]:
    out, i = [], 0
    while i < len(tokens):
        for seq, short in _FOLD:
            if tuple(tokens[i:i + len(seq)]) == seq:
                out.append(short)
                i += len(seq)
                break
        else:
            out.append(tokens[i])
            i += 1
    return set(out)


def _content_words(text: str) -> list[str | None]:
    """Words in order; stopwords and filler become None so adjacency is preserved."""
    out: list[str | None] = []
    for w in _WORD_RE.findall(text or ""):
        low = w.lower()
        out.append(None if (low in STOPWORDS or low in _WEAK or len(low) < 2) else w)
    return out


def _case(word: str) -> str:
    low = word.lower()
    if low in _SPECIAL_CASE:
        return _SPECIAL_CASE[low]
    if word.isupper() and len(word) <= 5:
        return word
    return word[:1].upper() + word[1:].lower()


def _split_tag(tag: str) -> list[str]:
    return [t.lower() for t in _CAMEL_RE.findall(tag.lstrip("#"))]


def _terms(text: str) -> set[str]:
    return _fold([w.lower() for w in _content_words(text) if w])


def audience_hashtags(
    audience: str,
    *,
    topic: str = "",
    pillar: str = "",
    profile: dict | None = None,
    cap: int = 5,
) -> dict:
    """Rank hashtags for this audience. Returns the tags plus how each was scored."""
    aud_terms = _terms(audience)
    if not aud_terms:
        return {"hashtags": [], "scored": [], "reason": "no audience terms"}
    extra_terms = _terms(f"{topic} {pillar}") - aud_terms

    cands: dict[str, dict] = {}

    words = _content_words(audience)
    for i, w in enumerate(words):
        if not w:
            continue
        cands.setdefault("#" + _case(w), {"tokens": [w.lower()], "origin": "audience"})
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt:
            cands.setdefault("#" + _case(w) + _case(nxt),
                             {"tokens": [w.lower(), nxt.lower()], "origin": "audience"})

    author_counts: dict[str, int] = {}
    for tag, count in ((profile or {}).get("hashtags", {}).get("top") or []):
        author_counts[tag.lower()] = count
        cands.setdefault(tag, {"tokens": _split_tag(tag), "origin": "author"})

    scored = []
    for tag, c in cands.items():
        toks = _fold(c["tokens"])
        aud_hits = toks & aud_terms
        if not aud_hits:
            continue
        top_hits = toks & extra_terms
        used = author_counts.get(tag.lower(), 0)
        score = AUDIENCE_WEIGHT * len(aud_hits) + TOPIC_WEIGHT * len(top_hits) + (
            AUTHOR_PRIOR if used else 0.0)
        scored.append({"tag": tag, "score": score, "audience_terms": sorted(aud_hits),
                       "topic_terms": sorted(top_hits), "author_uses": used,
                       "origin": "author" if used else c["origin"], "tokens": toks})

    scored.sort(key=lambda s: (-s["score"], -s["author_uses"], s["tag"].lower()))

    chosen: list[dict] = []
    covered: set[str] = set()
    for s in scored:
        if len(chosen) >= cap:
            break
        # A composed tag that only repeats words already covered adds noise, not reach.
        # The author's own tags are kept even then: those are the ones their followers use.
        if s["origin"] == "audience" and s["tokens"] <= covered:
            continue
        if any(s["tokens"] == c["tokens"] for c in chosen):
            continue
        chosen.append(s)
        covered |= s["tokens"]

    for s in scored:
        s["tokens"] = sorted(s["tokens"])
    return {
        "hashtags": [s["tag"] for s in chosen],
        "scored": [{k: v for k, v in s.items()} for s in scored[:15]],
        "audience_terms": sorted(aud_terms),
    }


def apply_hashtags(text: str, tags: list[str]) -> str:
    """Replace whatever hashtags trail the post with the audience-derived line."""
    if not tags:
        return text
    body = _TRAILING_TAGS_RE.sub("", (text or "").rstrip()).rstrip()
    return f"{body}\n\n{' '.join(tags)}"
