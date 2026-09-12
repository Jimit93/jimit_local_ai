"""Predicted-vs-actual evaluation for generated LinkedIn posts.

This is the QA layer, not a second style engine: every measurement reuses the
primitives that already exist - `nlp.features.extract_features`, the measured
style profile, `style_engine.style_constraints` and `validator.validate_post`.
Nothing here calls a model, so an evaluation is free and repeatable.

The *predicted profile* is what the learned writing architecture says a post on
this topic should look like. It is derived from the corpus (style statistics +
BM25 neighbours + the topic classifier), never from a second LLM pass.

Retrieval here is deliberately lexical (BM25 only). The evaluator has to be able
to run while the generator holds the model in RAM, and loading MiniLM/FAISS a
second time is exactly the kind of cost this project exists to avoid.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from backend.config import REPORTS_DIR, ensure_dirs
from backend.ingestion.linkedin.ingest import writing_corpus
from backend.nlp import features as F
from backend.nlp.keywords import BM25Index, keywords_simple, tokenize
from backend.services.style_engine import load_profile, style_constraints

EVAL_DIR = Path(REPORTS_DIR) / "evaluation"

# Documented weights. Content and factual accuracy carry the most weight because a
# post that reads perfectly but cites none of the supplied evidence fails the job
# it was asked to do; efficiency is scored but deliberately small, since the
# harness benchmark already measures cost in detail.
WEIGHTS = {
    "content": 0.20,
    "style": 0.20,
    "factual": 0.20,
    "structure": 0.15,
    "technical": 0.10,
    "audience": 0.10,
    "efficiency": 0.05,
}

_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")

# The author is looking for work and clients, not recruiting staff. Measured:
# of the 9 corpus posts whose CTA is labelled "hiring", 0 announce a vacancy.
_RECRUITING_RE = re.compile(
    r"\b(?:we(?:'re| are) hiring|i(?:'m| am) hiring|now hiring|hiring for|"
    r"join (?:my|our) team|apply now|send (?:me )?your (?:cv|resume))\b", re.I)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM_RE.findall(text or "")}


def _ratio(a: float, b: float) -> float:
    """1.0 = identical, decaying with relative difference (same shape as the back-test)."""
    if a == 0 and b == 0:
        return 1.0
    m = max(abs(a), abs(b), 1e-9)
    return round(max(0.0, 1.0 - abs(a - b) / m), 4)


def _in_range(v: float, lo: float, hi: float) -> float:
    """1.0 inside the band, decaying outside it relative to the band width."""
    if lo <= v <= hi:
        return 1.0
    width = max(hi - lo, 1e-9)
    d = (lo - v) if v < lo else (v - hi)
    return round(max(0.0, 1.0 - d / width), 4)


# ------------------------------------------------------------------ predicted

def build_predicted_profile(
    topic: str,
    *,
    audience: str = "",
    pillar: str = "",
    facts: str = "",
    length: str = "medium",
    k: int = 6,
) -> dict:
    """What the learned architecture expects a post on this topic to look like."""
    profile = load_profile()
    if not profile:
        return {"error": "no style profile"}
    c = style_constraints(profile)

    posts = writing_corpus(types=("post",), min_words=20)
    texts = [p["text"] for p in posts]

    # Small, representative neighbour sample - lexical, no embeddings.
    bm = BM25Index(texts)
    hits = bm.normalised_search(f"{topic} {pillar}", top_k=k)
    neighbours = [
        {"doc_id": posts[i]["id"], "created_at": posts[i].get("created_at"),
         "bm25": score, "words": len(tokenize(texts[i]))}
        for i, score in hits
    ]
    nb_feats = [F.extract_features(texts[i]) for i, _ in hits]

    # ML: which content pillar the corpus classifier assigns to this topic.
    try:
        from backend.ml.topics import rule_label

        topic_label, topic_conf = rule_label(topic)
    except Exception:
        topic_label, topic_conf = None, 0.0

    L = profile["length"]
    lo, hi = c["target_words"]
    presets = {"short": (0.6, 0.9), "medium": (0.9, 1.3), "long": (1.3, 2.2)}
    f_lo, f_hi = presets.get((length or "medium").lower(), (0.9, 1.3))
    exp_lo, exp_hi = max(30, int(lo * f_lo)), max(60, int(hi * f_hi))

    def _mean(key: str) -> float:
        vals = [f[key] for f in nb_feats] or [0.0]
        return round(sum(vals) / len(vals), 4)

    hook_dist = profile["hooks"]["distribution"]
    expected_hook = (c.get("preferred_hooks") or ["claim"])[0]

    # Numbers the post is expected to be able to cite, from the supplied facts only.
    fact_numbers = sorted(_numbers(facts))

    return {
        "generated_at": time.time(),
        "source": "config/style_profile.json (77 posts) + BM25 neighbours + rule_label",
        "content_pillar": pillar or topic_label or "engineering",
        "content_pillar_ml": {"label": topic_label, "confidence": round(topic_conf, 4)},
        "audience": audience or "AI engineers and technical buyers",
        "expected_length": {"min": exp_lo, "max": exp_hi,
                            "corpus_median": int(L["words"]["median"])},
        "hook": {
            "type": expected_hook,
            "rate_in_corpus": hook_dist.get(expected_hook, 0.0),
            "expected_characteristics": [
                F.__dict__.get("HOOK_HINT", {}).get(expected_hook)
                or "a direct factual statement about the topic",
                f"first line under {int(L['avg_sentence_len']['p75']) + 8} words",
            ],
            "alternatives": list(hook_dist.items())[:3],
        },
        "structure": [
            "hook (first line, standalone)",
            f"{max(1, int(L['paragraphs']['p25']))}-{max(2, int(L['paragraphs']['p75']))} "
            f"short paragraphs",
            "concrete evidence (measured numbers) in the body",
            "closing line: positioning or CTA",
        ],
        "technical_density": {
            "floor": c["min_technical_density"],
            "corpus_median": profile["technical"]["density"]["median"],
            "neighbour_mean": _mean("technical_density"),
        },
        "sentence_length_mean": {
            "corpus_median": L["avg_sentence_len"]["median"],
            "max_allowed": c["max_sentence_words"],
            "neighbour_mean": _mean("avg_sentence_len"),
        },
        "paragraph_length_mean": {
            "target_paragraphs": c["target_paragraphs"],
            "neighbour_mean": _mean("paragraph_count"),
        },
        "cta_type": {
            "expected": bool(c["cta_expected"]),
            "usage_rate": profile["cta"]["usage_rate"],
            "preferred": c.get("preferred_ctas", []),
            "semantics": ("author seeks opportunities and collaboration; "
                          "the 'hiring' label does NOT mean the author is recruiting "
                          "(measured: 0 of 9 such posts announce a vacancy)"),
        },
        "hashtag_count": {"min": c["hashtags"][0], "max": c["hashtags"][1],
                          "usage_rate": profile["hashtags"]["usage_rate"]},
        "emoji_policy": ("sparing, allowed" if c["emoji_allowed"] else "none"),
        "emoji_usage_rate": profile["emoji"]["usage_rate"],
        "expected_numbers": fact_numbers,
        "expected_number_count_min": min(2, len(fact_numbers)),
        "vocabulary": {
            "topic_keywords": keywords_simple(f"{topic} {facts}", top=12),
            "corpus_top_terms": [t for t, _ in profile["vocabulary"]["top_terms"][:12]],
        },
        "style_features": [
            "plain engineer voice, concrete over promotional",
            "no markdown (LinkedIn renders it literally)",
            f"bullets {'expected' if c['bullets_expected'] else 'not expected'}",
            "no generic LinkedIn filler",
        ],
        "banned_phrases": F.GENERIC_PHRASES[:12],
        "retrieved_examples": neighbours,
    }


# ------------------------------------------------------------------- scoring

def _content_score(text: str, predicted: dict, topic: str) -> dict:
    kws = [k for k in predicted.get("vocabulary", {}).get("topic_keywords", [])]
    toks = set(tokenize(text))
    hit = [k for k in kws if k in toks]
    cover = len(hit) / max(1, len(kws))
    return {"score": round(cover, 4), "keywords_expected": kws, "keywords_present": hit}


def _factual_score(text: str, predicted: dict, facts: str, validation: dict) -> dict:
    """Two halves: nothing invented, and the supplied evidence actually used."""
    allowed = _numbers(facts)
    used = _numbers(text)
    invented = sorted(used - allowed)
    cited = sorted(used & allowed)
    grounded = 1.0 if not invented else 0.0

    need = predicted.get("expected_number_count_min", 0)
    if not allowed:
        usage = 1.0                      # no facts supplied: nothing to cite
    elif need == 0:
        usage = 1.0
    else:
        usage = min(1.0, len(cited) / need)
    return {
        "score": round(0.5 * grounded + 0.5 * usage, 4),
        "numbers_grounded": not invented,
        "invented_numbers": invented,
        "facts_available": len(allowed),
        "facts_cited": cited,
        "facts_citation_rate": round(len(cited) / max(1, len(allowed)), 4),
        "validator_said_grounded": validation.get("checks", {}).get("numbers_grounded"),
    }


def _style_score(text: str, predicted: dict) -> dict:
    f = F.extract_features(text)
    exp_len = predicted["expected_length"]
    parts = {
        "length": _in_range(f["word_count"], exp_len["min"], exp_len["max"]),
        "sentence_length": _ratio(f["avg_sentence_len"],
                                  predicted["sentence_length_mean"]["corpus_median"]),
        "paragraphs": _in_range(f["paragraph_count"],
                                *predicted["paragraph_length_mean"]["target_paragraphs"]),
        "technical_density": 1.0 if f["technical_density"] >= predicted[
            "technical_density"]["floor"] else _ratio(
            f["technical_density"], predicted["technical_density"]["floor"]),
        "hook_match": 1.0 if f["hook_type"] == predicted["hook"]["type"] else 0.0,
        "hashtag_behaviour": 1.0 if predicted["hashtag_count"]["min"] <= f[
            "hashtag_count"] <= max(predicted["hashtag_count"]["max"], 3) else 0.0,
        "emoji_behaviour": 1.0 if (predicted["emoji_policy"] != "none"
                                   or f["emoji_count"] == 0) else 0.0,
        "no_generic_phrases": 0.0 if F.generic_phrase_hits(text) else 1.0,
    }
    return {"score": round(sum(parts.values()) / len(parts), 4), "parts": parts,
            "actual": {"words": f["word_count"], "hook": f["hook_type"],
                       "paragraphs": f["paragraph_count"],
                       "avg_sentence_len": f["avg_sentence_len"],
                       "technical_density": f["technical_density"],
                       "hashtags": f["hashtag_count"], "emoji": f["emoji_count"]}}


def _structure_score(text: str, predicted: dict) -> dict:
    f = F.extract_features(text)
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    lo, hi = predicted["paragraph_length_mean"]["target_paragraphs"]
    parts = {
        "has_hook": 1.0 if f["hook_type"] != "none" else 0.0,
        "paragraph_count": _in_range(len(paras), lo, hi),
        "evidence_in_body": 1.0 if _numbers(text) else 0.0,
        "has_closing": 1.0 if len(paras) >= 2 else 0.0,
        "no_markdown": 0.0 if re.search(r"\*\*|^#{1,6}\s", text, re.M) else 1.0,
        "sentence_cap": 1.0 if f["max_sentence_len"] <= predicted[
            "sentence_length_mean"]["max_allowed"] else 0.0,
    }
    return {"score": round(sum(parts.values()) / len(parts), 4), "parts": parts}


def _technical_score(text: str, predicted: dict) -> dict:
    f = F.extract_features(text)
    floor = predicted["technical_density"]["floor"]
    density_ok = min(1.0, f["technical_density"] / max(floor, 1e-9))
    terms = set(F.technical_terms(text))
    expected_terms = {t for t in predicted["vocabulary"]["topic_keywords"]}
    overlap = len(terms & expected_terms) / max(1, min(len(expected_terms), 6))
    return {
        "score": round(min(1.0, 0.6 * density_ok + 0.4 * min(1.0, overlap)), 4),
        "technical_density": f["technical_density"], "floor": floor,
        "technical_terms": sorted(terms)[:12],
    }


def _audience_score(text: str, predicted: dict, audience: str) -> dict:
    """Does the post address the intended reader - and not invert the relationship?"""
    toks = set(tokenize(text))
    aud_terms = [t for t in tokenize(audience) if len(t) > 3]
    addressed = [t for t in aud_terms if t in toks]
    coverage = len(addressed) / max(1, min(len(aud_terms), 5))

    recruiting = bool(_RECRUITING_RE.search(text))
    seeking_audience = bool(re.search(r"\bhiring manager|recruit|client|customer|"
                                      r"companies|employer\b", audience, re.I))
    # Telling hiring managers that YOU are hiring inverts the ask.
    inversion = recruiting and seeking_audience
    parts = {
        "audience_terms_present": round(min(1.0, coverage), 4),
        "no_audience_inversion": 0.0 if inversion else 1.0,
        "cta_appropriate": 0.0 if inversion else 1.0,
    }
    return {"score": round(sum(parts.values()) / len(parts), 4), "parts": parts,
            "recruiting_language": recruiting, "inversion_detected": inversion}


def _efficiency_score(metrics: dict) -> dict:
    calls = metrics.get("llm_calls", 1) or 1
    tokens = metrics.get("tokens", 0)
    # Budget: the design target is ONE call; tokens judged against the observed
    # single-call envelope for this prompt shape (~1000).
    call_part = 1.0 if calls <= 1 else round(max(0.0, 1.0 - 0.5 * (calls - 1)), 4)
    tok_part = round(max(0.0, min(1.0, 1.0 - max(0, tokens - 1000) / 2000)), 4)
    return {"score": round(0.7 * call_part + 0.3 * tok_part, 4),
            "llm_calls": calls, "tokens": tokens,
            "latency_ms": metrics.get("latency_ms")}


def evaluate(
    text: str,
    predicted: dict,
    *,
    topic: str = "",
    audience: str = "",
    facts: str = "",
    validation: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    """Score one generated post against its predicted profile. Deterministic."""
    validation = validation or {}
    metrics = metrics or {}
    comp = {
        "content": _content_score(text, predicted, topic),
        "style": _style_score(text, predicted),
        "factual": _factual_score(text, predicted, facts, validation),
        "structure": _structure_score(text, predicted),
        "technical": _technical_score(text, predicted),
        "audience": _audience_score(text, predicted, audience),
        "efficiency": _efficiency_score(metrics),
    }
    scores = {k: v["score"] for k, v in comp.items()}
    overall = round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 4)
    return {
        "overall_score": overall,
        "scores": scores,
        "weights": WEIGHTS,
        "components": comp,
        "validator_style_match": validation.get("style_match"),
        "validator_status": validation.get("status"),
    }


def save(name: str, payload: dict) -> Path:
    ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = EVAL_DIR / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
