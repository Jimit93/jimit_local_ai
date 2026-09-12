"""Writing-style engine.

Learns *how* the user writes, not *what* they claimed in 2024. Style knowledge and
factual knowledge are kept strictly apart: the profile stores distributions,
structures and phrasing habits; historical claims stay in `linkedin_documents` and
are only ever surfaced as dated, attributed evidence.
"""
from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

from backend.config import CONFIG_DIR, ensure_dirs
from backend.db import execute, executemany, query, remember
from backend.ingestion.linkedin.ingest import writing_corpus
from backend.nlp import features as F
from backend.nlp.keywords import phrase_frequencies, vocabulary_profile

STYLE_PROFILE_PATH = CONFIG_DIR / "style_profile.json"


def _dist(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(stats.fmean(s), 2),
        "median": round(stats.median(s), 2),
        "p25": round(s[int(len(s) * 0.25)], 2),
        "p75": round(s[min(len(s) - 1, int(len(s) * 0.75))], 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "stdev": round(stats.pstdev(s), 2) if len(s) > 1 else 0.0,
    }


def _rate(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else 0.0


def compute_document_features(docs: list[dict]) -> list[dict]:
    """Extract features for each document. Deterministic; no model involved."""
    out = []
    for d in docs:
        f = F.extract_features(d["text"])
        out.append({"doc": d, "features": f})
    return out


def persist_document_features(items: list[dict]) -> int:
    """Write per-document features into `style_features` (upsert by doc_id)."""
    rows = []
    for it in items:
        d, f = it["doc"], it["features"]
        rows.append(
            (
                d["id"], f["char_count"], f["word_count"], f["sentence_count"],
                f["paragraph_count"], f["avg_sentence_len"], f["avg_word_len"],
                f["hashtag_count"], f["emoji_count"], f["url_count"],
                f["question_count"], f["exclamation_count"], f["number_count"],
                f["bullet_count"], f["caps_word_count"], f["technical_density"],
                f["readability"], f["hook_type"], ",".join(f["cta_types"]),
                int(f["has_cta"]),
                json.dumps(
                    {
                        "structure": f["structure"],
                        "technical_terms": f["technical_terms"],
                        "repetition": f["repetition"],
                        "generic_phrases": f["generic_phrases"],
                        "hook_text": f["hook_text"][:300],
                        "type_token_ratio": f["type_token_ratio"],
                    },
                    ensure_ascii=False,
                ),
            )
        )
    executemany(
        "INSERT INTO style_features(doc_id,char_count,word_count,sentence_count,"
        "paragraph_count,avg_sentence_len,avg_word_len,hashtag_count,emoji_count,url_count,"
        "question_count,exclamation_count,number_count,bullet_count,caps_word_count,"
        "technical_density,readability,hook,cta,has_cta,features_json) "
        "VALUES(" + ",".join("?" * 21) + ") "
        "ON CONFLICT(doc_id) DO UPDATE SET word_count=excluded.word_count,"
        "char_count=excluded.char_count,sentence_count=excluded.sentence_count,"
        "paragraph_count=excluded.paragraph_count,avg_sentence_len=excluded.avg_sentence_len,"
        "avg_word_len=excluded.avg_word_len,hashtag_count=excluded.hashtag_count,"
        "emoji_count=excluded.emoji_count,url_count=excluded.url_count,"
        "question_count=excluded.question_count,exclamation_count=excluded.exclamation_count,"
        "number_count=excluded.number_count,bullet_count=excluded.bullet_count,"
        "caps_word_count=excluded.caps_word_count,technical_density=excluded.technical_density,"
        "readability=excluded.readability,hook=excluded.hook,cta=excluded.cta,"
        "has_cta=excluded.has_cta,features_json=excluded.features_json",
        rows,
    )
    return len(rows)


def build_profile(
    doc_types: tuple[str, ...] = ("post",),
    exclude_doc_ids: set[int] | None = None,
    persist: bool = True,
    min_words: int = 20,
) -> dict:
    """Aggregate a style profile.

    `exclude_doc_ids` exists so the back-test can build a profile that has never
    seen the evaluation posts (no leakage).
    """
    docs = [
        d for d in writing_corpus(types=doc_types, min_words=min_words)
        if not exclude_doc_ids or d["id"] not in exclude_doc_ids
    ]
    if not docs:
        return {"error": "no documents", "n_documents": 0}

    items = compute_document_features(docs)
    feats = [it["features"] for it in items]
    texts = [it["doc"]["text"] for it in items]

    hooks = [f["hook_type"] for f in feats]
    ctas = [c for f in feats for c in f["cta_types"]]
    struct = [f["structure"] for f in feats]

    def count_rates(values: list[str]) -> dict:
        total = len(values) or 1
        return {
            k: round(v / total, 4)
            for k, v in sorted(
                {v: values.count(v) for v in set(values)}.items(), key=lambda kv: -kv[1]
            )
        }

    profile = {
        "generated_at": time.time(),
        "n_documents": len(docs),
        "doc_types": list(doc_types),
        "date_range": {
            "earliest": min((d["created_at"] or "" for d in docs), default=None),
            "latest": max((d["created_at"] or "" for d in docs), default=None),
        },
        "length": {
            "words": _dist([f["word_count"] for f in feats]),
            "chars": _dist([f["char_count"] for f in feats]),
            "sentences": _dist([f["sentence_count"] for f in feats]),
            "paragraphs": _dist([f["paragraph_count"] for f in feats]),
            "avg_sentence_len": _dist([f["avg_sentence_len"] for f in feats]),
            "avg_word_len": _dist([f["avg_word_len"] for f in feats]),
            "avg_para_words": _dist([s["avg_para_words"] for s in struct]),
        },
        "structure": {
            "bullet_usage_rate": _rate([s["has_bullets"] for s in struct]),
            "bullets_per_post": _dist([s["bullet_count"] for s in struct]),
            "lines_per_post": _dist([s["line_count"] for s in struct]),
            "one_line_paragraph_rate": round(
                sum(s["one_line_paragraphs"] for s in struct)
                / max(1, sum(s["paragraph_count"] for s in struct)), 4
            ),
            "ends_with_hashtags_rate": _rate([s["ends_with_hashtags"] for s in struct]),
        },
        "punctuation": {
            "questions_per_post": _dist([f["question_count"] for f in feats]),
            "exclamations_per_post": _dist([f["exclamation_count"] for f in feats]),
            "colons_per_post": _dist([f["colon_count"] for f in feats]),
            "dashes_per_post": _dist([f["dash_count"] for f in feats]),
            "caps_words_per_post": _dist([f["caps_word_count"] for f in feats]),
        },
        "emoji": {
            "usage_rate": _rate([f["emoji_count"] > 0 for f in feats]),
            "per_post": _dist([f["emoji_count"] for f in feats]),
        },
        "hashtags": {
            "usage_rate": _rate([f["hashtag_count"] > 0 for f in feats]),
            "per_post": _dist([f["hashtag_count"] for f in feats]),
            "top": [
                (h, sum(1 for f in feats if h in f["hashtags"]))
                for h in sorted(
                    {h for f in feats for h in f["hashtags"]},
                    key=lambda x: -sum(1 for f in feats if x in f["hashtags"]),
                )[:20]
            ],
        },
        "hooks": {
            "distribution": count_rates(hooks),
            "hook_words": _dist([f["hook_words"] for f in feats]),
            "examples": {
                h: [f["hook_text"][:160] for f in feats if f["hook_type"] == h][:3]
                for h in set(hooks)
            },
        },
        "cta": {
            "usage_rate": _rate([f["has_cta"] for f in feats]),
            "distribution": count_rates(ctas),
            "examples": [f["cta_text"][:180] for f in feats if f["has_cta"]][:5],
        },
        "technical": {
            "density": _dist([f["technical_density"] for f in feats]),
            "metrics_per_post": _dist([f["number_count"] for f in feats]),
            "top_terms": [
                (t, sum(1 for f in feats if t in f["technical_terms"]))
                for t in sorted(
                    {t for f in feats for t in f["technical_terms"]},
                    key=lambda x: -sum(1 for f in feats if x in f["technical_terms"]),
                )[:30]
            ],
        },
        "readability": _dist([f["readability"] for f in feats]),
        "repetition": _dist([f["repetition"] for f in feats]),
        "type_token_ratio": _dist([f["type_token_ratio"] for f in feats]),
        "vocabulary": vocabulary_profile(texts),
        "recurring_phrases": phrase_frequencies(texts, top=30, min_count=3),
        "generic_phrase_hits": sorted(
            {p for f in feats for p in f["generic_phrases"]}
        ),
        "url_usage_rate": _rate([f["url_count"] > 0 for f in feats]),
    }

    if persist:
        persist_document_features(items)
        ensure_dirs()
        Path(STYLE_PROFILE_PATH).write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        remember("style", "profile_summary", summarize_profile(profile),
                 source="derived", confidence=1.0)
    return profile


def load_profile() -> dict | None:
    if Path(STYLE_PROFILE_PATH).exists():
        return json.loads(Path(STYLE_PROFILE_PATH).read_text(encoding="utf-8"))
    return None


def summarize_profile(profile: dict | None = None) -> str:
    """Compact, token-cheap style brief - this is what goes into a prompt."""
    p = profile or load_profile()
    if not p or p.get("n_documents", 0) == 0:
        return "No style profile available."
    L, S, H, C, T = (p["length"], p["structure"], p["hooks"], p["cta"], p["technical"])
    top_hooks = list(H["distribution"].items())[:3]
    top_ctas = list(C["distribution"].items())[:2]
    parts = [
        f"Typical length {int(L['words']['p25'])}-{int(L['words']['p75'])} words "
        f"(median {int(L['words']['median'])}).",
        f"{int(L['paragraphs']['median'])} short paragraphs, "
        f"~{L['avg_sentence_len']['median']} words/sentence.",
        f"Openings: {', '.join(f'{k} {int(v*100)}%' for k, v in top_hooks)}.",
        f"Bullets in {int(S['bullet_usage_rate']*100)}% of posts.",
        f"CTA in {int(C['usage_rate']*100)}% of posts"
        + (f" ({', '.join(k for k, _ in top_ctas)}).") if C["usage_rate"] else ".",
        f"Hashtags in {int(p['hashtags']['usage_rate']*100)}% of posts "
        f"(median {int(p['hashtags']['per_post']['median'])}).",
        f"Emoji in {int(p['emoji']['usage_rate']*100)}% of posts.",
        f"Technical density median {T['density']['median']}, "
        f"{int(T['metrics_per_post']['median'])} numbers per post.",
    ]
    return " ".join(parts)


HOOK_GUIDANCE = {
    "claim": "a direct factual statement about the topic",
    "metric": "a concrete number or measured result",
    "question": "a short question",
    "what_if": "a 'what if' framing",
    "story": "a first-person moment in time",
    "announcement": "what you built or shipped",
    "milestone": "the milestone you reached",
    "contrarian": "a widely held belief you disagree with",
    "listicle": "a numbered promise, e.g. '3 things I learned'",
    "problem": "the problem you are solving",
    "label": "a two or three word label, a colon, then the point",
    "greeting": "a short greeting to your network",
    "gratitude": "a thank-you",
    "transformation": "a before and after framing",
}


def style_instructions(profile: dict | None = None, target_words: tuple[int, int] | None = None
                       ) -> list[str]:
    """Imperative writing rules for the prompt.

    Deliberately free of distribution statistics: a small model treats percentages
    in its context as content to reproduce, and starts writing "59% claim rate"
    into the post. Numbers here are instructions, not measurements.
    """
    p = profile or load_profile()
    if not p or p.get("n_documents", 0) == 0:
        return ["Write a short, concrete, professional post."]
    c = style_constraints(p)
    L = p["length"]
    lo, hi = target_words or c["target_words"]
    paras = max(2, int(L["paragraphs"]["median"]))
    hook = (c.get("preferred_hooks") or ["claim"])[0]
    rules = [
        f"Length: {lo}-{hi} words.",
        f"Use about {paras} short paragraphs, one or two sentences each.",
        f"Average sentence length around {int(L['avg_sentence_len']['median'])} words; "
        f"never exceed {c['max_sentence_words']}.",
        f"Open with {HOOK_GUIDANCE.get(hook, 'a direct statement')}.",
        "Write plainly and concretely, the way an engineer describes what they built.",
    ]
    rules.append("Use a short bulleted list." if c["bullets_expected"]
                 else "Do not use bullet points unless the topic is a list.")
    rules.append("Emoji are acceptable but sparing." if c["emoji_allowed"]
                 else "Do not use emoji.")
    hmax = c["hashtags"][1]
    rules.append(f"End with at most {hmax} hashtags on their own line." if hmax
                 else "Do not use hashtags.")
    if c["cta_expected"] and c.get("preferred_ctas"):
        rules.append(f"Close with a {c['preferred_ctas'][0]} call to action.")
    return rules


def style_constraints(profile: dict | None = None) -> dict:
    """Machine-readable targets used by the generator and the validator."""
    p = profile or load_profile()
    if not p or p.get("n_documents", 0) == 0:
        return {}
    L = p["length"]
    return {
        "target_words": [int(L["words"]["p25"]), int(L["words"]["p75"])],
        "target_paragraphs": [
            max(1, int(L["paragraphs"]["p25"])), max(2, int(L["paragraphs"]["p75"]))
        ],
        "max_sentence_words": max(18, int(L["avg_sentence_len"]["p75"]) + 8),
        "hashtags": [0, max(0, int(p["hashtags"]["per_post"]["p75"]))],
        "emoji_allowed": p["emoji"]["usage_rate"] > 0.25,
        "bullets_expected": p["structure"]["bullet_usage_rate"] > 0.4,
        "cta_expected": p["cta"]["usage_rate"] > 0.4,
        "preferred_hooks": list(p["hooks"]["distribution"].keys())[:3],
        "preferred_ctas": list(p["cta"]["distribution"].keys())[:2],
        "min_technical_density": round(max(0.02, p["technical"]["density"]["p25"]), 4),
        "banned_phrases": F.GENERIC_PHRASES,
    }


def profile_stats() -> dict:
    """Cheap counts for the Knowledge page."""
    rows = query("SELECT doc_type, COUNT(*) c FROM linkedin_documents GROUP BY doc_type")
    return {r["doc_type"]: r["c"] for r in rows}
