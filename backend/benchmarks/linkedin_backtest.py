"""Historical back-test of the LinkedIn writer.

For each held-out post the system sees only a *brief* derived from it (topic label,
keywords, target length) - never the post itself. It then writes a reconstruction,
and the two are compared on style characteristics.

The goal is STYLE REPRODUCTION, not text memorisation: high verbatim overlap is
treated as a failure, not a success.

Leakage control:
  - the style profile is rebuilt excluding every evaluation post,
  - example retrieval excludes them too,
  - the split is deterministic (seeded) so runs are comparable.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from backend.config import REPORTS_DIR, ensure_dirs
from backend.ingestion.linkedin.ingest import writing_corpus
from backend.nlp import features as F
from backend.nlp.keywords import keywords_simple, lexical_overlap, tokenize
from backend.services.linkedin_writer import generate
from backend.services.style_engine import build_profile


def _brief_from(post: dict, corpus_texts: list[str]) -> dict:
    """Everything the generator is allowed to know about a held-out post."""
    from backend.ml.topics import rule_label

    text = post["text"]
    kws = keywords_simple(text, top=6)
    topic_label, _ = rule_label(text)
    feats = F.extract_features(text)
    return {
        "topic": ", ".join(kws[:5]) or (topic_label or "professional update"),
        "topic_label": topic_label,
        "target_words": (max(30, int(feats["word_count"] * 0.8)),
                         max(60, int(feats["word_count"] * 1.25))),
        "keywords": kws,
    }


def _compare(original: str, generated: str) -> dict:
    o, g = F.extract_features(original), F.extract_features(generated)

    def ratio(a: float, b: float) -> float:
        """1.0 = identical, decaying with relative difference."""
        if a == 0 and b == 0:
            return 1.0
        m = max(abs(a), abs(b), 1e-9)
        return round(max(0.0, 1.0 - abs(a - b) / m), 4)

    vocab = lexical_overlap(original, generated)
    o_toks, g_toks = set(tokenize(original)), set(tokenize(generated))

    # Verbatim copying check: longest shared word run.
    ow, gw = original.lower().split(), generated.lower().split()
    longest = 0
    gset = {" ".join(gw[i:i + 6]) for i in range(max(0, len(gw) - 5))}
    for i in range(max(0, len(ow) - 5)):
        if " ".join(ow[i:i + 6]) in gset:
            longest = max(longest, 6)

    return {
        "length_similarity": ratio(o["word_count"], g["word_count"]),
        "sentence_len_similarity": ratio(o["avg_sentence_len"], g["avg_sentence_len"]),
        "paragraph_similarity": ratio(o["paragraph_count"], g["paragraph_count"]),
        "technical_density_similarity": ratio(o["technical_density"], g["technical_density"]),
        "readability_similarity": ratio(o["readability"], g["readability"]),
        "hook_match": o["hook_type"] == g["hook_type"],
        "cta_match": o["has_cta"] == g["has_cta"],
        "cta_type_match": bool(set(o["cta_types"]) & set(g["cta_types"])) or (
            not o["cta_types"] and not g["cta_types"]),
        "hashtag_behaviour_match": (o["hashtag_count"] > 0) == (g["hashtag_count"] > 0),
        "emoji_behaviour_match": (o["emoji_count"] > 0) == (g["emoji_count"] > 0),
        "vocabulary_overlap": vocab,
        "topic_term_recall": round(
            len(o_toks & g_toks) / max(1, len(o_toks)), 4),
        "verbatim_6gram_copied": longest >= 6,
        "original": {"words": o["word_count"], "hook": o["hook_type"],
                     "cta": o["cta_types"], "hashtags": o["hashtag_count"],
                     "tech_density": o["technical_density"]},
        "generated": {"words": g["word_count"], "hook": g["hook_type"],
                      "cta": g["cta_types"], "hashtags": g["hashtag_count"],
                      "tech_density": g["technical_density"]},
    }


def _style_score(c: dict) -> float:
    """Aggregate style-reproduction score, penalising verbatim copying."""
    parts = [
        c["length_similarity"], c["sentence_len_similarity"], c["paragraph_similarity"],
        c["technical_density_similarity"], c["readability_similarity"],
        1.0 if c["hook_match"] else 0.0,
        1.0 if c["cta_match"] else 0.0,
        1.0 if c["hashtag_behaviour_match"] else 0.0,
        1.0 if c["emoji_behaviour_match"] else 0.0,
    ]
    score = sum(parts) / len(parts)
    if c["verbatim_6gram_copied"]:
        score *= 0.5
    return round(score, 4)


def run_backtest(n_eval: int = 6, seed: int = 7, min_words: int = 40,
                 persist: bool = True) -> dict:
    ensure_dirs()
    t0 = time.time()
    posts = [p for p in writing_corpus(types=("post",), min_words=min_words)]
    if len(posts) < n_eval + 10:
        return {"error": f"not enough posts ({len(posts)}) for a {n_eval}-post evaluation"}

    rng = random.Random(seed)
    eval_posts = rng.sample(posts, n_eval)
    eval_ids = {p["id"] for p in eval_posts}

    # Rebuild the profile without the evaluation posts, then restore it afterwards.
    train_profile = build_profile(exclude_doc_ids=eval_ids, persist=False)
    corpus_texts = [p["text"] for p in posts if p["id"] not in eval_ids]

    rows = []
    for p in eval_posts:
        brief = _brief_from(p, corpus_texts)
        out = generate(
            brief["topic"],
            facts="",
            profile_override=train_profile,
            exclude_ids=eval_ids,
            target_words=brief["target_words"],
            repair=False,
            temperature=0.7,
        )
        if "error" in out:
            rows.append({"doc_id": p["id"], "error": out["error"]})
            continue
        cmp = _compare(p["text"], out["text"])
        rows.append({
            "doc_id": p["id"],
            "created_at": p["created_at"],
            "brief": brief,
            "generated": out["text"],
            "original_preview": p["text"][:300],
            "comparison": cmp,
            "style_score": _style_score(cmp),
            "llm_calls": out["metrics"]["llm_calls"],
            "tokens": out["metrics"]["tokens"],
            "latency_ms": out["metrics"]["latency_ms"],
            "example_ids": [e["doc_id"] for e in out["examples"]],
        })

    ok = [r for r in rows if "comparison" in r]
    agg = {}
    if ok:
        keys = ["length_similarity", "sentence_len_similarity", "paragraph_similarity",
                "technical_density_similarity", "readability_similarity",
                "vocabulary_overlap", "topic_term_recall"]
        agg = {k: round(sum(r["comparison"][k] for r in ok) / len(ok), 4) for k in keys}
        agg["hook_match_rate"] = round(
            sum(1 for r in ok if r["comparison"]["hook_match"]) / len(ok), 4)
        agg["cta_match_rate"] = round(
            sum(1 for r in ok if r["comparison"]["cta_match"]) / len(ok), 4)
        agg["hashtag_behaviour_match_rate"] = round(
            sum(1 for r in ok if r["comparison"]["hashtag_behaviour_match"]) / len(ok), 4)
        agg["verbatim_copy_rate"] = round(
            sum(1 for r in ok if r["comparison"]["verbatim_6gram_copied"]) / len(ok), 4)
        agg["mean_style_score"] = round(sum(r["style_score"] for r in ok) / len(ok), 4)
        agg["mean_llm_calls"] = round(sum(r["llm_calls"] for r in ok) / len(ok), 2)
        agg["mean_tokens"] = round(sum(r["tokens"] for r in ok) / len(ok), 1)
        agg["mean_latency_ms"] = round(sum(r["latency_ms"] for r in ok) / len(ok), 1)

    # Any example the generator saw must not be an evaluation post.
    leakage = [r for r in ok if set(r["example_ids"]) & eval_ids]

    report = {
        "seed": seed,
        "n_eval": n_eval,
        "n_train": len(posts) - n_eval,
        "eval_doc_ids": sorted(eval_ids),
        "train_profile_documents": train_profile.get("n_documents"),
        "aggregate": agg,
        "leakage_detected": len(leakage),
        "duration_s": round(time.time() - t0, 1),
        "results": rows,
    }
    if persist:
        Path(REPORTS_DIR / "linkedin_backtest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        Path(REPORTS_DIR / "linkedin_backtest.md").write_text(
            render_md(report), encoding="utf-8")
        build_profile()  # restore the full-corpus profile on disk
    return report


def render_md(rep: dict) -> str:
    a = rep["aggregate"]
    L = ["# LinkedIn style back-test\n",
         f"{rep['n_eval']} held-out posts, {rep['n_train']} posts used for the style profile "
         f"(seed {rep['seed']}). Evaluation posts were excluded from both the profile and "
         f"example retrieval; leakage checks found **{rep['leakage_detected']}** violations.\n",
         "## Aggregate\n",
         "| Measure | Value |", "|---|---|"]
    for k, v in a.items():
        L.append(f"| {k.replace('_', ' ')} | {v} |")
    L += ["\n## Per post\n",
          "| Post date | Style score | Words orig→gen | Hook orig→gen | CTA match | "
          "Vocab overlap | Verbatim copy |", "|---|---|---|---|---|---|---|"]
    for r in rep["results"]:
        if "comparison" not in r:
            L.append(f"| {r.get('doc_id')} | error: {r.get('error')} | | | | | |")
            continue
        c = r["comparison"]
        L.append(
            f"| {(r['created_at'] or '')[:10]} | {r['style_score']} | "
            f"{c['original']['words']}→{c['generated']['words']} | "
            f"{c['original']['hook']}→{c['generated']['hook']} | "
            f"{'yes' if c['cta_match'] else 'no'} | {c['vocabulary_overlap']} | "
            f"{'YES' if c['verbatim_6gram_copied'] else 'no'} |"
        )
    L.append("\n**Reading this table:** vocabulary overlap is expected to be low - the aim is "
             "reproducing structure, rhythm and habits, not wording. A `YES` in the verbatim "
             "column is a failure and halves that post's score.\n")
    return "\n".join(L)
