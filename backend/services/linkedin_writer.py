"""LinkedIn generation pipeline.

Deterministic work first, one LLM call, deterministic QC after:

    style profile (measured)  ->  example retrieval (hybrid, capped at 1-3)
      ->  content plan (rules) ->  ONE Qwen call  ->  style validation
      ->  optional single repair call, only when validation actually failed

Historical posts are supplied as *style* evidence. Facts come only from what the
user supplies in this request, so the model cannot recycle a stale claim from 2024.
"""
from __future__ import annotations

import re
import time

from backend.config import SETTINGS
from backend.harness import prompt_builder as PB
from backend.harness.validator import STATUS_OK, validate_post
from backend.llm import manager as LLM
from backend.retrieval.index import index
from backend.services.style_engine import (
    load_profile,
    style_constraints,
    style_instructions,
    summarize_profile,
)

LENGTH_PRESETS = {"short": (0.6, 0.9), "medium": (0.9, 1.3), "long": (1.3, 2.2)}


def _target_words(length: str, constraints: dict) -> tuple[int, int]:
    lo, hi = constraints.get("target_words", [50, 120])
    f_lo, f_hi = LENGTH_PRESETS.get((length or "medium").lower(), (0.9, 1.3))
    return max(30, int(lo * f_lo)), max(60, int(hi * f_hi))


def select_examples(topic: str, k: int | None = None, topic_filter: str | None = None,
                    exclude_ids: set[int] | None = None) -> list[dict]:
    """A small, representative set - never hundreds of posts.

    `exclude_ids` keeps the back-test honest: an evaluation post must never be
    retrieved as an example of how to write itself.
    """
    k = k or SETTINGS.harness.max_style_examples
    hits = index().search(topic, top_k=max(k * 3, 6), doc_type="post", topic=topic_filter,
                          exclude_ids=exclude_ids)
    seen: list[str] = []
    out: list[dict] = []
    from backend.harness.context_manager import compress
    from backend.nlp.keywords import lexical_overlap

    cap = SETTINGS.harness.style_example_max_chars
    for h in hits:
        # Retrieval matches a chunk, but style lives in the whole post: its opening,
        # its paragraph rhythm and its sign-off. Always take the full document here.
        full = h.doc_text or h.text
        if any(lexical_overlap(full, s) > 0.7 for s in seen):
            continue
        seen.append(full)
        out.append({
            "doc_id": h.doc_id, "score": round(h.score, 4), "created_at": h.created_at,
            # Capped: the opening and rhythm carry the style, the tail rarely adds more.
            "text": compress(full, cap), "url": h.url,
        })
        if len(out) >= k:
            break
    return out


def build_plan(topic: str, constraints: dict, examples: list[dict], cta: str | None) -> dict:
    """Deterministic content plan - structure decided by rules, not by the model."""
    preferred_hooks = constraints.get("preferred_hooks") or ["claim"]
    hook = preferred_hooks[0]
    if any(ch.isdigit() for ch in topic) and "metric" in preferred_hooks:
        hook = "metric"
    return {
        "hook_type": hook,
        "paragraphs": constraints.get("target_paragraphs", [2, 4]),
        "bullets": bool(constraints.get("bullets_expected")),
        "cta": cta or (constraints.get("preferred_ctas") or [None])[0],
        "hashtags": constraints.get("hashtags", [0, 3])[1],
        "emoji_allowed": constraints.get("emoji_allowed", False),
    }


def generate(
    topic: str,
    *,
    audience: str = "",
    pillar: str = "",
    length: str = "medium",
    cta: str = "",
    facts: str = "",
    n_examples: int | None = None,
    temperature: float = 0.75,
    repair: bool = True,
    previous: str | None = None,
    instruction: str | None = None,
    profile_override: dict | None = None,
    exclude_ids: set[int] | None = None,
    target_words: tuple[int, int] | None = None,
) -> dict:
    """Generate (or improve) one post. Returns the text plus full execution metadata."""
    t0 = time.perf_counter()
    profile = profile_override or load_profile()
    if not profile:
        return {"error": "No style profile. Run ingestion + style build first."}
    constraints = style_constraints(profile)
    brief = summarize_profile(profile)
    examples = select_examples(topic, k=n_examples, exclude_ids=exclude_ids)
    plan = build_plan(topic, constraints, examples, cta)
    lo, hi = target_words or _target_words(length, constraints)

    rules = style_instructions(profile, target_words=(lo, hi))

    if previous and instruction:
        prompt = PB.build_prompt(
            objective="Revise one LinkedIn post while keeping the author's voice.",
            task=f"Revision instruction: {instruction}\n\nCurrent post:\n{previous}",
            constraints=rules,
            output_format="The revised post text only. No title and no commentary.",
        )
    else:
        prompt = PB.writer_prompt(
            topic=topic,
            style_rules=rules,
            examples=[e["text"] for e in examples],
            facts=facts,
            audience=audience,
            pillar=pillar,
            cta=plan["cta"] or "",
            target_words=(lo, hi),
        )

    calls, tokens = 0, 0
    # ~1.35 tokens per English word, plus slack for hashtags: a tight cap is the
    # only thing that reliably stops a small model from running long.
    r = LLM.generate(
        prompt, system=PB.SYSTEM_WRITER, max_tokens=int(hi * 1.6) + 40,
        temperature=temperature,
    )
    calls += 1
    tokens += r.total_tokens
    text = _enforce_length(_strip_wrapper(r.text), hi)
    verdict = validate_post(text, constraints, [e["text"] for e in examples],
                            facts=facts, topic=topic)

    repaired = False
    if repair and verdict["status"] != STATUS_OK and verdict["issues"]:
        # One targeted repair call, listing only what actually failed.
        fix_prompt = PB.build_prompt(
            objective="Fix specific problems in a LinkedIn post without rewriting its message.",
            task=f"Post:\n{text}",
            constraints=[f"Fix: {i}" for i in verdict["issues"]] + [
                f"Keep length between {lo} and {hi} words.",
                "Keep the same facts and the same message.",
            ],
            output_format="The corrected post text only.",
        )
        r2 = LLM.generate(fix_prompt, system=PB.SYSTEM_WRITER,
                          max_tokens=int(hi * 1.6) + 40, temperature=0.4)
        calls += 1
        tokens += r2.total_tokens
        cand = _enforce_length(_strip_wrapper(r2.text), hi)
        v2 = validate_post(cand, constraints, [e["text"] for e in examples],
                           facts=facts, topic=topic)
        if v2["style_match"] >= verdict["style_match"]:
            text, verdict, repaired = cand, v2, True

    return {
        "text": text,
        "validation": verdict,
        "plan": plan,
        "style_brief": brief,
        "examples": [
            {"doc_id": e["doc_id"], "created_at": e["created_at"], "score": e["score"],
             "preview": e["text"][:180]} for e in examples
        ],
        "metrics": {
            "llm_calls": calls,
            "tokens": tokens,
            "prompt_tokens": r.prompt_tokens,
            "prompt_chars": len(prompt),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": r.model,
            "runtime": r.runtime,
            "repaired": repaired,
            "source_count": len(examples),
        },
    }


_TAIL_RE = re.compile(r"^\s*(?:#\w+[\s,]*)+$")


def _enforce_length(text: str, hi: int, slack: float = 1.25) -> str:
    """Trim an over-long post the way an editor would: whole paragraphs off the end.

    Small models routinely run 2x over the word budget. Dropping trailing paragraphs
    is deterministic, keeps the opening (which carries the hook) and preserves a
    hashtag or CTA tail line.
    """
    from backend.tools.deterministic import word_count

    limit = int(hi * slack)
    if word_count(text) <= limit:
        return text
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    tail = paras[-1] if len(paras) > 1 and _TAIL_RE.match(paras[-1]) else None
    body = paras[:-1] if tail else paras
    tail_words = word_count(tail) if tail else 0

    kept: list[str] = []
    used = 0
    for p in body:
        w = word_count(p)
        if kept and used + w + tail_words > limit:
            break
        kept.append(p)
        used += w
    if not kept:
        kept = [body[0]]
    if word_count(" ".join(kept)) + tail_words > limit:
        # A single paragraph is still over budget: cut it at a sentence boundary.
        from backend.tools.deterministic import sentences

        out, n = [], 0
        for s in sentences(" ".join(kept)):
            sw = word_count(s)
            if out and n + sw + tail_words > limit:
                break
            out.append(s)
            n += sw
        kept = [" ".join(out)]
    if tail:
        kept.append(tail)
    return "\n\n".join(kept).strip()


_LABEL_RE = re.compile(
    r"^(?:title|post|linkedin post|draft|greeting|intro|introduction|body|caption|"
    r"claim|hook|opening|cta|call to action|hashtags|technical details|fact|facts|"
    r"key features|output)\s*:\s*",
    re.I,
)


def _strip_wrapper(text: str) -> str:
    """Strip the scaffolding smaller models add around the actual post.

    Code fences, surrounding quotes, assistant preambles, and the title/label lines
    that a 1.8B model in particular likes to prepend. "Headline:" is deliberately
    *not* stripped - that is one of the author's own opening patterns.
    """
    t = (text or "").strip()
    # LinkedIn renders markdown literally, so bold/heading markers are never wanted.
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    if len(t) > 40 and t[0] in "\"“'" and t[-1] in "\"”'":
        t = t[1:-1].strip()
    for lead in ("Here's the post:", "Here is the post:", "Sure!", "Certainly!",
                 "Here's a LinkedIn post:", "Of course!"):
        if t.lower().startswith(lead.lower()):
            t = t[len(lead):].strip()

    lines = [ln for ln in t.split("\n")]
    # A first line that is entirely quoted, with real content after it, is a title.
    while len(lines) > 1:
        first = lines[0].strip()
        quoted_title = (len(first) > 10 and first[0] in "\"“" and first[-1] in "\"”"
                        and not first[1:-1].rstrip().endswith((".", "!", "?")))
        if quoted_title:
            lines = lines[1:]
            continue
        stripped = _LABEL_RE.sub("", first)
        if stripped != first:
            lines[0] = stripped.strip()
            if not lines[0]:
                lines = lines[1:]
            continue
        break
    return "\n".join(lines).strip()
