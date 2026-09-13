"""Describe -> Generate -> Test -> Refine, as bounded micro-iterations.

Source idea: "AI App Development Lifecycle & Rapid Iteration" (Category 3 notes).
  - one big prompt is not enough; quality comes from many small feedback passes;
  - granular passes catch drift early instead of letting it compound.

Adapted to a CPU-only machine instead of copied:
  - DESCRIBE  the brief and the deterministic content plan (already in linkedin_writer).
  - GENERATE  ONE model call - the same call one-shot mode makes, with no blanket repair.
  - TEST      the deterministic validator + the predicted-vs-actual evaluator. Free.
  - REFINE    one failing check per iteration, one sentence at a time. Deterministic
              fixers run first and cost nothing; a model fix rewrites a single sentence,
              not the post, and is capped by `max_llm_fixes`.
  - DRIFT     every candidate is re-tested; anything that is not strictly better on
              defects (or equal on defects and no worse on score) is rolled back.

"20x" is kept as a ceiling on micro-iterations, not a target number of model calls:
20 full generations here would cost 5-15 minutes. Whether extra iterations help is
measured by backend/benchmarks/refine_backtest.py, not assumed.
"""
from __future__ import annotations

import re
import time

from backend.config import SETTINGS
from backend.db import query
from backend.harness import prompt_builder as PB
from backend.harness.context_manager import compress
from backend.harness.validator import CRITICAL_CHECKS, STATUS_OK, validate_post
from backend.llm import manager as LLM
from backend.nlp import features as F
from backend.nlp.hashtags import apply_hashtags
from backend.services.linkedin_writer import (
    _TAIL_RE, _enforce_length, _finish, _strip_wrapper, _target_words, _unfinished, generate)
from backend.services.style_engine import load_profile, style_constraints
from backend.tools.deterministic import sentences, word_count

MAX_ITERATIONS = 20
MAX_LLM_FIXES = 3

# Critical checks first; within a tier, the cheapest reliable fixer first.
PRIORITY = ["complete_ending", "no_placeholders", "no_meta_text", "no_markdown",
            "numbers_grounded", "no_generic_phrases", "facts_used", "not_copied",
            "length", "sentence_length", "hashtags", "emoji", "technical_density"]

_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]️?")
_META_RE = re.compile(r"^\s*(?:here(?:'s| is) (?:a|your)|sure[,!]|certainly|i'?ve (?:written|drafted)|"
                      r"as an ai|note:)", re.I)


# ------------------------------------------------------------------ helpers

def _split(text: str) -> tuple[str, str | None]:
    paras = [p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    if len(paras) > 1 and _TAIL_RE.match(paras[-1]):
        return "\n\n".join(paras[:-1]), paras[-1]
    return "\n\n".join(paras), None


def _join(body: str, tail: str | None) -> str:
    body = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", body)).strip()
    return f"{body}\n\n{tail}" if tail else body


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM_RE.findall(text or "")}


def _banned_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in F.GENERIC_PHRASES if p in low]


def _drop_sentence(body: str, sentence: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", body.replace(sentence, "", 1)).strip()


def _complete_sentence(raw: str, stop_reason: str) -> str | None:
    """First line of a micro-fix, only if it is one finished sentence."""
    line = next((ln.strip() for ln in _strip_wrapper(raw).splitlines() if ln.strip()), "")
    line = line.strip("\"“”' ")
    if not line or stop_reason == "length" or not re.search(r"[.!?]$", line):
        return None
    return line


# ------------------------------------------------------------------ fixers
# Each returns (candidate_text | None, note). None means "nothing this fixer can do".

def fix_complete_ending(text, ctx):
    out = _finish(text, truncated=True)
    return (out if out != text else None), "cut back to the last complete sentence"


def fix_placeholders(text, ctx):
    body, tail = _split(text)
    bad = [s for s in sentences(body) if re.search(r"\[[^\]\n]{2,40}\]", s)]
    for s in bad:
        body = _drop_sentence(body, s)
    return (_join(body, tail) if bad else None), f"dropped {len(bad)} sentence(s) with placeholders"


def fix_meta_text(text, ctx):
    body, tail = _split(_strip_wrapper(text))
    first = (sentences(body) or [""])[0]
    if first and _META_RE.match(first):
        body = _drop_sentence(body, first)
    out = _join(body, tail)
    return (out if out != text else None), "removed assistant preamble"


def fix_markdown(text, ctx):
    out = _strip_wrapper(text)
    return (out if out != text else None), "stripped markdown"


def fix_invented_numbers(text, ctx):
    allowed = _numbers(ctx["facts"]) | _numbers(ctx["topic"])
    body, tail = _split(text)
    bad = [s for s in sentences(body) if _numbers(s) - allowed]
    for s in bad:
        body = _drop_sentence(body, s)
    return (_join(body, tail) if bad else None), f"dropped {len(bad)} sentence(s) with unsupported numbers"


def fix_generic_by_dropping(text, ctx):
    """Free first: a hype sentence often carries nothing, so try removing it."""
    body, tail = _split(text)
    target = next((s for s in sentences(body) if _banned_hits(s)), None)
    if not target:
        return None, "no sentence with a banned phrase"
    new_body = _drop_sentence(body, target)
    if word_count(new_body) < ctx["lo"]:
        return None, "dropping it would make the post too short"
    return _join(new_body, tail), f"dropped: '{target[:70]}'"


def fix_generic_by_rewrite(text, ctx):
    body, tail = _split(text)
    target = next((s for s in sentences(body) if _banned_hits(s)), None)
    if not target:
        return None, "no sentence with a banned phrase"
    words = max(8, word_count(target))
    prompt = PB.build_prompt(
        objective="Rewrite one sentence from a LinkedIn post.",
        context=f"Topic of the post: {ctx['topic']}",
        task=f"Sentence: {target}",
        constraints=[
            "Keep its meaning and every fact; add no new claims or numbers.",
            "Do not use any of these words or phrases: " + ", ".join(_banned_hits(target)) + ".",
            "Plain, concrete engineer voice. No hype.",
            f"At most {words} words, written as one complete sentence.",
        ],
        output_format="The rewritten sentence only.",
    )
    r = ctx["llm"](prompt, max_tokens=int(words * 1.6) + 20)
    new = _complete_sentence(r.text, r.stop_reason)
    if not new:
        return None, "rewrite was not one complete sentence"
    return _join(body.replace(target, new, 1), tail), f"rewrote: '{target[:50]}' -> '{new[:50]}'"


def fix_facts_verbatim(text, ctx):
    """Free first: quote the user's own fact sentence - grounded by construction."""
    body, tail = _split(text)
    have = _numbers(body)
    for s in sentences(ctx["facts"]):
        nums = _numbers(s)
        if len(nums) >= 2 and not nums <= have and word_count(s) <= 30:
            s = s if re.search(r"[.!?]$", s) else s + "."
            return _join(f"{body}\n\n{s}", tail), f"added a measured fact verbatim: '{s[:70]}'"
    return None, "no short fact sentence with two numbers"


def fix_facts_by_sentence(text, ctx):
    if not _numbers(ctx["facts"]):
        return None, "no numeric facts supplied"
    body, tail = _split(text)
    prompt = PB.build_prompt(
        objective="Write one sentence for a LinkedIn post that states measured results.",
        context=f"MEASURED FACTS (the only numbers allowed):\n{ctx['facts']}",
        task=f"Topic of the post: {ctx['topic']}",
        constraints=["Use at least two numbers written exactly as they appear in the facts.",
                     "Add no other numbers and no claims beyond the facts.",
                     "At most 30 words, written as one complete sentence."],
        output_format="The sentence only.",
    )
    r = ctx["llm"](prompt, max_tokens=70)
    new = _complete_sentence(r.text, r.stop_reason)
    if not new:
        return None, "sentence was not complete"
    return _join(f"{body}\n\n{new}", tail), f"added: '{new[:70]}'"


def fix_length(text, ctx):
    if word_count(text) <= ctx["hi"]:
        return None, "too short - no grounded way to add content"
    out = _enforce_length(text, ctx["hi"], slack=1.0)
    return (out if out != text else None), "trimmed trailing paragraphs to the word budget"


def fix_long_sentence(text, ctx):
    body, tail = _split(text)
    limit = int(ctx["constraints"].get("max_sentence_words", 40) * 1.5)
    long = max(sentences(body), key=word_count, default="")
    if word_count(long) <= limit:
        return None, "no over-long sentence"
    m = re.search(r";\s+|,\s+(?:and|but|so|which)\s+", long)
    if not m or not (0.25 < m.start() / max(1, len(long)) < 0.75):
        return None, "no clean split point"
    left, right = long[:m.start()].rstrip(",; "), long[m.end():].strip()
    new = f"{left}. {right[:1].upper()}{right[1:]}"
    return _join(body.replace(long, new, 1), tail), "split an over-long sentence"


def fix_hashtags(text, ctx):
    cap = max(ctx["constraints"].get("hashtags", [0, 5])[1], 3)
    if ctx["tags"]:
        out = apply_hashtags(text, ctx["tags"][:cap])
    else:
        body, tail = _split(text)
        tags = re.findall(r"#[A-Za-z]\w*", tail or "")
        out = _join(body, " ".join(tags[:cap]) if tags else None)
    return (out if out != text else None), "set hashtags to the allowed count"


def fix_emoji(text, ctx):
    out = re.sub(r"[ \t]{2,}", " ", _EMOJI_RE.sub("", text)).strip()
    return (out if out != text else None), "removed emoji"


FIXERS: dict[str, list[tuple[str, str, callable]]] = {
    "complete_ending": [("finish", "deterministic", fix_complete_ending)],
    "no_placeholders": [("drop_placeholders", "deterministic", fix_placeholders)],
    "no_meta_text": [("drop_preamble", "deterministic", fix_meta_text)],
    "no_markdown": [("strip_markdown", "deterministic", fix_markdown)],
    "numbers_grounded": [("drop_unsupported_numbers", "deterministic", fix_invented_numbers)],
    "no_generic_phrases": [("drop_hype_sentence", "deterministic", fix_generic_by_dropping),
                           ("rewrite_hype_sentence", "llm", fix_generic_by_rewrite)],
    "facts_used": [("quote_fact", "deterministic", fix_facts_verbatim),
                   ("write_fact_sentence", "llm", fix_facts_by_sentence)],
    "length": [("trim_length", "deterministic", fix_length)],
    "sentence_length": [("split_sentence", "deterministic", fix_long_sentence)],
    "hashtags": [("fix_hashtags", "deterministic", fix_hashtags)],
    "emoji": [("remove_emoji", "deterministic", fix_emoji)],
    # not_copied and technical_density have no fixer: neither can be repaired without
    # writing new content, which is a generation, not a refinement. They are reported.
}


# ------------------------------------------------------------------ loop

def _defects(v: dict, text: str) -> tuple[int, int, int]:
    failed = [k for k, ok in v["checks"].items() if not ok]
    return (sum(1 for k in failed if k in CRITICAL_CHECKS), len(failed), len(_banned_hits(text)))


def refine(
    topic: str,
    *,
    audience: str = "",
    pillar: str = "",
    length: str = "medium",
    cta: str = "",
    facts: str = "",
    temperature: float = 0.75,
    max_iterations: int = MAX_ITERATIONS,
    max_llm_fixes: int = MAX_LLM_FIXES,
) -> dict:
    from backend.benchmarks.evaluation import build_predicted_profile, evaluate

    t0 = time.perf_counter()

    # DESCRIBE + GENERATE: exactly the one-shot first call, minus its blanket repair.
    base = generate(topic, audience=audience, pillar=pillar, length=length, cta=cta,
                    facts=facts, temperature=temperature, repair=False)
    if "error" in base:
        return base

    profile = load_profile()
    constraints = style_constraints(profile)
    lo, hi = _target_words(length, constraints)
    cap = SETTINGS.harness.style_example_max_chars
    examples = []
    for e in base["examples"]:
        rows = query("SELECT text FROM linkedin_documents WHERE id=?", (e["doc_id"],))
        if rows:
            examples.append(compress(rows[0]["text"], cap))

    usage = {"llm_calls": base["metrics"]["llm_calls"], "tokens": base["metrics"]["tokens"]}

    def llm(prompt: str, max_tokens: int):
        r = LLM.generate(prompt, system=PB.SYSTEM_WRITER, max_tokens=max_tokens, temperature=0.4)
        usage["llm_calls"] += 1
        usage["tokens"] += r.total_tokens
        return r

    ctx = {"topic": topic, "facts": facts or "", "lo": lo, "hi": hi, "tags": base.get("hashtags") or [],
           "constraints": constraints, "llm": llm}
    predicted = build_predicted_profile(topic, audience=audience, pillar=pillar, facts=facts,
                                        length=length)

    unfinished0 = _unfinished(base["text"], bool(base["metrics"].get("truncated")))

    def test(text: str, unfinished: bool = False) -> tuple[dict, float]:
        v = validate_post(text, constraints, examples, facts=facts, topic=topic, unfinished=unfinished)
        ev = evaluate(text, predicted, topic=topic, audience=audience, facts=facts, validation=v,
                      metrics={"llm_calls": usage["llm_calls"], "tokens": usage["tokens"]})
        return v, ev["overall_score"]

    text = base["text"]
    v, overall = test(text, unfinished0)
    initial = {"status": v["status"], "style_match": v["style_match"], "overall": overall,
               "issues": list(v["issues"])}
    trace = [{"iteration": 0, "stage": "generate", "check": None, "fixer": "qwen (one call)",
              "kind": "llm", "accepted": True, "note": "first draft",
              "status": v["status"], "style_match": v["style_match"], "overall": overall,
              "issues": list(v["issues"]), "llm_calls": usage["llm_calls"]}]
    curve = [overall]
    llm_fixes = 0
    exhausted: set[tuple[str, str]] = set()      # (check, fixer) pairs that cannot help now
    stopped = "iteration_cap"

    for it in range(1, max_iterations + 1):
        failing = [k for k in PRIORITY if v["checks"].get(k) is False]
        if not failing and v["status"] == STATUS_OK:
            stopped = "passed"
            break
        step = None
        for check in failing:
            for name, kind, fn in FIXERS.get(check, []):
                if (check, name) in exhausted:
                    continue
                if kind == "llm" and llm_fixes >= max_llm_fixes:
                    exhausted.add((check, name))
                    continue
                step = (check, name, kind, fn)
                break
            if step:
                break
        if not step:
            unfixable = [k for k in failing if not FIXERS.get(k)]
            budget_blocked = [k for k in failing
                              if any(kind == "llm" for _, kind, _ in FIXERS.get(k, []))
                              and llm_fixes >= max_llm_fixes]
            stopped = ("llm_budget_exhausted" if budget_blocked else
                       "only_unfixable_issues" if unfixable else "no_fixer_could_improve")
            break

        check, name, kind, fn = step
        calls_before = usage["llm_calls"]
        t_step = time.perf_counter()
        try:
            cand, note = fn(text, ctx)
        except Exception as e:                       # a fixer must never sink the loop
            cand, note = None, f"fixer error: {type(e).__name__}: {e}"
        if kind == "llm" and usage["llm_calls"] > calls_before:
            llm_fixes += 1

        accepted = False
        if cand is None or cand.strip() == text.strip():
            exhausted.add((check, name))
            v_new, overall_new = v, overall
        else:
            # A draft cut off by the token cap stays unfinished until a candidate
            # actually ends on a complete sentence.
            v_new, overall_new = test(cand, unfinished0 and _unfinished(cand, True))
            before, after = _defects(v, text), _defects(v_new, cand)
            accepted = after < before or (after == before and overall_new >= overall)
            if accepted:
                text, v, overall = cand, v_new, overall_new
                if v["checks"].get(check) is not False:
                    exhausted = {p for p in exhausted if p[0] != check}
            else:
                exhausted.add((check, name))          # drift guard: roll back, do not retry

        curve.append(overall)
        trace.append({
            "iteration": it, "stage": "refine", "check": check, "fixer": name, "kind": kind,
            "accepted": accepted, "note": note,
            "status": v["status"], "style_match": v["style_match"], "overall": overall,
            "candidate_status": v_new["status"] if cand else None,
            "candidate_overall": overall_new if cand else None,
            "issues": list(v["issues"]), "llm_calls": usage["llm_calls"],
            "latency_ms": round((time.perf_counter() - t_step) * 1000, 1),
        })

    if trace[-1]["iteration"] < max_iterations and stopped == "iteration_cap":
        stopped = "passed" if v["status"] == STATUS_OK else stopped

    out = dict(base)
    out["text"] = text
    out["validation"] = v
    out["metrics"] = {
        **base["metrics"],
        "llm_calls": usage["llm_calls"],
        "tokens": usage["tokens"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "repaired": False,
    }
    out["refine"] = {
        "stages": ["describe", "generate", "test", "refine"],
        "brief": {"topic": topic, "audience": audience, "pillar": pillar, "length": length,
                  "cta": cta, "facts": facts},
        "iterations": len(trace) - 1,
        "max_iterations": max_iterations,
        "llm_fixes_used": llm_fixes,
        "max_llm_fixes": max_llm_fixes,
        "accepted_steps": sum(1 for t in trace[1:] if t["accepted"]),
        "stopped_because": stopped,
        "initial": initial,
        "final": {"status": v["status"], "style_match": v["style_match"], "overall": overall,
                  "issues": list(v["issues"])},
        "score_curve": curve,
        "trace": trace,
    }
    return out
