"""The routing harness.

    classify -> route -> execute -> verify -> escalate only if needed -> return

Each category has an ordered chain of processors, cheapest first. A processor
returns a confidence; if it clears the threshold for its layer the chain stops
there and no LLM is touched. `mode="baseline"` bypasses everything and sends the
raw task to the LLM - that is what the benchmark compares against.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from backend.common import ProcessorResult
from backend.config import SETTINGS
from backend.harness import metrics
from backend.harness.context_manager import build_context, estimate_tokens, extract_payload
from backend.harness.prompt_builder import SYSTEM_ANALYST, SYSTEM_BASE, build_prompt
from backend.harness.task_classifier import classify_task
from backend.harness.validator import (
    STATUS_FAIL,
    STATUS_LOW,
    STATUS_OK,
    validate_generic,
    validate_numeric,
)
from backend.llm import manager as LLM
from backend.nlp import features as NF
from backend.nlp.keywords import phrase_frequencies
from backend.tools import deterministic as D
from backend.tools.registry import call as tool_call

LAYER_THRESHOLD = {
    "deterministic": SETTINGS.harness.deterministic_min_conf,
    "nlp": SETTINGS.harness.nlp_min_conf,
    "ml": SETTINGS.harness.ml_min_conf,
    "retrieval": SETTINGS.harness.retrieval_min_conf,
    "llm": 0.0,
}

Processor = Callable[[str, dict], ProcessorResult]


# --------------------------------------------------------------- processors

def p_calculator(task: str, ctx: dict) -> ProcessorResult:
    if not D.looks_like_math(task):
        return ProcessorResult(None, 0.0, "calculator", 0.0, "deterministic",
                               error="not an arithmetic expression")
    res = tool_call("calculate", expression=task)
    if res.ok:
        res.meta["answer"] = f"{res.result['value']}"
    return res


def p_datetime(task: str, ctx: dict) -> ProcessorResult:
    t0 = time.perf_counter()
    m = re.search(r"(\d{4}-\d{2}-\d{2}).{0,20}?(\d{4}-\d{2}-\d{2})", task)
    if m:
        r = tool_call("date_diff", a=m.group(1), b=m.group(2))
        if r.ok:
            r.meta["answer"] = f"{r.result['days']} days"
        return r
    if re.search(r"\b(?:date|time|today|now)\b", task, re.I):
        r = tool_call("now")
        if r.ok:
            r.meta["answer"] = f"{r.result['date']} ({r.result['weekday']}) {r.result['time']} UTC"
        return r
    return ProcessorResult(None, 0.0, "datetime", (time.perf_counter() - t0) * 1000,
                           "deterministic", error="no date in task")


def p_text_stats(task: str, ctx: dict) -> ProcessorResult:
    """Counting/extraction over user-supplied text. Needs a payload to work on."""
    t0 = time.perf_counter()
    payload = ctx.get("payload") or extract_payload(task)
    if not payload:
        return ProcessorResult(None, 0.0, "text_stats", (time.perf_counter() - t0) * 1000,
                               "nlp", error="no text supplied to analyse")
    s = D.text_stats(payload)
    low = task.lower()
    answer = None
    if "url" in low or "link" in low:
        answer = ", ".join(s["urls"]) or "none"
    elif "hashtag" in low:
        answer = ", ".join(s["hashtags"]) or "none"
    elif "mention" in low:
        answer = str(s["mention_count"])
    elif "character" in low or "char count" in low:
        answer = str(s["char_count"])
    elif "sentence" in low:
        answer = str(s["sentence_count"])
    elif "readab" in low or "reading level" in low:
        answer = str(NF.flesch_reading_ease(payload))
    elif "word" in low:
        answer = str(s["word_count"])
    return ProcessorResult(
        result={"answer": answer, "stats": s}, confidence=0.98 if answer else 0.8,
        processor="nlp_text_stats", latency_ms=(time.perf_counter() - t0) * 1000, layer="nlp",
        meta={"answer": answer or json.dumps(
            {k: v for k, v in s.items() if not isinstance(v, list)})},
    )


def p_topic_classify(task: str, ctx: dict) -> ProcessorResult:
    from backend.ml.topics import classify

    payload = ctx.get("payload") or extract_payload(task) or task
    r = classify(payload)
    if r.ok and r.result:
        r.meta["answer"] = r.result
    return r


def p_keyword_search(task: str, ctx: dict) -> ProcessorResult:
    """Exact search. No embeddings, no model - this is the cheap path."""
    from backend.retrieval.index import index

    t0 = time.perf_counter()
    term = _extract_search_term(task)
    if not term:
        return ProcessorResult(None, 0.0, "keyword_search", (time.perf_counter() - t0) * 1000,
                               "deterministic", error="no search term found")
    hits = index().keyword_search(term, top_k=20)
    # "How many documents contain X" is answered exactly by an exact search, and zero
    # is a correct answer - escalating to a fuzzy search there would answer a different
    # question. The suggestion is surfaced instead, not acted on.
    return ProcessorResult(
        result={"term": term, "count": len(hits), "hits": [h.to_dict(220) for h in hits[:8]],
                "suggest_semantic": not hits},
        confidence=0.95, processor="exact_keyword_search",
        latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
        meta={"answer": f"{len(hits)} document(s) contain '{term}'"
                        + ("" if hits else " (no exact match - try a semantic search)")},
    )


def p_semantic_search(task: str, ctx: dict) -> ProcessorResult:
    from backend.retrieval.index import retrieve

    q = _strip_search_verbs(task)
    r = retrieve(q, top_k=5)
    if r.ok:
        # The answer has to carry the retrieved content, not just a count - the count
        # alone is useless to the user and unverifiable by the benchmark.
        lines = [
            f"{i + 1}. [{h.get('created_at') or 'undated'} · {h.get('doc_type')} · "
            f"score {h.get('score')}] {h.get('text', '').splitlines()[0][:160]}"
            for i, h in enumerate(r.result)
        ]
        r.meta["answer"] = (
            f"{len(r.result)} semantically related document(s):\n" + "\n".join(lines)
            if lines else "No related documents found."
        )
        r.confidence = max(r.confidence, 0.4 if r.result else 0.0)
    return r


def p_structured_data(task: str, ctx: dict) -> ProcessorResult:
    """Answer corpus questions with SQL, not with a model."""
    t0 = time.perf_counter()
    low = task.lower()
    try:
        if re.search(r"\bper (?:topic|category)\b|\bby topic\b", low):
            rows = D.sql_query(
                "SELECT topic, COUNT(*) n FROM topics GROUP BY topic ORDER BY n DESC")
            ans = ", ".join(f"{r['topic']}: {r['n']}" for r in rows)
        elif re.search(r"\bby (?:type|document type)\b|\bwhat is in (?:my|the) (?:database|"
                       r"knowledge base)\b|\btotals\b", low):
            rows = D.sql_query(
                "SELECT doc_type, COUNT(*) n FROM linkedin_documents GROUP BY doc_type "
                "ORDER BY n DESC")
            ans = ", ".join(f"{r['doc_type']}: {r['n']}" for r in rows)
        elif "embedding" in low:
            rows = D.sql_query("SELECT COUNT(*) n FROM chunk_embeddings")
            ans = f"{rows[0]['n']} chunk embeddings"
        else:
            dt = _doc_type_in(low)
            if dt:
                rows = D.sql_query(
                    "SELECT COUNT(*) n FROM linkedin_documents WHERE doc_type=?", (dt,))
                ans = f"{rows[0]['n']} {dt}s"
            else:
                rows = D.sql_query("SELECT COUNT(*) n FROM linkedin_documents")
                ans = f"{rows[0]['n']} documents"
        return ProcessorResult(
            result={"rows": rows, "answer": ans}, confidence=0.97,
            processor="sqlite_query", latency_ms=(time.perf_counter() - t0) * 1000,
            layer="deterministic", meta={"answer": ans},
        )
    except Exception as e:
        return ProcessorResult(None, 0.0, "sqlite_query", (time.perf_counter() - t0) * 1000,
                               "deterministic", error=str(e))


def p_ranking(task: str, ctx: dict) -> ProcessorResult:
    """Top-N questions answered from measured statistics."""
    from backend.ingestion.linkedin.ingest import writing_corpus
    from backend.services.style_engine import load_profile

    t0 = time.perf_counter()
    low = task.lower()
    n = int((re.search(r"top\s+(\d+)", low) or re.search(r"\b(\d+)\b", low) or [0, "5"])[1])
    n = max(1, min(n, 25))
    prof = load_profile()
    if not prof:
        return ProcessorResult(None, 0.0, "ranking", (time.perf_counter() - t0) * 1000,
                               "nlp", error="no style profile")
    if "hashtag" in low:
        items = prof["hashtags"]["top"][:n]
    elif "phrase" in low:
        items = [(p["phrase"], p["count"]) for p in prof["recurring_phrases"][:n]]
    elif "verb" in low:
        items = prof["vocabulary"]["top_verbs"][:n]
    elif "hook" in low or "opening" in low:
        items = list(prof["hooks"]["distribution"].items())[:n]
    elif "technical" in low or "term" in low:
        items = prof["technical"]["top_terms"][:n]
    elif "topic" in low:
        from backend.ml.topics import topic_counts

        items = list(topic_counts().items())[:n]
    else:
        items = prof["vocabulary"]["top_terms"][:n]
    if not items:
        texts = [d["text"] for d in writing_corpus()]
        items = [(p["phrase"], p["count"]) for p in phrase_frequencies(texts, top=n)]
    ans = ", ".join(f"{k} ({v})" for k, v in items)
    return ProcessorResult(
        result={"items": items, "answer": ans}, confidence=0.93, processor="corpus_statistics",
        latency_ms=(time.perf_counter() - t0) * 1000, layer="nlp", meta={"answer": ans},
    )


def p_linkedin_writer(task: str, ctx: dict) -> ProcessorResult:
    from backend.services.linkedin_writer import generate

    t0 = time.perf_counter()
    topic = _strip_write_verbs(task)
    out = generate(topic, **{k: v for k, v in (ctx.get("writer_options") or {}).items()})
    if "error" in out:
        return ProcessorResult(None, 0.0, "linkedin_writer",
                               (time.perf_counter() - t0) * 1000, "hybrid", error=out["error"])
    m = out["metrics"]
    return ProcessorResult(
        result=out, confidence=out["validation"]["style_match"],
        processor="linkedin_writer(rag+qwen+qc)",
        latency_ms=(time.perf_counter() - t0) * 1000, layer="hybrid",
        llm_used=True, llm_calls=m["llm_calls"], tokens=m["tokens"],
        meta={"answer": out["text"], "validation": out["validation"]},
    )


def p_llm(task: str, ctx: dict) -> ProcessorResult:
    """Final fallback. Retrieval-grounded when the question is about the corpus."""
    t0 = time.perf_counter()
    blocks: list[tuple[str, str]] = []
    grounded = False
    if ctx.get("ground", True) and _mentions_corpus(task):
        from backend.retrieval.index import index as ridx

        hits = ridx().search(_strip_search_verbs(task), top_k=3)
        if hits:
            grounded = True
            blocks.append((
                "Excerpts from the author's own posts (dated evidence, may be out of date)",
                "\n\n".join(f"[{h.created_at or 'n/a'}] {h.text[:600]}" for h in hits),
            ))
    if ctx.get("stats_block"):
        blocks.append(("Pre-computed statistics (authoritative)", ctx["stats_block"]))
    payload = ctx.get("payload")
    if payload:
        blocks.append(("Text supplied by the user", payload))

    context, acct = build_context(blocks)
    prompt = build_prompt(
        objective="Answer the user's request accurately and concisely.",
        task=task,
        context=context,
        constraints=["Do not invent facts, numbers or dates.",
                     "If the context is insufficient, say what is missing."],
    )
    system = SYSTEM_ANALYST if ctx.get("stats_block") else SYSTEM_BASE
    try:
        r = LLM.generate(prompt, system=system,
                         max_tokens=ctx.get("max_tokens", 400),
                         temperature=ctx.get("temperature", 0.4))
    except Exception as e:
        return ProcessorResult(None, 0.0, "qwen_llm", (time.perf_counter() - t0) * 1000,
                               "llm", error=f"{type(e).__name__}: {e}")
    return ProcessorResult(
        result=r.text, confidence=0.6, processor=f"qwen_llm({r.model})",
        latency_ms=(time.perf_counter() - t0) * 1000, layer="llm", llm_used=True,
        llm_calls=1, tokens=r.total_tokens,
        meta={"answer": r.text, "grounded": grounded, "context": acct,
              "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens},
    )


# ----------------------------------------------------------------- helpers

_SEARCH_TERM_RE = re.compile(
    r"(?:mentioning|containing|mentions?|that say|with the word|including the word|"
    r"about|for)\s+[\"']?([\w\s\-+#.]{2,40}?)[\"']?\s*$", re.I)


def _extract_search_term(task: str) -> str | None:
    m = _SEARCH_TERM_RE.search(task.strip().rstrip("?."))
    if m:
        return m.group(1).strip()
    q = re.search(r"[\"']([^\"']{2,40})[\"']", task)
    return q.group(1).strip() if q else None


def _strip_search_verbs(task: str) -> str:
    return re.sub(
        r"^\s*(?:find|show|list|search|get|which|what)\b.{0,30}?\b(?:posts?|comments?|"
        r"documents?|writing|content)\b\s*(?:that are |that |about |related to |on )?",
        "", task.strip(), flags=re.I).strip(" ?.") or task


def _strip_write_verbs(task: str) -> str:
    return re.sub(
        r"^\s*(?:please\s+)?(?:write|draft|compose|generate|create)\s+(?:me\s+)?(?:a|an|one)?\s*"
        r"(?:short|long|brief)?\s*(?:linkedin\s+)?(?:post|update|article)?\s*(?:about|on|for)?\s*",
        "", task.strip(), flags=re.I).strip() or task


def _doc_type_in(low: str) -> str | None:
    for t in ("post", "comment", "reaction", "message", "vote", "skill", "position"):
        if re.search(rf"\b{t}s?\b", low):
            return t
    return None


def _mentions_corpus(task: str) -> bool:
    return bool(re.search(r"\b(?:my|i)\b.{0,20}\b(?:posts?|writing|style|content|comments?)\b|"
                          r"\bmy linkedin\b", task, re.I))


# ------------------------------------------------------------------ chains

CHAINS: dict[str, list[Processor]] = {
    "deterministic": [p_calculator, p_datetime, p_llm],
    "text_processing": [p_text_stats, p_llm],
    "classification": [p_topic_classify, p_llm],
    "retrieval": [p_keyword_search, p_semantic_search, p_llm],
    "semantic_search": [p_semantic_search, p_llm],
    "ranking": [p_ranking, p_llm],
    "structured_data": [p_structured_data, p_llm],
    "creative_generation": [p_linkedin_writer],
    "coding": [p_llm],
    "reasoning": [p_llm],
}


def _accepted(res: ProcessorResult) -> bool:
    if not res.ok:
        return False
    return res.confidence >= LAYER_THRESHOLD.get(res.layer, 0.5)


def run_task(
    task: str,
    *,
    mode: str = "harness",
    options: dict | None = None,
    record: bool = True,
) -> dict:
    """Execute one task. `mode='baseline'` sends it straight to the LLM."""
    t0 = time.perf_counter()
    task_id = metrics.new_task_id()
    opts = dict(options or {})
    ctx = {"payload": extract_payload(task), **opts}
    before = metrics.snapshot()
    steps: list[dict] = []
    escalations = 0

    if mode == "baseline":
        res = p_llm(task, {**ctx, "ground": False})
        category, cls_conf, cls_proc = "baseline", 1.0, "none"
        steps.append({"processor": res.processor, "confidence": res.confidence,
                      "accepted": True, "latency_ms": round(res.latency_ms, 2)})
    else:
        cls = classify_task(task)
        category, cls_conf, cls_proc = cls.result, cls.confidence, cls.processor
        chain = CHAINS.get(category, [p_llm])
        res = None
        for proc in chain:
            r = proc(task, ctx)
            ok = _accepted(r)
            steps.append({
                "processor": r.processor, "layer": r.layer, "confidence": round(r.confidence, 4),
                "accepted": ok, "latency_ms": round(r.latency_ms, 2),
                "error": r.error,
            })
            res = r
            if ok:
                break
            escalations += 1
        escalations = max(0, escalations - (0 if res and not _accepted(res) else 0))

    answer = _answer_of(res)
    verdict = _verify(task, answer, res, category)
    after = metrics.snapshot()
    total_ms = (time.perf_counter() - t0) * 1000

    out = {
        "task_id": task_id,
        "mode": mode,
        "task_type": category,
        "classifier": {"processor": cls_proc, "confidence": round(cls_conf, 4)},
        "answer": answer,
        "result": res.result if res else None,
        "processor": res.processor if res else None,
        "layer": res.layer if res else None,
        "confidence": round(res.confidence, 4) if res else 0.0,
        "llm_used": bool(res and res.llm_used),
        "llm_calls": res.llm_calls if res else 0,
        "tokens": res.tokens if res else 0,
        "estimated_prompt_tokens": estimate_tokens(task),
        "latency_ms": round(total_ms, 2),
        "escalations": escalations,
        "steps": steps,
        "status": verdict["status"],
        "validation": verdict,
        "error": res.error if res else "no processor produced a result",
        "resources": {
            "cpu_percent": after["cpu_percent"],
            "rss_mb": after["rss_mb"],
            "rss_delta_mb": round(after["rss_mb"] - before["rss_mb"], 1),
        },
    }
    if record:
        metrics.record_task(
            task_id=task_id, task_type=category, processor=out["processor"],
            layer=out["layer"], confidence=out["confidence"], latency_ms=total_ms,
            llm_used=out["llm_used"], llm_calls=out["llm_calls"], tokens=out["tokens"],
            escalations=escalations, success=verdict["status"] in (STATUS_OK, "PARTIAL_SUCCESS"),
            error_type=(res.error.split(":")[0] if res and res.error else None),
            input_chars=len(task),
        )
    return out


def _answer_of(res: ProcessorResult | None) -> str:
    if res is None:
        return ""
    if res.meta.get("answer") is not None:
        a = res.meta["answer"]
        return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
    if isinstance(res.result, str):
        return res.result
    if res.result is None:
        return ""
    return json.dumps(res.result, ensure_ascii=False)[:2000]


def _verify(task: str, answer: str, res: ProcessorResult | None, category: str) -> dict:
    if res is None or not res.ok:
        return {"status": STATUS_FAIL, "issues": [res.error if res else "no result"]}
    if category == "deterministic" and D.looks_like_math(task):
        v = validate_numeric(task, answer)
        if v.get("checked"):
            return v
    if category == "creative_generation" and isinstance(res.result, dict):
        return res.result.get("validation", {"status": STATUS_OK, "issues": []})
    base = validate_generic(answer)
    if res.confidence < LAYER_THRESHOLD.get(res.layer, 0.5) and base["status"] == STATUS_OK:
        base["status"] = STATUS_LOW
    return base
