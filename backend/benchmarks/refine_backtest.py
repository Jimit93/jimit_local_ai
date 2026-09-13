"""Backtest: one-shot generation vs the Describe -> Generate -> Test -> Refine loop.

Same briefs, same model, same deterministic scorers for both modes. Two comparisons:

  - PAIRED (the clean one): the refine loop's own first draft vs that same draft after
    refinement. Same text in, so the difference is exactly what Refine did.
  - END-TO-END: one-shot vs refine loop as a user would run them. NOT the same draft:
    llama.cpp is seeded once when the model loads, so every call advances the random
    state and the refine call samples a different draft than the one-shot call before
    it. Read this row as cost and outcome of each path, not as a controlled test.

What it answers, with numbers rather than the source notes' assertion:
  - does refinement raise the pass rate and the score?
  - does it keep every post on a complete sentence?
  - what does it cost in model calls, tokens and seconds?
  - where does the score stop improving - is "20x" a real effect or a plateau at 2?

Facts in the briefs are measurements this project recorded - never invented.
Must run inside the backend process (POST /api/writer/refine-backtest) so it uses the
model that is already loaded instead of starting a second copy.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from statistics import mean

from backend.config import REPORTS_DIR, ensure_dirs

OUT_JSON = Path(REPORTS_DIR) / "evaluation" / "refine_backtest.json"
OUT_MD = Path(REPORTS_DIR) / "evaluation" / "refine_backtest.md"

BRIEFS: list[dict] = [
    {
        "id": "harness_proof",
        "topic": ("How I built a fully local AI agent that runs on a CPU-only laptop: a routing "
                  "harness that sends every task to the cheapest processor that can answer it "
                  "reliably, so the language model is the last resort instead of the default."),
        "audience": "AI engineering hiring managers and companies that want AI automation built",
        "pillar": "engineering depth, proof",
        "facts": ("Measured on one benchmark run of 23 tasks, LLM-only baseline vs the routing "
                  "harness on the same machine: LLM calls 23 to 4 (-82.6%). Tokens 8364 to 2214 "
                  "(-73.5%). Total latency 283.8s to 48.4s (-83.0%). Accuracy 39.1% to 100%. "
                  "19 of the 23 tasks were answered with no LLM call at all."),
        "facts_source": "reports/benchmark_report.md, run b79f795399",
    },
    {
        "id": "business_automation_no_facts",
        "topic": ("An AI agent that takes over manual business work - Excel sheets, emails, "
                  "reports and invoices - so the team can focus on decisions."),
        "audience": "CEOs and small business owners",
        "pillar": "practical AI",
        "facts": "",
        "facts_source": "none - mirrors the draft that was returned as SUCCESS",
    },
    {
        "id": "cpu_inference",
        "topic": "Running quantized language models fully offline on a CPU-only laptop.",
        "audience": "AI engineers and ML developers",
        "pillar": "engineering depth",
        "facts": ("Measured on this machine, CPU only, no GPU: Qwen1.5-1.8B-chat Q4_K_M runs at "
                  "about 14.7 tokens per second. Qwen2.5-7B-instruct Q4_0 runs at about 5.4 "
                  "tokens per second."),
        "facts_source": "CLAUDE.md, Known constraints on this machine",
    },
    {
        "id": "deterministic_validator",
        "topic": ("Why I check AI-generated LinkedIn posts with deterministic rules instead of "
                  "asking a second LLM to judge them."),
        "audience": "ML engineers and AI product teams",
        "pillar": "engineering depth",
        "facts": ("A post that cited 0 of 18 supplied measured numbers scored 100% on the old "
                  "validator. After a deterministic facts_used check was added, the same writer "
                  "cited 3 numbers on its first call, still using 1 LLM call."),
        "facts_source": "reports/evaluation/evaluation_report.md",
    },
    {
        "id": "rag_small_corpus",
        "topic": "Semantic chunking for retrieval over a small personal writing corpus.",
        "audience": "RAG developers and data engineers",
        "pillar": "engineering depth",
        "facts": ("The corpus has 6143 LinkedIn export documents, of which 136 are posts and 77 "
                  "are usable for style. They are split into 127 chunks with 127 MiniLM embeddings."),
        "facts_source": "data/agent.db counts",
    },
]


def _summary(out: dict, overall: float) -> dict:
    v = out["validation"]
    from backend.harness.validator import CRITICAL_CHECKS
    from backend.nlp import features as F

    failed = [k for k, ok in v["checks"].items() if not ok]
    text = out["text"]
    return {
        "status": v["status"],
        "passed": v["status"] == "SUCCESS",
        "style_match": v["style_match"],
        "overall": overall,
        "critical_failures": [k for k in failed if k in CRITICAL_CHECKS],
        "failed_checks": failed,
        "issues": v["issues"],
        "complete_ending": v["checks"].get("complete_ending", True),
        "banned_phrases": F.generic_phrase_hits(text),
        "hashtags": [t for t in text.split() if t.startswith("#")][-5:],
        "words": v["stats"]["words"],
        "llm_calls": out["metrics"]["llm_calls"],
        "tokens": out["metrics"]["tokens"],
        "latency_ms": out["metrics"]["latency_ms"],
        "truncated_first_draft": bool(out["metrics"].get("truncated")),
        "text": text,
    }


def run_refine_backtest(
    briefs: list[dict] | None = None,
    *,
    max_iterations: int = 20,
    max_llm_fixes: int = 3,
    persist: bool = True,
) -> dict:
    from backend.benchmarks.evaluation import build_predicted_profile, evaluate
    from backend.config import SETTINGS
    from backend.services.linkedin_writer import generate
    from backend.services.refine_loop import refine

    ensure_dirs()
    t0 = time.time()
    run_id = uuid.uuid4().hex[:10]
    rows = []
    for b in (briefs or BRIEFS):
        kw = {k: b.get(k, "") for k in ("topic", "audience", "pillar", "facts")}
        predicted = build_predicted_profile(kw["topic"], audience=kw["audience"],
                                            pillar=kw["pillar"], facts=kw["facts"])

        def score(out: dict) -> float:
            return evaluate(out["text"], predicted, topic=kw["topic"], audience=kw["audience"],
                            facts=kw["facts"], validation=out["validation"],
                            metrics=out["metrics"])["overall_score"]

        one = generate(kw["topic"], audience=kw["audience"], pillar=kw["pillar"], facts=kw["facts"])
        ref = refine(kw["topic"], audience=kw["audience"], pillar=kw["pillar"], facts=kw["facts"],
                     max_iterations=max_iterations, max_llm_fixes=max_llm_fixes)
        if "error" in one or "error" in ref:
            rows.append({"id": b["id"], "error": one.get("error") or ref.get("error")})
            continue
        r = ref["refine"]
        rows.append({
            "id": b["id"],
            "brief": {**kw, "facts_source": b.get("facts_source", "")},
            "one_shot": {**_summary(one, score(one)), "repaired": one["metrics"].get("repaired")},
            "refine": {
                **_summary(ref, score(ref)),
                "iterations": r["iterations"], "accepted_steps": r["accepted_steps"],
                "llm_fixes_used": r["llm_fixes_used"], "stopped_because": r["stopped_because"],
                "initial": r["initial"], "score_curve": r["score_curve"],
                "trace": [{k: t.get(k) for k in ("iteration", "check", "fixer", "kind",
                                                 "accepted", "status", "overall", "note")}
                          for t in r["trace"]],
            },
        })

    ok = [row for row in rows if "error" not in row]

    def agg(mode: str) -> dict:
        m = [row[mode] for row in ok]
        if not m:
            return {}
        return {
            "pass_rate": round(sum(x["passed"] for x in m) / len(m), 4),
            "complete_ending_rate": round(sum(x["complete_ending"] for x in m) / len(m), 4),
            "hashtag_rate": round(sum(bool(x["hashtags"]) for x in m) / len(m), 4),
            "mean_overall": round(mean(x["overall"] for x in m), 4),
            "mean_style_match": round(mean(x["style_match"] for x in m), 4),
            "mean_critical_failures": round(mean(len(x["critical_failures"]) for x in m), 4),
            "posts_with_banned_phrases": sum(bool(x["banned_phrases"]) for x in m),
            "mean_llm_calls": round(mean(x["llm_calls"] for x in m), 3),
            "mean_tokens": round(mean(x["tokens"] for x in m), 1),
            "mean_latency_ms": round(mean(x["latency_ms"] for x in m), 1),
            "total_llm_calls": sum(x["llm_calls"] for x in m),
        }

    def agg_draft() -> dict:
        d = [row["refine"]["initial"] for row in ok]
        if not d:
            return {}
        return {
            "pass_rate": round(sum(x["status"] == "SUCCESS" for x in d) / len(d), 4),
            "mean_overall": round(mean(x["overall"] for x in d), 4),
            "mean_style_match": round(mean(x["style_match"] for x in d), 4),
        }

    # Score by micro-iteration, averaged across briefs (a finished loop holds its last value).
    curves = [row["refine"]["score_curve"] for row in ok]
    depth = max((len(c) for c in curves), default=0)
    curve = [round(mean(c[min(i, len(c) - 1)] for c in curves), 4) for i in range(depth)]
    gains = [round(curve[i] - curve[i - 1], 4) for i in range(1, len(curve))]
    last_gain = max((i for i, g in enumerate(gains, start=1) if g > 0.001), default=0)

    report = {
        "run_id": run_id,
        "started_at": t0,
        "duration_s": round(time.time() - t0, 1),
        "model": _model_name(),
        "seed": SETTINGS.llm.seed,
        "settings": {"max_iterations": max_iterations, "max_llm_fixes": max_llm_fixes},
        "n_briefs": len(rows),
        "aggregate": {"one_shot": agg("one_shot"), "refine_draft": agg_draft(),
                      "refine": agg("refine")},
        "comparison_note": ("refine_draft -> refine is paired (same draft before and after "
                            "refinement). one_shot -> refine is end-to-end on different "
                            "samples: the seed is applied once at model load."),
        "iteration_curve": curve,
        "iteration_gains": gains,
        "last_improving_iteration": last_gain,
        "max_iterations_used": max((row["refine"]["iterations"] for row in ok), default=0),
        "rows": rows,
    }
    if persist:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        OUT_MD.write_text(_markdown(report), encoding="utf-8")
    return report


def latest() -> dict | None:
    return json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else None


def _model_name() -> str | None:
    from backend.llm import manager as LLM

    mf = LLM.selected_model()
    return mf.name if mf else None


def _markdown(r: dict) -> str:
    a, b = r["aggregate"].get("one_shot", {}), r["aggregate"].get("refine", {})
    d = r["aggregate"].get("refine_draft", {})
    lines = [
        "# Refine-loop backtest",
        "",
        f"Run `{r['run_id']}` · model `{r['model']}` · seed {r['seed']} · "
        f"{r['n_briefs']} briefs · {r['duration_s']} s",
        "",
        "**Read the columns correctly.** *Refine draft → Refine loop* is paired: the same "
        "draft before and after refinement. *One-shot* is a different sample - the seed is "
        "applied once at model load, so it compares the two paths end to end, not as a "
        "controlled test.",
        "",
        "| Metric | One-shot | Refine draft (before fixes) | Refine loop |",
        "|---|---|---|---|",
    ]
    for key, label in (("pass_rate", "Pass rate (SUCCESS)"), ("complete_ending_rate", "Complete ending"),
                       ("hashtag_rate", "Has hashtags"), ("mean_overall", "Mean overall score"),
                       ("mean_critical_failures", "Mean critical failures"),
                       ("posts_with_banned_phrases", "Posts with banned phrases"),
                       ("mean_llm_calls", "Mean LLM calls"), ("mean_tokens", "Mean tokens"),
                       ("mean_latency_ms", "Mean latency (ms)")):
        lines.append(f"| {label} | {a.get(key, '—')} | {d.get(key, '—')} | {b.get(key, '—')} |")
    lines += ["", "## Score by micro-iteration (mean across briefs)", "",
              " → ".join(str(x) for x in r["iteration_curve"]),
              "", f"Last iteration that still improved the mean: **{r['last_improving_iteration']}** "
              f"of up to {r['settings']['max_iterations']} allowed "
              f"(deepest loop used {r['max_iterations_used']}).", ""]
    if r.get("reviewer_findings"):
        lines += ["## What the checks could not see (read from the generated text)", ""]
        lines += [f"- {f}" for f in r["reviewer_findings"]] + [""]
    lines += ["## Per brief", ""]
    for row in r["rows"]:
        if "error" in row:
            lines.append(f"- **{row['id']}** — error: {row['error']}")
            continue
        o, f = row["one_shot"], row["refine"]
        lines.append(
            f"- **{row['id']}** — one-shot {o['status']} {o['overall']} "
            f"({o['llm_calls']} calls) → refine {f['status']} {f['overall']} "
            f"({f['llm_calls']} calls, {f['iterations']} iterations, "
            f"stopped: {f['stopped_because']})")
    return "\n".join(lines) + "\n"
