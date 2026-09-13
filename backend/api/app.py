"""FastAPI backend. Localhost-only by default; no outbound calls anywhere."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import REPORTS_DIR, SETTINGS, ensure_dirs
from backend.db import counts as db_counts
from backend.db import recall, remember
from backend.harness import metrics
from backend.harness.router import run_task
from backend.llm import manager as LLM
from backend.llm.discovery import scan_models, select_model
from backend.services import style_engine as SE
from backend.tools.deterministic import system_metrics

_warm: dict = {"state": "cold"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the model and the index in the background so the first request is fast."""
    import os

    ensure_dirs()
    from backend.agent.governor import governor
    from backend.agent.proc import cap_current_process
    from backend.agent.worker import worker

    # Hard CPU cap for this server and everything it starts (llama.cpp, embeddings, tests).
    # Enabled by start_backend.bat; off by default so test clients do not cap the test runner.
    if os.environ.get("JLA_SELF_CPU_CAP", "0") == "1":
        _warm["self_cpu_cap"] = {"percent": SETTINGS.agent.cpu_cap_percent - 5,
                                 "applied": cap_current_process(SETTINGS.agent.cpu_cap_percent - 5)}
    governor().start()
    try:
        _warm["recovered_tasks"] = worker().recover()
    except Exception as e:
        _warm["recover_error"] = str(e)

    def _bg() -> None:
        _warm["state"] = "warming"
        t0 = time.perf_counter()
        try:
            _warm.update(LLM.warmup())
        except Exception as e:
            _warm["error"] = str(e)
        _warm["state"] = "ready"
        _warm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # JLA_SKIP_WARMUP=1: a second (test) instance must not load another copy of the model.
    if os.environ.get("JLA_SKIP_WARMUP") == "1":
        _warm["state"] = "ready"
        _warm["skipped"] = True
    else:
        threading.Thread(target=_bg, daemon=True).start()
    yield
    worker().stop(2.0)
    governor().stop()


app = FastAPI(title="Jimit Local AI", version="1.0.0", lifespan=lifespan)

# The UI is served by Vite in dev; only localhost origins are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.agent_routes import router as agent_router  # noqa: E402 - after app creation

app.include_router(agent_router)

# ------------------------------------------------------------------ models

class TaskIn(BaseModel):
    task: str = Field(min_length=1, max_length=20000)
    mode: str = "harness"            # harness | baseline
    options: dict = Field(default_factory=dict)


class WriteIn(BaseModel):
    topic: str = Field(min_length=3, max_length=2000)
    audience: str = ""
    pillar: str = ""
    length: str = "medium"
    cta: str = ""
    facts: str = ""
    n_examples: int | None = None
    temperature: float = 0.75


class ImproveIn(BaseModel):
    previous: str
    instruction: str
    topic: str = ""
    length: str = "medium"


class SearchIn(BaseModel):
    q: str
    mode: str = "hybrid"             # hybrid | keyword | semantic
    top_k: int = 8
    doc_type: str | None = None
    topic: str | None = None


class SettingsIn(BaseModel):
    prefer_size: str | None = None   # auto | small | large
    temperature: float | None = None
    max_tokens: int | None = None
    n_ctx: int | None = None
    max_style_examples: int | None = None


class ScoreIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    topic: str = ""
    audience: str = ""
    pillar: str = ""
    length: str = "medium"
    facts: str = ""
    validation: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)


class FeedbackIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    post: str = ""
    run_id: str = ""


class BenchmarkIn(BaseModel):
    suites: list[str] | None = None
    include_baseline: bool = True
    limit: int | None = None


# ------------------------------------------------------------------ routes

@app.get("/api/health")
def health() -> dict:
    mf = LLM.selected_model()
    return {
        "ok": True,
        "warmup": _warm,
        "llm_loaded": LLM.is_loaded(),
        "model": mf.name if mf else (LLM.llm().name if LLM.is_loaded() else None),
        "runtime": LLM.llm().runtime if LLM.is_loaded() else "llama.cpp",
        "system": system_metrics(),
    }


@app.get("/api/system")
def system() -> dict:
    import platform
    import sys

    import psutil

    mf = select_model(SETTINGS.llm.prefer_size)
    vm = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "gpu": False,
        "models_found": [m.to_dict() for m in scan_models()],
        "selected_model": mf.to_dict() if mf else None,
        "llm_settings": {
            "prefer_size": SETTINGS.llm.prefer_size, "n_ctx": SETTINGS.llm.n_ctx,
            "n_threads": SETTINGS.llm.n_threads, "temperature": SETTINGS.llm.temperature,
            "max_tokens": SETTINGS.llm.max_tokens,
        },
        "harness_settings": {
            "max_style_examples": SETTINGS.harness.max_style_examples,
            "max_context_chars": SETTINGS.harness.max_context_chars,
        },
        "db": db_counts(),
        "usage": LLM.USAGE,
        "warmup": _warm,
    }


@app.post("/api/task")
def task(inp: TaskIn) -> dict:
    if inp.mode not in ("harness", "baseline"):
        raise HTTPException(400, "mode must be 'harness' or 'baseline'")
    return run_task(inp.task, mode=inp.mode, options=inp.options)


@app.post("/api/writer/generate")
def writer_generate(inp: WriteIn) -> dict:
    from backend.services.linkedin_writer import generate

    out = generate(
        inp.topic, audience=inp.audience, pillar=inp.pillar, length=inp.length,
        cta=inp.cta, facts=inp.facts, n_examples=inp.n_examples,
        temperature=inp.temperature,
    )
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.post("/api/writer/improve")
def writer_improve(inp: ImproveIn) -> dict:
    from backend.services.linkedin_writer import generate

    out = generate(inp.topic or "revision", length=inp.length,
                   previous=inp.previous, instruction=inp.instruction)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.get("/api/knowledge/stats")
def knowledge_stats() -> dict:
    from backend.ml.topics import topic_counts
    from backend.retrieval.index import stats as rstats

    prof = SE.load_profile()
    return {
        "documents": db_counts(),
        "by_type": SE.profile_stats(),
        "topics": topic_counts(),
        "retrieval": rstats(),
        "style_profile": {
            "exists": prof is not None,
            "n_documents": (prof or {}).get("n_documents", 0),
            "date_range": (prof or {}).get("date_range"),
            "summary": SE.summarize_profile(prof) if prof else None,
        },
    }


@app.post("/api/knowledge/search")
def knowledge_search(inp: SearchIn) -> dict:
    from backend.retrieval.index import index

    idx = index()
    t0 = time.perf_counter()
    if inp.mode == "keyword":
        hits = idx.keyword_search(inp.q, top_k=inp.top_k)
    else:
        alpha = 1.0 if inp.mode == "semantic" else 0.5
        hits = idx.search(inp.q, top_k=inp.top_k, alpha=alpha,
                          doc_type=inp.doc_type, topic=inp.topic)
    return {
        "mode": inp.mode,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "llm_used": False,
        "count": len(hits),
        "hits": [h.to_dict(500) for h in hits],
    }


@app.get("/api/style/profile")
def style_profile() -> dict:
    p = SE.load_profile()
    if not p:
        raise HTTPException(404, "no style profile - run ingestion first")
    return {"profile": p, "summary": SE.summarize_profile(p),
            "constraints": SE.style_constraints(p)}


@app.post("/api/ingest")
def ingest_now(rebuild_style: bool = True) -> dict:
    from backend.ingestion.linkedin.ingest import ingest
    from backend.ml.topics import train_from_corpus
    from backend.retrieval.index import index

    rep = ingest()
    out = {"ingest": rep.to_dict()}
    if rebuild_style:
        out["style"] = {"n_documents": SE.build_profile().get("n_documents", 0)}
        out["topics"] = train_from_corpus()
        out["index"] = index(rebuild=True, rechunk=True).build(rechunk=True)
    return out


@app.get("/api/metrics/summary")
def metrics_summary() -> dict:
    return {"summary": metrics.summary(), "recent": metrics.recent(20),
            "process": metrics.snapshot(), "llm_usage": LLM.USAGE}


@app.get("/api/benchmark/latest")
def benchmark_latest() -> dict:
    f = Path(REPORTS_DIR) / "benchmark_results.json"
    if not f.exists():
        raise HTTPException(404, "no benchmark has been run yet")
    return json.loads(f.read_text(encoding="utf-8"))


@app.post("/api/benchmark/run")
def benchmark_run(inp: BenchmarkIn) -> dict:
    from backend.benchmarks.runner import run_benchmark

    return run_benchmark(suites=inp.suites, include_baseline=inp.include_baseline,
                         limit=inp.limit)


@app.get("/api/evaluation/report")
def evaluation_report() -> dict:
    """Serve the predicted-vs-actual artifacts written by the QA pass."""
    d = Path(REPORTS_DIR) / "evaluation"
    out: dict = {}
    for key, name in (("predicted", "predicted_profile.json"),
                      ("comparison", "comparison.json"),
                      ("iterations", "iterations.json")):
        f = d / name
        if f.exists():
            out[key] = json.loads(f.read_text(encoding="utf-8"))
    if not out:
        raise HTTPException(404, "no evaluation has been run yet")
    return out


@app.post("/api/evaluation/score")
def evaluation_score(inp: ScoreIn) -> dict:
    """Score a post against the predicted profile for its brief. No model call."""
    from backend.benchmarks.evaluation import build_predicted_profile, evaluate

    predicted = build_predicted_profile(
        inp.topic, audience=inp.audience, pillar=inp.pillar, facts=inp.facts,
        length=inp.length)
    if "error" in predicted:
        raise HTTPException(400, predicted["error"])
    result = evaluate(inp.text, predicted, topic=inp.topic, audience=inp.audience,
                      facts=inp.facts, validation=inp.validation, metrics=inp.metrics)
    return {"predicted": predicted, "evaluation": result}


@app.post("/api/feedback")
def post_feedback(inp: FeedbackIn) -> dict:
    """Classify feedback to the layer that owns the fix, then persist it."""
    from backend.services.feedback import record_feedback

    out = record_feedback(inp.text, post=inp.post, run_id=inp.run_id)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.get("/api/feedback")
def get_feedback(limit: int = 50) -> dict:
    from backend.services.feedback import list_feedback

    return {"items": list_feedback(limit)}


@app.get("/api/settings")
def get_settings() -> dict:
    return {
        "llm": {
            "prefer_size": SETTINGS.llm.prefer_size, "temperature": SETTINGS.llm.temperature,
            "max_tokens": SETTINGS.llm.max_tokens, "n_ctx": SETTINGS.llm.n_ctx,
            "n_threads": SETTINGS.llm.n_threads,
        },
        "harness": {
            "max_style_examples": SETTINGS.harness.max_style_examples,
            "max_context_chars": SETTINGS.harness.max_context_chars,
            "thresholds": {
                "deterministic": SETTINGS.harness.deterministic_min_conf,
                "nlp": SETTINGS.harness.nlp_min_conf,
                "ml": SETTINGS.harness.ml_min_conf,
                "retrieval": SETTINGS.harness.retrieval_min_conf,
            },
        },
        "preferences": recall("preference"),
    }


@app.post("/api/settings")
def set_settings(inp: SettingsIn) -> dict:
    reload_needed = False
    if inp.prefer_size and inp.prefer_size != SETTINGS.llm.prefer_size:
        SETTINGS.llm.prefer_size = inp.prefer_size
        reload_needed = True
    if inp.n_ctx and inp.n_ctx != SETTINGS.llm.n_ctx:
        SETTINGS.llm.n_ctx = int(inp.n_ctx)
        reload_needed = True
    if inp.temperature is not None:
        SETTINGS.llm.temperature = float(inp.temperature)
    if inp.max_tokens is not None:
        SETTINGS.llm.max_tokens = int(inp.max_tokens)
    if inp.max_style_examples is not None:
        SETTINGS.harness.max_style_examples = int(inp.max_style_examples)
    remember("preference", "llm_settings", {
        "prefer_size": SETTINGS.llm.prefer_size, "temperature": SETTINGS.llm.temperature,
        "max_tokens": SETTINGS.llm.max_tokens, "n_ctx": SETTINGS.llm.n_ctx,
        "max_style_examples": SETTINGS.harness.max_style_examples,
    }, source="user")
    if reload_needed:
        LLM.llm(force_reload=True)  # rebuilt lazily on next call
    return {"ok": True, "reload_needed": reload_needed, **get_settings()}
