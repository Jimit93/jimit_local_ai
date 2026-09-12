"""Cold/warm start and per-stage latency breakdown. Measured, not estimated."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402

T: dict[str, float] = {}
proc = psutil.Process()


def mark(name: str, t0: float) -> None:
    T[name] = round((time.perf_counter() - t0) * 1000, 1)


t = time.perf_counter()
from backend.harness.router import run_task  # noqa: E402
from backend.llm import manager as LLM  # noqa: E402
from backend.retrieval.index import index  # noqa: E402
from backend.services.linkedin_writer import generate, select_examples  # noqa: E402
mark("import_backend", t)

t = time.perf_counter(); index().build(use_embeddings=True); mark("index_build_cold", t)
t = time.perf_counter(); index().search("local ai agent", top_k=3); mark("search_warm", t)
t = time.perf_counter(); select_examples("local ai agent on cpu"); mark("select_examples", t)
t = time.perf_counter(); LLM.llm().load(); mark("llm_load_cold", t)
t = time.perf_counter(); LLM.generate("Reply with OK.", max_tokens=4, temperature=0.0)
mark("llm_tiny_call_warm", t)

t = time.perf_counter()
out = generate("running Qwen locally on CPU with a routing harness",
               facts="Measured: 14.7 tok/s on 1.8B q4_k_m, 136 posts ingested.")
mark("writer_full_warm", t)

t = time.perf_counter(); run_task("What is 25 * 40?"); mark("task_calc", t)
t = time.perf_counter(); run_task("how many posts do i have"); mark("task_sql", t)
t = time.perf_counter(); run_task("Find all LinkedIn posts mentioning RAG"); mark("task_keyword", t)
t = time.perf_counter(); run_task("Find posts semantically related to local AI")
mark("task_semantic_warm", t)

print(json.dumps({
    "stages_ms": T,
    "writer": {k: out["metrics"][k] for k in
               ("llm_calls", "tokens", "prompt_tokens", "prompt_chars", "latency_ms")},
    "rss_mb": round(proc.memory_info().rss / 1e6, 1),
    "model": LLM.selected_model().name if LLM.selected_model() else None,
}, indent=2))
