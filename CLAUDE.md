# Jimit Local AI — project rules

A fully local AI agent: Qwen (GGUF, llama.cpp) + a routing harness that keeps the
model out of work cheaper layers can do. Windows, CPU-only, no network at runtime.

## Permanent architectural rules

1. **Cheapest reliable processor wins.** deterministic → NLP → classical ML →
   retrieval → LLM. A processor may only escalate when its confidence is below the
   threshold for its layer (`router.LAYER_THRESHOLD`, values in
   `config.py: HarnessSettings`).
2. **The LLM is last in every chain.** If you add a processor chain, `p_llm` goes at
   the end or not at all. `tests/test_router.py` enforces this.
3. **Every processor returns a `ProcessorResult`** (`backend/common.py`): result,
   confidence, processor, latency, layer, llm_used, tokens. No exceptions. A
   processor that cannot answer returns confidence `0.0` and an `error`, never a
   guess — the router escalates on confidence alone.
4. **No fake intelligence.** A module called "ML" must contain a trained model; a
   module called "NLP" must do actual text processing. Never rename an LLM call.
5. **One model, one process.** `backend/llm/manager.py` is the only place a model is
   constructed. Never instantiate an adapter directly outside it.
6. **Nothing leaves the machine.** No outbound HTTP except `127.0.0.1:11434`
   (optional Ollama). HF offline mode is forced in `backend/retrieval/index.py`.
7. **Measured, never asserted.** Any number in a report or a docstring must come
   from a run. Benchmark ground truth is computed at run time, not hand-written.
8. **Style ≠ fact.** Historical posts are style evidence. Facts for a generated post
   come only from the user's input for that request.
9. **RAG retrieves chunks, style uses documents.** Ingestion splits documents with
   semantic chunking (`backend/retrieval/chunking.py`); retrieval scores chunks and
   collapses to one hit per document. `select_examples` must use `Hit.doc_text`,
   never the chunk — voice lives in the whole post.
10. **Chunk boundaries need a distance floor.** A percentile alone collapses to 0 on
    a single-topic document and makes every sentence a boundary (caught by
    `tests/test_chunking.py`).

## Layout

```
backend/config.py          paths + settings (no side effects on import)
backend/common.py          ProcessorResult
backend/db.py              SQLite (WAL), schema, memory helpers
backend/llm/               discovery, adapters (llama.cpp, ollama), singleton manager
backend/tools/             deterministic tools + registry
backend/nlp/               features (hooks, CTA, density), keywords (TF-IDF, BM25)
backend/ml/topics.py       weak-supervision labels + TF-IDF/LogReg classifier
backend/retrieval/chunking.py semantic chunking (embedding-boundary, adaptive)
backend/retrieval/index.py chunk-level: metadata filter → BM25 → MiniLM/FAISS fusion
backend/ingestion/linkedin ZIP/CSV parsing by header signature
backend/harness/           task_classifier, router, context_manager, prompt_builder,
                           validator, metrics
backend/services/          style_engine, linkedin_writer
backend/benchmarks/        tasks, runner (baseline vs harness), linkedin_backtest
backend/api/app.py         FastAPI, localhost only
backend/agent/             coding agent (docs/coding_agent/ARCHITECTURE.md): policy, proc,
                           code_tools, checkpoints, governor, ram, project_index, context,
                           memory, state, intents, executor, edits, planner, loop, worker
backend/api/agent_routes.py  /api/agent/* (run, tasks, events, files, memory, checkpoints, RAM)
backend/harness/completion.py  replies end on a complete sentence inside max_tokens
frontend/                  React + TS + Vite (dev server proxies /api → :8077)
```

## Coding-agent rules

11. **Every agent file change goes through `code_tools` with a checkpoint.** Never write to a
    workspace with plain `open()`; rollback depends on the snapshot taken first.
12. **Paths, commands and secrets go through `agent/policy.py`.** Commands are argv allowlists,
    never a shell. Delete, project scripts and process kills need a user approval the loop
    cannot grant itself.
13. **The loop owns sequencing; a model only proposes.** Edits are parsed and applied by
    `edits.py` (unique match, dry-run, syntax check) and verified by tests before a task completes.
14. **Heavy work is admitted by the governor** (`governor().slot("llm"|"tests"|...)`) and child
    processes run through `proc.run_process` (Job Object CPU cap).
15. **Agent tests never touch `data/agent.db`**: `tests/agent/conftest.py` swaps in a temporary
    database; backtests use their own databases under `data/backtest/`.

## Generated artifacts — not in git, rebuilt by ingestion

`data/agent.db`, `data/models/*.joblib` (intent + topic classifiers),
`config/style_profile.json` and `reports/_*.log` are all build products of
`ingest_linkedin.bat`. Do not hand-edit them, and do not treat a missing one as a
bug before checking whether ingestion has been run. Tests that need the corpus
`pytest.skip` when it is absent, so a green run on a machine without `data/` is
proving less than it looks.

## Commands

```
ingest_linkedin.bat            parse export → style profile → topics → embeddings
start_backend.bat              FastAPI on 127.0.0.1:8077
start_frontend.bat             Vite on 127.0.0.1:5173
start_agent.bat                both + opens the browser
run_tests.bat                  pytest
run_benchmark.bat              baseline vs harness, writes reports/
run_backtest.bat               held-out LinkedIn style back-test
python -m scripts.agent_backtest    coding-agent system backtest (real models, hidden tests)
python -m scripts.restart_backtest  kill/restart/resume with real processes (port 8079)
python -m scripts.stress_backtest   large repo, long session, bursts, failures, CPU/RAM pressure
python -m scripts.cpu_backtest      CPU policy measurements
scripts\test_backend.bat / test_frontend.bat   second instance on :8078 / :5174 for UI testing
```

Run Python from the repo root as a module — `python -m pytest tests/test_router.py -q`,
`python -m uvicorn ...`. Bare `pytest` fails with `ModuleNotFoundError: No module
named 'backend'`; there is no conftest.py and no installed package, so `backend`
resolves only via the CWD that `python -m` puts on `sys.path` (the `.bat` files also
set `PYTHONPATH`). On Windows also set `PYTHONIOENCODING=utf-8`, as the `.bat` files
do — report output contains non-ASCII characters.

Environment overrides: `JLA_MODEL_PREFER=auto|small|large`, `JLA_N_CTX`,
`JLA_THREADS`, `JLA_MODEL_DIR`, `JLA_LINKEDIN_SRC`, `JLA_DATA_DIR`, `JLA_DB_PATH`,
`JLA_PORT`, `JLA_TIMEOUT`, `JLA_EMBED_MODEL`, `JLA_TEST_LLM=1` (enables the two slow
real-inference tests, skipped by default).

## Conventions

- Python 3.11, no new dependencies without a measured reason. Everything currently
  used was already installed on this machine.
- Prompts are built by `backend/harness/prompt_builder.py` only. Never inline a
  prompt string in a service.
- **Never put statistics in a prompt's CONTEXT.** A small model reproduces them as
  content (measured: it wrote "59% claim rate" into a post). Convert the profile to
  imperative rules via `style_engine.style_instructions()`.
- Model-specific output scaffolding (titles, `Claim:` labels, placeholders,
  over-length) is cleaned deterministically in `linkedin_writer`, not by another
  LLM call.
- Regex-heavy Python must be written with the editor tools, not through shell
  heredocs — `\b` gets mangled into a literal backspace.
- Heavy imports (`retrieval.index`, `ml.topics`, the services) are function-local
  inside `router.py` processors on purpose: importing the router must not pull in
  FAISS, a sentence-transformer or a GGUF. Keep new processors lazy the same way.

## HTTP surface (`backend/api/app.py`, 127.0.0.1:8077)

```
GET  /api/health /api/system /api/settings /api/style/profile
GET  /api/knowledge/stats /api/metrics/summary /api/benchmark/latest
POST /api/task                 {task, mode: harness|baseline, options}  -> run_task
POST /api/writer/generate      {topic, audience, pillar, length, cta}
POST /api/writer/improve       {previous, instruction, topic, length}
POST /api/knowledge/search     {q, mode: hybrid|keyword|semantic, top_k, doc_type, topic}
POST /api/ingest /api/benchmark/run /api/settings
```

The model and the index are warmed in a background thread by the lifespan hook, so
the first request does not pay for the load. API tests stub `LLM.warmup` and
exercise only endpoints that must answer without the model.

## Data model (`backend/db.py: SCHEMA`)

`linkedin_documents` → `chunks` → `chunk_embeddings` is the retrieval spine;
`style_features` (one row per usable post) and `topics` feed the style engine and
the topic classifier; `agent_memory` is the key/value recall store; `task_runs` and
`benchmark_runs` are append-only measurement logs. SQLite in WAL mode.

## Known constraints on this machine

- CPU only (no CUDA). Qwen1.5-1.8B ≈ 14.7 tok/s; Qwen2.5-7B ≈ 5.4 tok/s.
- Free RAM is the binding constraint; model selection is RAM-aware.
- The corpus is large but thin where it matters: 6,143 documents, of which only 136
  are posts and 77 are usable for style. Prefer linear models and report
  cross-validation folds honestly — there are only 2 documents in the smallest
  topic class.

## Task-specific playbooks

Four skills in `.claude/skills/` carry the detail and are the source of truth for
their area — read the relevant one before changing that subsystem, and put new
procedural knowledge there rather than here: `harness-routing` (adding a processor,
thresholds, escalation), `rag-ingestion` (chunking, embeddings, retrieval
debugging), `linkedin-writer` (the generation pipeline and style engine),
`benchmarking` (baseline-vs-harness runs and the back-test).
