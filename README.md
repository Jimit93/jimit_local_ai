# Jimit Local AI

A local AI agent workstation. Qwen runs on this machine through llama.cpp; a routing
harness sends each task to the cheapest processor that can answer it reliably, and
only reaches the model when nothing cheaper will do. Your LinkedIn export is parsed
locally into a knowledge base and a measured writing-style profile.

Nothing leaves the machine.

---

## Requirements

Already satisfied on this machine — listed so the project can be rebuilt elsewhere.

- Windows 11, Python 3.11, Node 20+
- `llama-cpp-python`, `scikit-learn`, `numpy`, `scipy`, `faiss`,
  `sentence-transformers`, `fastapi`, `uvicorn`, `psutil`, `pydantic`, `joblib`
- A Qwen GGUF model on disk (discovered automatically, see below)
- `sentence-transformers/all-MiniLM-L6-v2` in the local HuggingFace cache

No model is downloaded. No new heavyweight dependency was added.

## Install

```bat
pip install -r requirements.txt
cd frontend && npm install
```

## Model configuration

Models are **discovered**, not hard-coded. The scanner walks
`C:\Users\pc\Quantized_AI_agent\models` (override with `JLA_MODEL_DIR`), skips
virtualenvs, reads format/quantisation/size, and handles split GGUF shards.

Selection is RAM-aware: `auto` picks the largest model that fits in ~80% of free
RAM. Override with `JLA_MODEL_PREFER=small|large|auto`, on the Settings page, or
per-process via the environment.

Found on this machine:

| Model | Quant | Size | Cold load | Throughput | RSS |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | q4_0 (2 shards) | 4.43 GB | 29 s | 5.4 tok/s | 6.1 GB |
| Qwen1.5-1.8B-Chat | q4_k_m | 1.22 GB | 2.7 s | 14.7 tok/s | 2.1 GB |

Ollama is installed but holds no Qwen instruct model, so llama.cpp is the runtime.
An Ollama adapter exists as a fallback (`backend/llm/ollama_adapter.py`).

## Ingest your LinkedIn data

```bat
ingest_linkedin.bat
```

Parses every CSV in the configured exports, **identifying files by their header
signature rather than their filename**, normalises and deduplicates records, then
builds the style profile, trains the topic classifier and embeds the corpus.

> **Note on the export you have:** the *Basic* export contains no posts, comments or
> articles — only profile, connections and messages. The writing corpus comes from
> the *Complete* export next to it (`Shares_*.csv`, `Comments_*.csv`). Both are read;
> only files that actually exist are used.

Privacy choices made during ingestion:
- Direct messages are stored **only where you are the sender**; other people's
  private messages are never written to the database.
- Messages are excluded from the style corpus by default (`WRITING_TYPES`), because
  chat register would pollute the post-writing profile.

## Run

```bat
start_agent.bat          :: backend + UI + browser
```

or separately:

```bat
start_backend.bat        :: http://127.0.0.1:8077
start_frontend.bat       :: http://127.0.0.1:5173
```

The backend binds to 127.0.0.1 only and warms the model and index in the background
at startup, so the first request does not pay the cold-start cost.

## Benchmarks

```bat
run_benchmark.bat        :: LLM-only baseline vs harness, same model, same process
run_backtest.bat         :: held-out LinkedIn style reproduction
run_tests.bat            :: pytest
```

Reports are written to `reports/benchmark_report.md`,
`reports/benchmark_results.json` and `reports/linkedin_backtest.md`.

### Measured results

23 tasks across 9 suites, Qwen1.5-1.8B pinned for both modes, one process,
warm-up excluded (run `b79f795399`, 356s):

| Metric | Baseline (LLM only) | Harness | Change |
|---|---|---|---|
| Accuracy | 39.1% | 100.0% | +60.9 pts |
| LLM calls | 23 | 4 | -82.6% |
| Tokens | 8,364 | 2,214 | -73.5% |
| Total latency | 283.8 s | 48.4 s | -83.0% |
| Median latency | 13,352 ms | 2.5 ms | |
| CPU time (all threads) | 2,576 s | 478 s | -81.5% |
| Peak RSS | 3,030 MB | 3,027 MB | - |
| Tasks answered with no LLM call | 0 / 23 | 19 / 23 | |

Per suite, the harness used **zero** LLM calls for deterministic, NLP,
classification, retrieval, semantic search, ranking and structured-data tasks, and
matched the baseline on reasoning — where the model is genuinely the right tool. The
four remaining calls are the two generation tasks and the two reasoning tasks.

The baseline is not a straw man: it is the same model, in the same process, asked
the same question. It scores 39% because a 1.8B model cannot count documents in a
database it cannot see, cannot do exact search, and gets arithmetic wrong — which is
precisely the argument for routing.

Ground truth for every task is computed at run time from SQLite or Python, so the
suite stays correct as the corpus changes. Grading modes are documented in the
report.

### LinkedIn style back-test

6 posts held out (seed 7), style profile rebuilt from the remaining
57 with the evaluation posts excluded from **both** the profile and example
retrieval. Leakage checks found **0** violations.

| Measure | Value |
|---|---|
| Length similarity | 0.7999 |
| Sentence-length similarity | 0.4569 |
| Paragraph similarity | 0.5889 |
| Technical-density similarity | 0.5783 |
| Hook archetype match | 50% |
| CTA match | 83% |
| Hashtag behaviour match | 50% |
| Vocabulary overlap | 0.121 |
| **Verbatim copying** | **0%** |
| Mean style score | 0.578 |
| LLM calls per post | 1.0 |

Low vocabulary overlap is the goal, not a failure: the system reproduces structure,
rhythm and habits, not wording. A copied 6-gram halves a post's score; none occurred.

Length similarity moved 0.35 → 0.48 → 0.66 → 0.80 across four measured iterations as
prompt statistics were removed from the model's context, token caps were tightened and
deterministic length enforcement was added. The honest caveat: at 0.58 mean style
score the 1.8B reproduces *shape* better than *voice*. Hook archetype matches half the
time. Set `JLA_MODEL_PREFER=large` for the 7B when the post matters.

## Semantic chunking

The RAG pipeline does not cut documents every N characters. For each document it
splits into sentences (never crossing a paragraph break — a blank line is a
boundary the author put there deliberately), embeds each sentence with the local
MiniLM, measures cosine distance between neighbours, and cuts where the distance
exceeds the 80th percentile *of that document's own distribution*, subject to an
absolute floor of 0.15 so a single-topic document is not shredded. Groups are then
merged to respect 45–180 word bounds.

Documents shorter than 90 words are returned as a single chunk: an 80-word post has
no internal structure worth splitting.

Measured on this corpus: 103 documents → 127 chunks, 14 documents split. Boundaries
land where the meaning turns — one post cuts at "Why this matters:" with a boundary
score of 0.96. Retrieval then returns the passage that matched rather than the whole
post: across three sample queries that was **53.9% less grounding context** for the
same answers (3,932 vs 8,537 characters).

Style examples are the deliberate exception — `select_examples` always uses the full
document, because a post's voice lives in its opening, rhythm and sign-off.

## Architecture

```
        User → React UI (Vite, 127.0.0.1:5173)
                     │
              FastAPI (127.0.0.1:8077)
                     │
              Task classifier          rules → TF-IDF/LogReg intent model
                     │
              Router (per-category processor chain, cheapest first)
   ┌─────────┬───────────┬────────────┬─────────────┬──────────────┐
   │ tools   │ NLP       │ classical  │ retrieval   │ Qwen (last)  │
   │ calc,   │ features, │ ML topics  │ metadata →  │ planning,    │
   │ SQL,    │ TF-IDF,   │ TF-IDF +   │ BM25 →      │ reasoning,   │
   │ regex   │ BM25      │ LogReg     │ MiniLM/FAISS│ generation   │
   └─────────┴───────────┴────────────┴─────────────┴──────────────┘
                     │
              Deterministic validation / QC
                     │
                   User
```

Each processor returns `{result, confidence, processor, latency, layer, tokens}`.
A processor whose confidence clears its layer threshold ends the chain; otherwise the
harness escalates. Thresholds are strictest for the cheapest layers.

**Layers**

- *Deterministic* — calculator (AST-safe), date maths, text statistics, regex and
  exact search, CSV/JSON parsing, sorting/filtering/aggregation, duplicate
  detection, filesystem metadata, read-only SQL.
- *NLP* — sentence/paragraph segmentation, hook and CTA archetype detection,
  technical-density lexicon, readability, repetition, keyword extraction, TF-IDF,
  BM25, n-gram phrase mining, vocabulary profiling.
- *Classical ML* — weak-supervision topic labels from keyword rules, then a
  TF-IDF + logistic-regression classifier for the cases rules abstain on; a second
  TF-IDF+LogReg model classifies task intent.
- *Retrieval (RAG)* — documents are split by **semantic chunking** before indexing,
  then: metadata filter → BM25 → MiniLM embeddings (384-d, cached in SQLite) →
  FAISS inner product, fused with a tunable lexical/semantic weight, and collapsed
  back to one hit per document.
- *LLM* — Qwen via llama.cpp: planning, reasoning, synthesis, generation.

## Pages

**Chat** — any task; shows processor, confidence, latency, whether the model was
used, token count, sources and the full execution trace.
**LinkedIn Writer** — brief → post, with style match, technical density, source
count, LLM calls, tokens; regenerate, improve, copy.
**Knowledge** — corpus counts, topics, style summary, hybrid/keyword/semantic search.
**Tasks** — every executed task with its real cost.
**Benchmarks** — run the suite, see baseline vs harness.
**System** — machine, discovered models, database, live resource use, re-ingest.
**Settings** — model preference, context size, temperature, style-example budget.

## Privacy model

- The server binds to `127.0.0.1`; CORS allows only localhost origins.
- There are no outbound HTTP calls in the codebase other than optional Ollama on
  `127.0.0.1:11434`.
- `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are forced before the embedding model
  loads, so it cannot reach the network.
- Your LinkedIn export, the SQLite database, the embeddings and every inference stay
  on this machine. No cloud LLM is ever called.
- Only your own sent messages are ingested; no credentials are stored anywhere.

## Known limitations

Measured, not hypothetical:

- **Number grounding checks presence, not attribution.** When facts are supplied, the
  validator rejects any number that does not appear in them — it caught the 7B
  inventing "50%" against a real 82.6%. It cannot tell that a correctly-quoted number
  was attached to the wrong claim (the 7B once labelled the 73.5% token reduction as a
  call reduction). Verifying attribution would need an LLM judge, which would spend
  exactly the tokens this architecture exists to save.
- **Generation quality is model-bound.** Qwen1.5-1.8B needs every deterministic guard
  (title stripping, length enforcement, placeholder and markdown checks) and still
  drifts toward content found in its style examples. Qwen2.5-7B writes a clean,
  in-voice post in a single call. Use `JLA_MODEL_PREFER=large` when the post matters.
- **The style back-test reproduces shape better than voice** (mean style score 0.58,
  hook archetype match 50%). Length, CTA and hashtag behaviour transfer well; tone
  does not fully.
- **The intent model is weak on unseen phrasings** (4-fold CV accuracy 0.54 on 82 seed
  examples). It only runs when no rule matches, and low confidence escalates to the
  LLM, so the failure mode is "pays for a model call", not "wrong answer".
- **Topic classes are tiny** (business: 2 documents). Cross-validation runs with 2
  folds and reports 0.91 accuracy — real, but on a small and imbalanced sample.
- **136 posts is a small corpus.** Everything above is a consequence of that.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "no style profile" | run `ingest_linkedin.bat` first |
| First semantic query takes ~20 s | MiniLM cold load; warm-up does it at startup — check `/api/health` shows `state: ready` |
| Model load is slow / RAM pressure | switch Settings → Model preference to *Small*, or set `JLA_MODEL_PREFER=small` |
| Generation runs long or adds titles | expected with the 1.8B; cleaned deterministically, use `large` for better output |
| `no posts found` after ingest | you pointed at a *Basic* export; use the full archive |
| `TypeError` from `llama_cpp` at exit | harmless teardown noise from the library; the manager releases the model via `atexit` |
| Port already in use | `JLA_PORT=8078` and update `frontend/vite.config.ts` proxy |

## Project layout

See `CLAUDE.md` for the module map and the rules that keep the architecture honest.
