# Coding agent — inspection findings, capability matrix, plan

Written 2026-09-13 after inspecting the repo and the machine and before any code change.
All numbers are measured on this machine.

## 1. Key finding: there is no "DeepSeek Harness"

A search of this repo and of every likely project folder under `C:\Users\pc` found no
DeepSeek harness. The only DeepSeek artefact is the Ollama model
`deepseek-coder:6.7b-instruct-q4_K_M` (4.08 GB).

The prompt says the DeepSeek Harness is to be extended *if one exists* and that no
competing orchestrator may be created. So:

- The **existing routing harness** (`backend/harness/`: classify → cheapest-first processor
  chain → confidence threshold → verify → record) is the central orchestrator. The coding
  agent plugs into it as processors and chains. It does not replace it.
- `deepseek-coder:6.7b` is used as the **capable code-model tier**, under that name.
  Nothing is renamed "DeepSeek" that is not DeepSeek (project rule 4).

## 2. Environment (measured)

| Item | Value |
|---|---|
| CPU | 18 logical / 14 physical, no CUDA |
| RAM | 16.7 GB total, **3.76 GB available** at inspection |
| Idle system CPU | 14.3 % (1 s sample) |
| Python / Node / git | 3.11.9 / 24.18.0 / 2.50.0 |
| Present and reusable | psutil 7.2, fastapi, sklearn 1.7, rank_bm25, faiss, sentence-transformers, llama-cpp, **pyflakes**, **playwright + chromium**, **pywin32 (win32job)** |
| Absent | ruff, black, jedi, vitest, tree-sitter grammars |
| Ollama models | deepseek-coder 6.7b (4.08 GB), qwen2.5-coder 0.5b (0.40 GB), llama3 8b, gemma3 1b |
| GGUF (llama.cpp) | Qwen2.5-7B q4_0 (4.43 GB), Qwen1.5-1.8B q4_k_m (1.22 GB) |
| llama.cpp threads | `n_threads = cpu_count - 2 = 16` → **conflicts with the 70 % CPU cap** (to measure) |

Baseline test suite: **166 passed, 2 skipped, 65.6 s**. Baseline benchmark (`b79f795399`):
harness 23/23 correct, 4 LLM calls, 2,214 tokens.

## 3. Capability matrix

| Feature | Exists | Working | Partial | Missing | Risk | Action |
|---|---|---|---|---|---|---|
| Routing harness | yes | yes (tests) | | | high — everything depends on it | **Extend**: new `workspace` category + agent processor |
| Task classifier (rules → TF-IDF/LR) | yes | yes | stale joblib is reused after SEED changes | | low | Reuse; retrain on seed-hash mismatch |
| Calculator | yes | yes | `345 × 67` fails (`×`, `÷` not normalised) | | low | Minimal fix |
| FS tools | `file_metadata`, `list_dir` | yes | no sandbox | read/write/edit/rename/delete/search | medium | New `code_tools` behind a policy |
| Git tools | | | | yes | low (read-only) | Add status/diff/log |
| Tests / lint / syntax tools | | | | yes | medium (subprocess) | Add, CPU-capped, timeouts |
| Command execution + guardrails | | | | yes | **high** | Allowlist, argv only, no shell, audit |
| NLP (tokenize, TF-IDF, BM25) | yes (corpus) | yes | no code index | | low | Reuse `BM25Index` |
| Embeddings (MiniLM/FAISS) | yes (corpus) | yes | | code embeddings | medium CPU | Only if measured to help |
| LLM manager | yes | yes | general Qwen only, no code tier, no RAM guard between runtimes | | **high** (RAM 3.8 GB) | Extend inside `manager.py` (rule 5) |
| Context budgeting | yes | yes | no code context | | low | Reuse `build_context` |
| Memory | `agent_memory` table | yes (prefs, feedback) | no layers | session/task/project | low | Layered API over the table |
| Task state / checkpoints / resume | | | | yes | medium | New tables + module |
| CPU governor | `metrics.snapshot` only | | | yes | medium | New, measured |
| Validation | post/numeric/generic | yes | no code validators | | low | Add syntax/test/diff-scope checks |
| Observability | `task_runs`, jsonl | yes | no step timeline | | low | Add agent events |
| Evaluation | LinkedIn eval + benchmark | yes | nothing for coding | | low | Add agent backtest |
| Frontend | 8 pages, SVG charts | yes | no agent UI | | medium | Add pages; reuse `Bits`, `EvalGraphs` |
| Audit log | | | | yes | | Add |
| Frontend/E2E tests | `tsc -b` only | | | yes | | Playwright (already installed) |

Must not be touched beyond the listed minimal fixes: ingestion, chunking, retrieval index,
LinkedIn writer, style engine, existing chains, existing tests.

## 4. Architecture

```
POST /api/agent/run ─► worker queue (governor-gated) ─► router.run_task(request, workspace)
                                                            │
                                  classify_task (rules → TF-IDF/LR)
                                                            │
             ┌──────────────────────────────┬───────────────┴────────────────┐
     workspace chain                  coding chain                   existing chains
 (deterministic tools:          p_coding_agent (hybrid):             (calc, search, …)
  read/search/git/tests…)       controlled loop below
                                                            │
  intent (rules → TF-IDF/LR) → plan template → for each step:
     governor.admit → select method (tool → NLP → ML → small LLM → capable LLM)
     → execute → validate → predicted-vs-actual → repair (bounded) → checkpoint → state
```

New package `backend/agent/`:

| Module | Responsibility |
|---|---|
| `policy.py` | workspace sandbox, path guard, secret detection/redaction, command allow/deny, approval gates |
| `code_tools.py` | read/write/edit/rename/delete(soft)/list/search files, code, symbols; git; tests; lint; syntax; process monitor |
| `governor.py` | background sampler, NORMAL/CAUTION/HIGH_LOAD/EMERGENCY, admission control, thread caps, subprocess CPU cap |
| `project_index.py` | incremental file index (mtime/size/sha1), Python AST + TS regex symbols, import graph, BM25 file ranking |
| `context.py` | request → files → symbols → snippets → memory → errors → budgeted package |
| `memory.py` | session / project / task / persistent layers over `agent_memory` |
| `state.py` | task state JSON, checkpoints (file snapshots), rollback, startup resume analysis |
| `intents.py` | coding intent classifier (rules → TF-IDF/LR), complexity estimate |
| `edits.py` | parse and apply SEARCH/REPLACE edit blocks deterministically |
| `loop.py` | controlled agent loop with entry/exit/success/failure, max iterations, timeouts, repeat-failure stop |
| `worker.py` | background execution, cancellation, heartbeat |

CPU policy: llama.cpp threads and Ollama `num_thread` are capped from the governor;
subprocesses (pytest, tsc) are placed in a Windows Job Object with a CPU rate cap.
All caps are validated by the CPU backtest, not assumed.

## 5. Phases (each ends with tests + a git checkpoint on `feature/coding-agent`)

0. Inspection, plan, checklist, baseline — **this document**
1. Policy + code tools + calculator fix
2. Resource governor + CPU measurement
3. Project index + context pipeline + ranking evaluation
4. Memory + task state + checkpoints + resume
5. Intents + LLM code tiers + harness integration
6. Controlled agent loop + edit application + repair
7. API + worker
8. Frontend (agent cockpit, workspace, memory, harness view, evaluation)
9. Backtests: E2E, CPU, restart/resume, stress, predicted-vs-actual, graphs
10. Feature audit → improvements → before/after → full retest
11. Docs, security audit, final acceptance
