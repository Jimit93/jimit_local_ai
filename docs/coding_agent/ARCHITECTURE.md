# Coding agent — architecture

A local coding assistant built into the existing routing harness. It inspects, edits and
tests code in registered workspaces, remembers its work across restarts, and keeps the
machine responsive. Every number quoted in this document links to the report it came from.

## 1. One orchestrator: the routing harness

There is no separate "DeepSeek Harness" on this machine (see PLAN.md §1). The existing
harness in `backend/harness/` is the single orchestrator; the agent is plugged into it:

```
request ─► task_classifier (rules → TF-IDF/LogReg)
            │
            ├─ deterministic / text / corpus categories ─► existing processor chains
            ├─ workspace ─► [p_workspace_tool ─► p_coding_agent]
            └─ coding    ─► [p_workspace_tool ─► p_coding_agent]
```

`p_workspace_tool` answers a deterministic coding intent with one sandboxed tool call.
`p_coding_agent` runs the controlled loop for everything else. The LLM is never first in a
chain (enforced by `tests/test_router.py`), and all 105 previously classified phrasings route
exactly as before (`tests/agent/classifier_baseline.json`).

## 2. Request understanding (`backend/agent/intents.py`)

| Stage | Method | Notes |
|---|---|---|
| Rules | anchored regexes | ~35 intents; order matters (specific first) |
| ML fallback | TF-IDF (word + char n-grams) + Logistic Regression | trained on seed phrasings; retrains when the seed changes (seed hash); 2-fold CV stored in `data/models/coding_intent_clf.json` |
| Low confidence | → `general_coding` | always safe: context + model answer, no edits |
| Parameters | regexes | files, directories, globs, line ranges, rename pairs, test targets, traceback presence |
| Complexity | transparent score | intent base + multi-file + error + length + hard topic → low/medium/high → model tier order |

Deterministic intents (≈24): file exists/info/read/list/outline, search files/code/symbols,
git status/diff/log, run tests, lint, syntax check, typecheck, parse JSON/CSV, system status,
memory query, checkpoints, rollback, project overview, rename/delete file, calculate.
Agent intents: explain, fix bug, analyze error, implement, refactor, rename symbol, write tests,
review, create file, general coding.

## 3. The controlled loop (`backend/agent/loop.py`, `planner.py`)

```
UNDERSTAND ─► PLAN ─► for each step: SELECT METHOD ─► EXECUTE ─► VERIFY
                                              ▲                      │ fail
                                              └── REPAIR (bounded) ◄─┘
      ─► REPORT + MEMORY + PREDICTED-vs-ACTUAL
```

Plan templates (deterministic, per intent):

| Intent | Steps |
|---|---|
| deterministic | `run_tool` |
| fix bug (tests present) | reproduce → locate → gather context → propose edit → apply (checkpoint) → verify syntax → verify lint → verify tests → report |
| implement / refactor / write tests | locate → context → propose → apply → syntax → **requirements** → lint → tests → report |
| explain / analyze / read-only mode | locate → context → answer → check answer → report |
| rename symbol | locate → token-level rename (no model) → syntax → tests → report |
| review | diff → automated review → model review (only when asked/complex) → report |

Loop contract — every task has:

| | |
|---|---|
| Entry | a persisted task with a plan |
| Exit | all steps done → `completed`; unrecoverable step → `failed`; limit hit or same failure twice → `blocked`; user → `cancelled`; approval needed → `awaiting_approval` |
| Limits | iteration budget = plan length + repairs×6 + 4 (hard cap 40); per-step attempts; `max_repairs` = 2; task deadline (`JLA_AGENT_TASK_TIMEOUT`, default 3600 s); subprocess timeouts |
| Repair | a failing verification returns to its responsible step (`propose_edit`), rolls the files back first, feeds the exact failure into the prompt, and escalates the model tier |
| Repeat guard | error signature (exception type + message, paths/numbers removed); the same signature after a repair stops the task |
| Unverified changes | rolled back automatically when a task ends failed/blocked |

Method selection per step is cheapest-first: deterministic tool → NLP ranking → memory replay /
cache → small code model → capable code model. Every choice is written to the timeline with a
one-line reason ("Deterministic tool used. Reason: the request maps to an exact tool…").
No chain-of-thought is stored or shown.

**Fault localisation without a model.** For a failing test, the loop reads the test function's
body, collects the functions it calls and maps them through the symbol index to their defining
source files (`functions_under_test`). Only code files are editable unless the request names
another file; a bug fix never edits the tests unless the request says the test is wrong.

**Edits** (`edits.py`): the model proposes SEARCH/REPLACE blocks; the module decides. A SEARCH
part must match exactly once (fallbacks: trailing whitespace, uniform re-indentation in either
direction, with REPLACE shifted by the same offset). An empty SEARCH on an existing file appends
(implement / write-tests / create only). Identical blocks from generation loops are applied once.
A placeholder path (`path/to/file.py`) is re-targeted only when exactly one file is editable and
the SEARCH matches it uniquely. Refused: files outside the allowed set, secret files, redacted
content, rewrites deleting >40% of a file, code that does not parse (the retry feedback shows the
resulting lines around the error). A proposal is dry-run before the checkpoint is taken.

**Requirements gate** (`h_verify_requirements`): existing tests cannot see new code, so for
implement / create / refactor / write-tests the functions the request spells out as calls
(`truncate(text, n)`) must exist at module level with that parameter count, a write-tests task
must produce `test_*` functions, and existing functions the request did not mention must be
unchanged. Two measured small-model habits are normalised deterministically first: a requested
function written after a `return` inside another function is moved to module level, and
unrequested changes to existing functions are restored from the checkpoint. Found by the system
backtest's hidden tests: without this gate, a task "passed" after editing the wrong function.

## 4. Layers and when each is used

| Layer | Used for | Implementation |
|---|---|---|
| Deterministic | calculator, file/git/test/lint tools, AST symbols, token rename, static review, validators | `code_tools.py`, `symbols.py`, `edits.py`, `executor.py` |
| NLP | file relevance (BM25 over identifier tokens fused with mention/symbol/traceback/module-name/import signals), memory search, follow-up resolution | `project_index.py`, `memory.py` |
| ML | intent classification fallback | `intents.py` (TF-IDF + LogReg) |
| Memory / cache | replay a verified fix for a recurring error; identical prompt over identical code | `memory.py`, `loop._llm_call` |
| Small code model | answers, first edit attempt | Ollama `qwen2.5-coder:3b` |
| Capable code model | complex tasks, escalation after a failed attempt | Ollama `deepseek-coder-v2:16b` (needs ≈11 GB free RAM) |
| General model | fallback when Ollama is unreachable | llama.cpp Qwen via `llm/manager.py` |

Measured ranking quality: `reports/agent/ranking_eval.json`. Embeddings were evaluated as an
option and not adopted (MiniLM load ≈20 s on this CPU; lexical fused ranking already at parity
on docstring queries and 4.5× better P@1 on module-name queries).

## 5. Models, RAM and the completion guarantee (`backend/llm/manager.py`)

* Models are constructed only in `manager.py` (project rule 5).
* Before a code model loads, RAM is checked (size × 1.15 + 1 GB floor) and reclaimed:
  tier 0 unloads idle agent-owned models automatically; tier 1 closes processes on the user's
  allowlist; anything else is only offered for per-process approval (`ram.py`). If the model
  still does not fit, the manager degrades to the largest installed code model that does, and
  the timeline records it.
* Replies end inside `max_tokens` on a complete sentence (`harness/completion.py`): prompts
  carry a word budget; a truncated reply is trimmed deterministically (last sentence / list
  item; dangling heading or lead-in and unclosed code fences removed); an unterminated edit block
  is dropped rather than applied. Measured: `reports/agent/completion_eval.json`.
* Thread count derives from the CPU cap (`config._default_threads`), measured in
  `reports/agent/cpu_backtest_phase2.json`.

## 6. Context pipeline (`backend/agent/context.py`)

request → ranked files → symbols the request/traceback point at → line ranges (whole file only if
≤150 lines) → project memory → task history → error output → labelled blocks under a character
budget (`harness.context_manager.build_context`), scaled down by the governor under load, with
secrets redacted. The accounting ("N chars of the repository not sent") is shown in the UI.

## 7. Memory, state, checkpoints, resume

| Layer | Store | Content |
|---|---|---|
| Session | `agent_messages` | requests + results; deterministic follow-ups ("undo that", "again", "continue") |
| Project | `agent_memory` kind `project` | languages, manifests, test command, conventions (from CLAUDE.md), most-connected files |
| Task | `agent_tasks.state_json` | plan, steps, active files, decisions, validation results, next action, owner process |
| Persistent | `agent_memory` kinds `decision`, `error_pattern` | decisions; error signatures with the verified fix (replayable) |

Checkpoints (`checkpoints.py`) store exact bytes of every file before the agent modifies it.
Rollback restores them and refuses to overwrite a file someone else changed afterwards.

Resume (`state.py`): on startup, tasks whose owning process is dead are marked `interrupted`,
then analysed — load → validate → project changed? → last checkpoint → last validation → safe
resume point. Changed files ⇒ resume from re-inspection and never automatically. A step that
had already applied its edit continues with verification; a half-written edit is restored from
its checkpoint and re-run. Corrupted state ⇒ `blocked`, never a crash.
Measured: `reports/agent/restart_backtest.json`.

## 8. Resource governor (`backend/agent/governor.py`, `proc.py`)

| State | System CPU | Behaviour |
|---|---|---|
| NORMAL | < 60 % | normal; 2 heavy jobs; full context |
| CAUTION | 60–70 % | 1 heavy job; context × 0.75; max tokens × 0.8 |
| HIGH_LOAD | > 70 % | no new heavy work (waits, bounded); child CPU caps tightened; prefer tools |
| EMERGENCY | > 90 % or RAM < 1 GB | stop starting work; reclaim agent RAM |

Entry reacts to a single sample; exit needs the moving average below the entry line minus 5
points for 3 consecutive samples. Enforcement: admission for heavy work (LLM, tests, builds),
Windows Job Objects with a live-adjustable CPU hard cap for every child process (tests,
typecheck), inference threads derived from the cap, and an optional hard cap on the server
process itself (`JLA_SELF_CPU_CAP=1`, set by `start_backend.bat`).
Measured: `reports/agent/cpu_backtest_phase2.json`, `cpu_protection_cap40.json`.

## 9. Security

* Workspaces must be registered; system folders, the whole profile and credential folders are
  refused. Every path is resolved (symlinks included) and must stay inside the workspace.
* Secret files (`.env`, keys, `id_rsa`, …) are never read; secret-looking values are redacted
  before anything reaches a model, the database, the timeline or the UI.
* Commands: argv only (no shell), allowlist (read-only git, `python -m pytest/pyflakes/…`,
  `npm run build/test/lint`, tsc/vite from node_modules), scrubbed environment, offline flags.
* Approvals: deleting files, running project scripts and killing processes need an explicit user
  approval the loop cannot grant itself. Everything is written to `agent_audit`.
* The server binds to 127.0.0.1; the only outbound call is Ollama on 127.0.0.1:11434.

## 10. Testing and evaluation

| What | Where |
|---|---|
| Unit / integration (tools, policy, governor, RAM, index, context, memory, state, intents, edits, loop, API) | `tests/agent/`, `tests/test_completion.py` |
| System backtest (real requests, real models, hidden tests, safety, resilience; predicted vs actual) | `scripts/agent_backtest.py` → `reports/agent/system_backtest*.json/.md` |
| Restart/resume with real process kills | `scripts/restart_backtest.py` → `reports/agent/restart_backtest.json` |
| Stress (3,000+ file repo, long session, bursts, failures, CPU and RAM pressure) | `scripts/stress_backtest.py` → `reports/agent/stress_backtest.json` |
| CPU backtest | `scripts/cpu_backtest.py` → `reports/agent/cpu_*.json` |
| Completion within token limits | `scripts/completion_eval.py` |
| Dashboard | UI → Agent Evaluation (reads `/api/agent/evaluation`) |

## 11. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Task fails with "InsufficientRAM" | another process holds the RAM (often a second backend with a model loaded) | close it, or allowlist an app in Memory → RAM; the agent already degrades to a smaller model when one fits |
| Steps wait with a "governor" event | CPU above the cap | expected; heavy work resumes when load drops |
| "approval required" | delete / script / kill | approve or deny in the Agent page banner |
| Task `interrupted` after restart and not resumed | files changed while stopped, or auto-resume limit reached | Memory → Unfinished tasks → Check → Resume |
| `blocked: the same failure repeated` | the model cannot fix it with the same approach | the change was rolled back; rephrase, add detail, or free RAM for the capable model |
| Answer ends with "…" | the token limit left no complete sentence | raise max tokens or ask a narrower question |
| Audit log shows no UI git polling | by design: read-only UI polls are not audited | denials, approvals and agent-task commands are always audited |
