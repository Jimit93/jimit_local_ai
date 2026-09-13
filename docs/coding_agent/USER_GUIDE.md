# Coding agent — user guide

## Start it

1. Double-click `start_agent.bat` in the project folder. It starts the backend (127.0.0.1:8077,
   CPU-capped) and the UI (127.0.0.1:5173) and opens the browser.
2. The left sidebar opens on **Coding Agent**. A green dot bottom-left means the backend is ready.

Ollama must be running for code models (it starts with Windows by default). Without it the agent
still works for every tool-based request and falls back to the general local model for questions.

## Give it a coding task

1. Pick the project in the workspace box (top right). To add another project, go to
   **Workspace** → type its folder path → **Register**. System folders and your whole user folder
   are refused on purpose.
2. Type the request and press **Run** (or Ctrl+Enter). Examples:
   * `where is the function run_task defined` — answered by a tool, no model
   * `run tests/agent/test_agent_policy.py` — runs pytest, CPU-capped
   * `fix the failing test in tests/test_calc.py` — reproduce → locate → edit → verify
   * `explain how the Governor class decides to defer heavy work`
   * `rename function add to add_numbers` — done without a model
   * `review my changes`
3. **Allow edits** (on by default) lets the agent change files. Every change is checkpointed and
   verified by tests; unverified changes are rolled back. Turn it off to get explanations only.
4. **Deep review** adds a model pass to `review my changes`.

Follow-ups in the same session: `undo that`, `again`, `continue`.

## Watch what it is doing

On the **Coding Agent** page:

* **Agent status** — current task, step, method (TOOL / NLP / MEMORY / LLM), CPU with the
  governor state, free RAM, the model in use, context size, tokens, LLM calls.
* **Workflow** — every step: ✓ done, ● running, ○ pending, ✕ failed, with a one-line summary.
  **Cancel** stops at the next step boundary.
* **Harness view** — the route the request took: Request → Classifier → Tool/NLP/ML/LLM →
  Validator → Evaluator → Repair → Result. The highlighted box is the active stage.
* **Routing decisions** — why each method was chosen, when the CPU governor made a step wait,
  when RAM forced a smaller model, repairs and rollbacks. This is a factual log, not the model's
  private reasoning.
* **Result** — the answer or the diff, **Undo change**, and **Predicted vs actual** (files, tests,
  LLM calls against what was expected before the task ran).

If a task needs permission (for example deleting a file), a yellow **Approval needed** box appears.

## Inspect memory and checkpoints

**Memory & Checkpoints** page:

* **Project memory** — languages, test command, conventions read from CLAUDE.md, most-connected
  files; searchable without a model.
* **Decisions & known issues** — record a decision the agent must respect; see recurring errors
  and how they were fixed.
* **Unfinished tasks** — anything interrupted by a crash, restart or closed window. **Check**
  shows the resume analysis (was the project changed? which step is safe?), **Resume** continues.
* **Checkpoints** — every snapshot, with **Diff** and **Rollback**. Rollback never overwrites a
  file you edited after the agent.
* **RAM** — processes by category. The agent frees RAM automatically only from its own models and
  from apps you add to the allowlist; **Close** shuts one app after you confirm. System and
  security processes are never offered.
* **Audit log** — every allowed, denied or approved action.

## View evaluations

**Agent Evaluation** page: CPU and RAM over the last hour (with the 70% cap line), where work was
done (tools vs NLP vs memory vs LLM), method per step, predicted-vs-actual per task, LLM budget vs
used, durations, repairs, and the latest CPU, ranking and system backtests.

Re-run the measurements from a terminal in the project folder:

```bash
python -m scripts.agent_backtest
```

```bash
python -m scripts.restart_backtest
```

```bash
python -m scripts.stress_backtest
```

```bash
python -m scripts.cpu_backtest
```

## Resume previous work

After a restart the backend re-opens unfinished tasks by itself:

* project unchanged → it resumes automatically from the safe step (at most twice per task);
* files changed while it was stopped → it waits; open **Memory & Checkpoints → Unfinished tasks**,
  press **Check** to see what changed, then **Resume** (it re-inspects before editing).

Typing `continue` in the same chat session also resumes the last task.

## Settings you may want

| Environment variable | Default | Effect |
|---|---|---|
| `JLA_CPU_CAP` | 70 | CPU percent the governor protects |
| `JLA_CODE_MODEL_SMALL` | qwen2.5-coder:3b | small code model |
| `JLA_CODE_MODEL_CAPABLE` | deepseek-coder-v2:16b | capable code model |
| `JLA_WORKSPACES` | this project | extra workspaces, `;`-separated |
| `JLA_RAM_KILL_ALLOW` | empty | apps the agent may close to free RAM, `;`-separated |
| `JLA_AGENT_AUTO_RESUME` | 1 | resume safe unfinished tasks on startup |
| `JLA_SELF_CPU_CAP` | 1 in `start_backend.bat` | hard CPU cap for the backend process tree |
