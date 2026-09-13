# Feature backlog

Discovered while running the QA evaluation task. Nothing here is implemented
unless it is marked IN CURRENT TASK. Items marked *(needs verification)* were
noticed in passing and have not been confirmed against the code yet.

Classification: value / implementation type / regression risk.

---

## HIGH PRIORITY

### Validator cannot detect unused evidence
- **Description**: `validate_post` checks that numbers used are a subset of the
  supplied facts, but never that supplied facts were used. A post citing nothing
  passes `numbers_grounded` vacuously and scores `style_match` 1.0.
- **Why needed**: measured — 18 numbers supplied, 0 cited, validator returned
  SUCCESS at 100%. The QC metric is currently misleading.
- **Component**: `backend/harness/validator.py`
- **Class**: CRITICAL / DETERMINISTIC / low risk
- **Status**: IN CURRENT TASK (iteration 2)

### Content plan ignores measured usage rates
- **Description**: `build_plan` sets `cta = preferred_ctas[0]` unconditionally,
  even when `style_constraints.cta_expected` is False. `plan["hashtags"]` takes
  the p75 (5) although hashtag median is 0 and usage 36%.
- **Why needed**: measured — `cta_expected` is False (20.8% < 40%) yet the prompt
  still ordered "End with a call to action of type: hiring."
- **Component**: `backend/services/linkedin_writer.py: build_plan`
- **Class**: CRITICAL / DETERMINISTIC / low risk
- **Status**: IN CURRENT TASK (iteration 1) — CTA half only; the hashtag half is
  queued below.

### Internal CTA label leaks into the prompt as instruction text
- **Description**: the label `hiring` is emitted verbatim as "a call to action of
  type: hiring". Measured: of 9 corpus posts whose CTA matches that pattern, 0
  announce a vacancy — they say "open to", "opportunities", "collaboration".
- **Why needed**: produced a factual inversion in the baseline output ("Hiring for
  more engineers") aimed at an audience of hiring managers.
- **Component**: `backend/services/style_engine.py` (add `CTA_GUIDANCE`, mirroring
  the existing `HOOK_GUIDANCE`)
- **Class**: CRITICAL / DETERMINISTIC / low risk
- **Status**: IN CURRENT TASK (iteration 1)

### Planned hook is never verified against the generated hook
- **Description**: `build_plan` picks `hook_type`, the prompt asks for it, and
  nothing checks the result. Baseline planned `claim`, produced `announcement`.
- **Why needed**: the plan is presented as deterministic control but is currently
  advisory only; a silent mismatch is invisible in the UI and in `style_match`.
- **Component**: `backend/harness/validator.py` (+ pass `plan` into `validate_post`)
- **Class**: HIGH VALUE / DETERMINISTIC / low risk
- **Effort**: small
- **Status**: BACKLOG

### Model selection can leave the machine RAM-starved
- **Description**: auto-selection loaded qwen2.5-7b (RSS 7.9 GB) leaving 1.13 GB
  free (93.2% used); free RAM fell to 0.38 GB during generation.
- **Why needed**: reliability. A second process (the evaluator, ingestion) can hit
  swap or OOM while the model is resident.
- **Component**: `backend/llm/discovery.py: select_model`
- **Class**: HIGH VALUE / INFRASTRUCTURE / medium risk (changes model choice)
- **Dependency**: needs a measured headroom threshold, not a guessed one.
- **Status**: BACKLOG

---

### Marketing-register copy passes every check
- **Description**: a refine-loop run through the UI returned SUCCESS 1.0 on a post
  containing "Our AI agent is the perfect solution for you", "freeing up valuable
  time" and "takes your team off the clock". None is on the banned list, and the
  author writes as "I", not "our".
- **Why it matters**: extending the phrase list one draft at a time is whack-a-mole.
- **Better fix**: an NLP register feature — second-person sales framing ("are you a
  … looking to", "perfect solution for you"), superlatives, first-person-plural
  company voice — measured against the 77 posts for a false-positive rate first,
  exactly as the phrase additions were.
- **Class**: HIGH VALUE / NLP / low risk · **Status**: BACKLOG

### Two truncation trims now exist
- **Description**: the coding-agent session added `backend/harness/completion.py`
  (`finish` / `trim_prose`), which `LLM.generate` now runs on every call. The writer
  has its own `linkedin_writer._finish`. They overlap but are not interchangeable:
  `trim_prose` counts a colon before a newline as a finished sentence and appends
  "…" when nothing complete is left - both wrong for a LinkedIn post (the exact
  failure in the pasted draft) - and it does not preserve a trailing hashtag line.
- **Why it matters**: two trims on one path is how a future edit to one silently
  breaks the other. Today the writer tolerates the runtime trim (a capped reply
  ending in "…" is treated as unfinished; covered by a regression test).
- **Action once `completion.py` is committed**: give `trim_prose` a post mode (no
  colon-as-ending, no ellipsis, keep hashtag tail) and delete `_finish`.
- **Class**: USEFUL / DETERMINISTIC / low risk · **Status**: BACKLOG — not touched
  now because the file is uncommitted work in another session.

## MEDIUM PRIORITY

### No retrieval-quality evaluation
- **Description**: `linkedin_backtest` measures *style reproduction*, but nothing
  measures whether retrieval returned relevant examples (precision@k, or whether
  a more relevant post existed and was missed).
- **Why needed**: retrieval is upstream of every generation failure, and the RAG
  audit cannot currently answer "was retrieval at fault?" with a number.
- **Component**: `backend/retrieval/index.py` + new eval in `backend/benchmarks/`
- **Class**: HIGH VALUE / RETRIEVAL / low risk (read-only measurement)
- **Status**: BACKLOG

### Hashtag target ignores usage rate
- **Description**: see "Content plan ignores measured usage rates". The plan asks
  for up to 5 hashtags while the corpus median is 0 (36% of posts use any).
- **Component**: `backend/services/linkedin_writer.py: build_plan`
- **Class**: USEFUL / DETERMINISTIC / low risk
- **Status**: BACKLOG (deliberately excluded from iteration 1 to keep the
  before/after comparison attributable to one change)

### Frontend health polling is aggressive
- **Description**: the UI issued dozens of `GET /api/health` calls per minute
  while idle; each one touches `psutil` and the model manager.
- **Why needed**: wasted CPU on a machine where CPU is the scarce resource.
- **Component**: `frontend/src/services/api.ts` / the polling hook
- **Class**: USEFUL / FRONTEND / low risk
- **Status**: BACKLOG

### Feedback is not captured as a durable, classified record
- **Description**: user feedback ("too generic", "wrong audience") has nowhere to
  live. `agent_memory` exists and is the natural store.
- **Why needed**: requested — feedback must be classified to a layer and become a
  regression test, not appended to a prompt.
- **Component**: `backend/db.py` (`agent_memory`), new API route, Writer UI
- **Class**: HIGH VALUE / BACKEND+FRONTEND / medium effort
- **Status**: IN CURRENT TASK (scoped to classification + persistence)

---

## LOW PRIORITY

### Settings page coverage *(needs verification)*
- Not yet audited against `SettingsIn` (prefer_size, temperature, max_tokens,
  n_ctx, max_style_examples). Retrieval `top_k` and layer thresholds are not
  exposed. **Class**: OPTIONAL / FRONTEND / low risk.

### Source attribution in generated output *(needs verification)*
- The API returns example `doc_id`s and the UI shows previews, but the post itself
  carries no provenance. Probably correct as-is for a LinkedIn post.
  **Class**: OPTIONAL.

---

## REJECTED / NOT NEEDED

### A second LLM pass to critique the post
- An LLM judge would cost another call on a CPU-only box to measure things the
  deterministic validator already measures better and for free. Rejected — this is
  exactly the escalation the harness exists to prevent.

### A separate evaluation database / vector store
- `agent_memory`, `task_runs` and `benchmark_runs` already exist in SQLite and are
  sufficient. Rejected under the no-duplicate-systems rule.

### Fine-tuning Qwen on the corpus
- Premature. The two measured failures are both below the model and fixable
  deterministically. Revisit only if prompt/context/validation fixes plateau.
