# Benchmark: LLM-only baseline vs routing harness

Run `b79f795399` · 2026-09-12 21:18 · 356.4s total

## Environment

- Model: `qwen1_5-1_8b-chat-q4_k_m.gguf` (1.22 GB) via llama.cpp, n_ctx 4096, 16 threads
- Machine: 18 logical cores, 16.7 GB RAM, GPU: False
- Warm-up (excluded from task timings): 24256.8 ms

## Totals

| Metric | Baseline (LLM-only) | Harness | Change |
|---|---|---|---|
| Tasks | 23 | 23 | |
| Correct | 9 | 23 | +60.9 pts |
| Accuracy | 39.1% | 100.0% | +60.9 pts |
| LLM calls | 23 | 4 | −82.6% |
| Tokens | 8364 | 2214 | −73.5% |
| Total latency | 283.8s | 48.4s | −83.0% |
| Median latency | 13352 ms | 2 ms | |
| CPU time | 2575.5s | 477.8s | −81.5% |
| Peak RSS | 3030 MB | 3027 MB | |
| Tasks with no LLM call | 0 | 19 | |

## By suite

| Suite | Harness acc. | Baseline acc. | Harness LLM calls | Baseline LLM calls | Harness latency | Baseline latency |
|---|---|---|---|---|---|---|
| classification | 100% | 0% | 0 | 3 | 0.0s | 49.0s |
| deterministic | 100% | 80% | 0 | 5 | 0.0s | 20.9s |
| generation | 100% | 0% | 2 | 2 | 30.2s | 36.5s |
| nlp | 100% | 33% | 0 | 3 | 0.0s | 14.8s |
| ranking | 100% | 0% | 0 | 1 | 0.0s | 20.0s |
| reasoning | 100% | 100% | 2 | 2 | 18.1s | 22.3s |
| retrieval | 100% | 0% | 0 | 2 | 0.0s | 40.9s |
| semantic_search | 100% | 100% | 0 | 2 | 0.0s | 38.0s |
| structured_data | 100% | 0% | 0 | 3 | 0.0s | 41.4s |

## Per task

| Task | Suite | Mode | Correct | Score | Processor | LLM calls | Tokens | Latency |
|---|---|---|---|---|---|---|---|---|
| det_1 | deterministic | harness | yes | 1.00 | `tool:calculate` | 0 | 0 | 4 ms |
| det_1 | deterministic | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 126 | 1882 ms |
| det_2 | deterministic | harness | yes | 1.00 | `tool:calculate` | 0 | 0 | 1 ms |
| det_2 | deterministic | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 130 | 971 ms |
| det_3 | deterministic | harness | yes | 1.00 | `tool:calculate` | 0 | 0 | 1 ms |
| det_3 | deterministic | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 225 | 5189 ms |
| det_4 | deterministic | harness | yes | 1.00 | `tool:calculate` | 0 | 0 | 2 ms |
| det_4 | deterministic | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 121 | 701 ms |
| det_5 | deterministic | harness | yes | 1.00 | `tool:date_diff` | 0 | 0 | 3 ms |
| det_5 | deterministic | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 375 | 12184 ms |
| nlp_1 | nlp | harness | yes | 1.00 | `nlp_text_stats` | 0 | 0 | 2 ms |
| nlp_1 | nlp | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 220 | 3652 ms |
| nlp_2 | nlp | harness | yes | 1.00 | `nlp_text_stats` | 0 | 0 | 1 ms |
| nlp_2 | nlp | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 178 | 1710 ms |
| nlp_3 | nlp | harness | yes | 1.00 | `nlp_text_stats` | 0 | 0 | 1 ms |
| nlp_3 | nlp | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 327 | 9467 ms |
| cls_1 | classification | harness | yes | 1.00 | `rule_keyword_topics` | 0 | 0 | 12 ms |
| cls_1 | classification | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 421 | 16392 ms |
| cls_2 | classification | harness | yes | 1.00 | `rule_keyword_topics` | 0 | 0 | 2 ms |
| cls_2 | classification | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 325 | 11462 ms |
| cls_3 | classification | harness | yes | 1.00 | `rule_keyword_topics` | 0 | 0 | 2 ms |
| cls_3 | classification | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 464 | 21100 ms |
| ret_1 | retrieval | harness | yes | 1.00 | `exact_keyword_search` | 0 | 0 | 2 ms |
| ret_1 | retrieval | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 509 | 22236 ms |
| ret_2 | retrieval | harness | yes | 1.00 | `exact_keyword_search` | 0 | 0 | 3 ms |
| ret_2 | retrieval | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 493 | 18621 ms |
| sem_1 | semantic_search | harness | yes | 0.50 | `hybrid_retrieval(bm25+minilm)` | 0 | 0 | 28 ms |
| sem_1 | semantic_search | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 510 | 18878 ms |
| sem_2 | semantic_search | harness | yes | 1.00 | `hybrid_retrieval(bm25+minilm)` | 0 | 0 | 21 ms |
| sem_2 | semantic_search | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 509 | 19122 ms |
| sql_1 | structured_data | harness | yes | 1.00 | `sqlite_query` | 0 | 0 | 2 ms |
| sql_1 | structured_data | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 415 | 15878 ms |
| sql_2 | structured_data | harness | yes | 1.00 | `sqlite_query` | 0 | 0 | 2 ms |
| sql_2 | structured_data | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 332 | 11563 ms |
| sql_3 | structured_data | harness | yes | 1.00 | `sqlite_query` | 0 | 0 | 2 ms |
| sql_3 | structured_data | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 391 | 13935 ms |
| rank_1 | ranking | harness | yes | 1.00 | `corpus_statistics` | 0 | 0 | 12 ms |
| rank_1 | ranking | baseline | no | 0.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 512 | 19998 ms |
| rea_1 | reasoning | harness | yes | 0.50 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 259 | 6885 ms |
| rea_1 | reasoning | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 318 | 8974 ms |
| rea_2 | reasoning | harness | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 366 | 11203 ms |
| rea_2 | reasoning | baseline | yes | 1.00 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 422 | 13352 ms |
| gen_1 | generation | harness | yes | 0.88 | `linkedin_writer(rag+qwen+qc)` | 1 | 785 | 15668 ms |
| gen_1 | generation | baseline | no | 0.71 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 523 | 18040 ms |
| gen_2 | generation | harness | yes | 1.00 | `linkedin_writer(rag+qwen+qc)` | 1 | 804 | 14502 ms |
| gen_2 | generation | baseline | no | 0.71 | `qwen_llm(qwen1_5-1_8b-chat-q4_k_m.` | 1 | 518 | 18446 ms |

### Grading

- `exact`: the answer must contain the value computed independently (SQL/Python).
- `set`: every required token must appear.
- `soft`: keyword-coverage heuristic — reported as a score, not as proof of quality.
- `style`: the deterministic style validator scores the generated post against the measured profile; the same validator grades the baseline's output.
