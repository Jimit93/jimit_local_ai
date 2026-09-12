"""Benchmark task definitions.

Ground truth is computed deterministically at run time (from SQLite or from Python),
never hand-written, so the suite stays correct as the corpus changes.

Each task declares how it is graded:
  exact  - the answer must contain the exact expected value
  set    - the answer must contain all required tokens
  soft   - heuristic coverage score (reported separately, never as "correctness")
  style  - graded by the deterministic style validator
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchTask:
    id: str
    suite: str
    prompt: str
    grade: str                                  # exact|set|soft|style
    expected: Callable[[], object] | None = None
    must_contain: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    note: str = ""


def _norm_nums(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def grade(task: BenchTask, answer: str) -> tuple[bool, float, str]:
    """Return (correct, score 0..1, explanation)."""
    a = (answer or "").strip()
    if not a:
        return False, 0.0, "empty answer"

    if task.grade == "exact":
        want = task.expected() if callable(task.expected) else task.expected
        if isinstance(want, (int, float)):
            nums = _norm_nums(a)
            ok = any(abs(n - float(want)) < 1e-6 for n in nums)
            return ok, 1.0 if ok else 0.0, f"expected {want}, found {nums[:6]}"
        want_s = str(want).lower()
        ok = want_s in a.lower()
        return ok, 1.0 if ok else 0.0, f"expected substring '{want_s}'"

    if task.grade == "set":
        need = [t.lower() for t in task.must_contain]
        hits = [t for t in need if t in a.lower()]
        ok = len(hits) == len(need)
        return ok, len(hits) / max(1, len(need)), f"{len(hits)}/{len(need)} required tokens"

    if task.grade == "soft":
        need = [t.lower() for t in task.must_contain]
        hits = [t for t in need if t in a.lower()]
        score = len(hits) / max(1, len(need))
        return score >= 0.5, score, f"coverage {len(hits)}/{len(need)}"

    return True, 1.0, "graded by style validator"


# ------------------------------------------------------------ ground truth

_NLP_SAMPLE = ("Local models keep data on the machine and cost nothing per token "
               "to run every single day")


def _word_count(text: str) -> int:
    from backend.tools.deterministic import word_count

    return word_count(text)


def _count(doc_type: str | None = None) -> int:
    from backend.tools.deterministic import sql_query

    if doc_type:
        return sql_query("SELECT COUNT(*) n FROM linkedin_documents WHERE doc_type=?",
                         (doc_type,))[0]["n"]
    return sql_query("SELECT COUNT(*) n FROM linkedin_documents")[0]["n"]


def _keyword_hits(term: str) -> int:
    from backend.retrieval.index import index

    return len(index().keyword_search(term, top_k=1000))


def _top_hashtag() -> str:
    from backend.services.style_engine import load_profile

    p = load_profile() or {}
    top = (p.get("hashtags", {}).get("top") or [("", 0)])[0][0]
    return top.lstrip("#")


def build_tasks() -> list[BenchTask]:
    T = BenchTask
    return [
        # A. deterministic -------------------------------------------------
        T("det_1", "deterministic", "What is 25 * 40?", "exact", lambda: 1000),
        T("det_2", "deterministic", "What is 15% of 2400?", "exact", lambda: 360),
        T("det_3", "deterministic", "Calculate (1875 + 425) / 4", "exact", lambda: 575),
        T("det_4", "deterministic", "What is 2^12?", "exact", lambda: 4096),
        T("det_5", "deterministic", "How many days between 2026-01-01 and 2026-09-12?",
          "exact", lambda: 254),

        # B. NLP / text processing ----------------------------------------
        T("nlp_1", "nlp", f'How many words are in this text: "{_NLP_SAMPLE}"',
          "exact", lambda: _word_count(_NLP_SAMPLE)),
        T("nlp_2", "nlp",
          'Extract the hashtags from this post: "Shipped the local agent today #AI '
          '#LocalLLM #Engineering"',
          "set", must_contain=["#ai", "#localllm", "#engineering"]),
        T("nlp_3", "nlp",
          'Extract the URLs from: "Full write-up here https://example.com/post and the '
          'repo at https://github.com/x/y"',
          "set", must_contain=["https://example.com/post", "https://github.com/x/y"]),

        # C. classification ------------------------------------------------
        T("cls_1", "classification",
          'Classify this post: "The 4 hour BTCUSD chart shows a break of structure below '
          'the supply zone, so I am waiting for the retest before entry."',
          "exact", lambda: "trading"),
        T("cls_2", "classification",
          'Classify this post: "I built a RAG pipeline with local embeddings and a '
          'quantized model so inference runs offline on CPU."',
          "exact", lambda: "ai_engineering"),
        T("cls_3", "classification",
          'Classify this post: "Completed my capstone project and received the '
          'professional certification from the program."',
          "exact", lambda: "learning"),

        # D. retrieval (exact) ---------------------------------------------
        T("ret_1", "retrieval", "Find all LinkedIn posts mentioning RAG",
          "exact", lambda: _keyword_hits("RAG")),
        T("ret_2", "retrieval", "Show posts containing the word Qwen",
          "exact", lambda: _keyword_hits("Qwen")),

        # E. semantic retrieval --------------------------------------------
        T("sem_1", "semantic_search", "Find posts semantically related to local AI",
          "soft", must_contain=["ai", "local"]),
        T("sem_2", "semantic_search", "What have I written about risk management",
          "soft", must_contain=["risk"]),

        # F. structured data -----------------------------------------------
        T("sql_1", "structured_data", "How many posts do I have?", "exact",
          lambda: _count("post")),
        T("sql_2", "structured_data", "How many comments were ingested?", "exact",
          lambda: _count("comment")),
        T("sql_3", "structured_data", "How many documents are in the knowledge base?",
          "exact", lambda: _count()),

        # G. ranking / corpus statistics -----------------------------------
        T("rank_1", "ranking", "What are the top 3 hashtags I use?", "exact",
          lambda: _top_hashtag()),

        # H. reasoning ------------------------------------------------------
        T("rea_1", "reasoning",
          "Explain in two sentences why running a quantized model on CPU is slower than "
          "on a GPU.", "soft",
          must_contain=["memory", "parallel"]),
        T("rea_2", "reasoning",
          "Should I use a 1.8B or a 7B local model when only 3 GB of RAM is free? "
          "Answer in two sentences.", "soft",
          must_contain=["1.8", "ram"]),

        # I. generation (style-graded) -------------------------------------
        T("gen_1", "generation",
          "Write a LinkedIn post about building a local AI agent that routes tasks to "
          "deterministic tools before calling the model.", "style",
          options={"writer_options": {"length": "short"}}),
        T("gen_2", "generation",
          "Write a LinkedIn post about measuring latency and token cost of a local "
          "inference stack.", "style",
          options={"writer_options": {"length": "short"}}),
    ]


SUITES = ["deterministic", "nlp", "classification", "retrieval", "semantic_search",
          "structured_data", "ranking", "reasoning", "generation"]
