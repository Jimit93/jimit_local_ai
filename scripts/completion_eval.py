"""Measure sentence completion within a token limit on the real code model.

A: raw prompt, raw output          (what a plain call returns)
B: raw prompt, deterministic trim  (repair only)
C: budgeted prompt, trim           (prevention + repair, what the app does)

Usage: python -m scripts.completion_eval [--max-tokens 60] [--tier small]
"""
from __future__ import annotations

import argparse
import json
import re
import time

from backend.config import REPORTS_DIR
from backend.harness.completion import budget_instruction, finish
from backend.llm import manager as LLM

PROMPTS = [
    "Explain in detail how a CPU governor with hysteresis protects a laptop from sustained high load.",
    "Describe every step a coding agent should take to fix a failing unit test safely.",
    "Explain why BM25 ranking works well for finding relevant source files, with examples.",
    "List and explain the risks of letting a language model edit files without checkpoints.",
]
COMPLETE = re.compile(r"""[.!?…](?:["')\]]+)?\s*$|```\s*$""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--tier", default="small")
    args = ap.parse_args()
    rows = []
    for p in PROMPTS:
        raw = LLM.generate_code(p, tier=args.tier, max_tokens=args.max_tokens, temperature=0.0, finish="none")
        trimmed, meta_b = finish(raw.text, raw.stop_reason, "prose")
        budgeted = LLM.generate_code(p + "\n\n" + budget_instruction(args.max_tokens), tier=args.tier,
                                     max_tokens=args.max_tokens, temperature=0.0, finish="prose")
        rows.append({
            "prompt": p[:60], "model": raw.model,
            "A_raw": {"stop": raw.stop_reason, "complete": bool(COMPLETE.search(raw.text)), "words": len(raw.text.split()),
                      "tail": raw.text[-70:]},
            "B_trim": {"complete": bool(COMPLETE.search(trimmed)) or trimmed.endswith("…"), "words": len(trimmed.split()),
                       "action": meta_b["action"], "tail": trimmed[-70:]},
            "C_budget_trim": {"stop": budgeted.stop_reason, "complete": bool(COMPLETE.search(budgeted.text)) or budgeted.text.endswith("…"),
                              "natural_finish": budgeted.stop_reason not in ("length",), "words": len(budgeted.text.split()),
                              "action": budgeted.meta["completion"]["action"], "tail": budgeted.text[-70:]},
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    n = len(rows)
    summary = {
        "max_tokens": args.max_tokens, "n": n,
        "A_raw_complete_rate": sum(r["A_raw"]["complete"] for r in rows) / n,
        "A_raw_truncated_rate": sum(r["A_raw"]["stop"] == "length" for r in rows) / n,
        "B_trim_complete_rate": sum(r["B_trim"]["complete"] for r in rows) / n,
        "C_complete_rate": sum(r["C_budget_trim"]["complete"] for r in rows) / n,
        "C_natural_finish_rate": sum(r["C_budget_trim"]["natural_finish"] for r in rows) / n,
        "C_mean_words": round(sum(r["C_budget_trim"]["words"] for r in rows) / n, 1),
    }
    out = {"ts": time.time(), "summary": summary, "rows": rows}
    (REPORTS_DIR / "agent").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "agent" / "completion_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
