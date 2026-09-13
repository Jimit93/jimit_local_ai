"""Run the coding-agent system backtest.

    python -m scripts.agent_backtest [--suites deterministic,coding,safety,resilience] [--cases id1,id2]
                                     [--label system_backtest] [--db data/backtest/agent_backtest.db]

Uses its own database so fixture tasks never mix with the user's task history.
Writes reports/agent/<label>.json and .md.
"""
from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--suites", default="")
ap.add_argument("--cases", default="")
ap.add_argument("--label", default="system_backtest")
ap.add_argument("--db", default=os.path.join("data", "backtest", "agent_backtest.db"))
args = ap.parse_args()
os.makedirs(os.path.dirname(args.db), exist_ok=True)
os.environ["JLA_DB_PATH"] = os.path.abspath(args.db)  # before any backend import

import json  # noqa: E402

from backend.benchmarks.agent_backtest import run_backtest  # noqa: E402
from backend.config import REPORTS_DIR  # noqa: E402


def to_markdown(rep: dict) -> str:
    t, pv, res = rep["totals"], rep["predicted_vs_actual"], rep["resources"]
    lines = [f"# {rep['label']}", "",
             f"Run at {rep['ts']:.0f} · {rep['duration_s']} s · power plugged: {rep['power_plugged']} · "
             f"small model {rep['models']['small']['name'] if rep['models']['small'] else '-'}", "",
             f"**{t['passed']}/{t['cases']} cases passed ({t['pass_rate']:.0%})** · {t['llm_calls']} LLM calls · {t['tokens']} tokens", "",
             "| Suite | Passed | Pass rate | LLM calls | Tokens | Mean s |", "|---|---|---|---|---|---|"]
    for name, s in rep["suites"].items():
        lines.append(f"| {name} | {s['passed']}/{s['total']} | {s['pass_rate']:.0%} | {s['llm_calls']} | {s['tokens']} | {s['mean_duration_s']} |")
    lines += ["", "## Predicted vs actual", "",
              f"- status accuracy {pv['status_accuracy']}", f"- LLM budget adherence {pv['llm_budget_adherence']}",
              f"- method accuracy (LLM used iff predicted) {pv['method_accuracy']}", f"- files changed F1 (mean) {pv['files_f1_mean']}",
              "", "## Resources", "",
              f"- CPU avg {res['cpu_avg']}% · peak {res['cpu_peak']}% · samples above 70%: {res['cpu_above_70_fraction']}",
              f"- minimum free RAM {res['ram_min_gb']} GB", "", "## Cases", "",
              "| Case | Suite | Result | s | LLM | Detail |", "|---|---|---|---|---|---|"]
    for c in rep["cases"]:
        detail = c["detail"].replace("|", "/")[:140]
        lines.append(f"| {c['id']} | {c['suite']} | {'PASS' if c['passed'] else 'FAIL'} | {c['actual']['duration_s']} | "
                     f"{c['actual']['llm_calls']} | {detail} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rep = run_backtest(suites=[s for s in args.suites.split(",") if s] or None,
                       case_ids=[c for c in args.cases.split(",") if c] or None, label=args.label)
    out = REPORTS_DIR / "agent"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / f"{args.label}.md").write_text(to_markdown(rep), encoding="utf-8")
    print(json.dumps({"totals": rep["totals"], "suites": rep["suites"], "predicted_vs_actual": rep["predicted_vs_actual"],
                      "resources": {k: v for k, v in rep["resources"].items() if k not in ("series", "transitions", "protection_events")}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
