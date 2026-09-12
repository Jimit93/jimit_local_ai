"""Run the full benchmark suite (baseline vs harness) and write the reports."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-baseline", action="store_true",
                    help="run only the harness (much faster)")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", choices=["auto", "small", "large"], default=None,
                    help="pin the model so runs are comparable")
    args = ap.parse_args()

    if args.model:
        os.environ["JLA_MODEL_PREFER"] = args.model

    from backend.benchmarks.runner import run_benchmark
    from backend.config import REPORTS_DIR

    rep = run_benchmark(suites=args.suites, include_baseline=not args.no_baseline,
                        limit=args.limit)
    t = rep["totals"]
    c = rep["comparison"]
    print(json.dumps({"run_id": rep["run_id"], "duration_s": rep["duration_s"],
                      "model": rep["environment"]["model"],
                      "harness": t["harness"], "baseline": t["baseline"],
                      "comparison": c}, indent=2))
    print(f"\nreports: {REPORTS_DIR / 'benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
