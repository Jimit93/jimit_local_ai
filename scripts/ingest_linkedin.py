"""Full ingestion pipeline: parse -> style profile -> topics -> embeddings index."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="clear documents before ingesting")
    ap.add_argument("--path", nargs="*", default=None,
                    help="explicit export folder(s) or zip(s); defaults to the configured ones")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    from backend.ingestion.linkedin.ingest import ingest
    from backend.ml.topics import train_from_corpus
    from backend.retrieval.index import index
    from backend.services.style_engine import build_profile, summarize_profile

    t0 = time.perf_counter()
    rep = ingest(paths=args.path, reset=args.reset)
    print(f"[1/4] parsed {rep.files_used}/{rep.files_seen} files, {rep.rows_read} rows -> "
          f"{rep.inserted} new documents ({rep.duplicates} duplicates, "
          f"{rep.skipped_empty} empty) in {rep.duration_s}s")
    print(f"      by type: {rep.by_type}")
    if not rep.by_type.get("post"):
        print("      WARNING: no posts found. A 'Basic' LinkedIn export contains no "
              "Shares/Comments files - request the full archive for writing data.")

    prof = build_profile()
    print(f"[2/4] style profile built from {prof.get('n_documents', 0)} posts")
    print(f"      {summarize_profile(prof)}")

    meta = train_from_corpus()
    print(f"[3/4] topics: rule coverage {meta.get('rule_coverage')}, "
          f"CV accuracy {meta.get('cv_accuracy_mean')} ({meta.get('cv_folds')} folds), "
          f"distribution {meta.get('topic_distribution')}")

    if args.skip_embeddings:
        print("[4/4] embeddings skipped")
    else:
        stats = index(rebuild=True, rechunk=True).build(use_embeddings=True, rechunk=True)
        print(f"[4/4] semantic chunking + index: {json.dumps(stats)}")

    print(f"done in {round(time.perf_counter() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
