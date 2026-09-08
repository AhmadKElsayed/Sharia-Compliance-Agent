#!/usr/bin/env python
"""Build the Qdrant collection from the corpus.

Usage:
    python scripts/ingest.py                # create if absent, upsert in place
    python scripts/ingest.py --recreate     # drop and rebuild the collection
    python scripts/ingest.py --dry-run      # chunk only; no network calls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.chunking import chunk_corpus  # noqa: E402
from app.rag.ingest import CORPUS_DIR, build_embedder, build_store, ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection (required after changing the embedding model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk the corpus and report, without embedding or writing",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_DIR, help="corpus directory"
    )
    args = parser.parse_args()

    if args.dry_run:
        chunks = chunk_corpus(args.corpus)
        docs = sorted({c.doc_id for c in chunks})
        words = [len(c.text.split()) for c in chunks]
        print(f"corpus     : {args.corpus}")
        print(f"documents  : {len(docs)} ({', '.join(docs)})")
        print(f"chunks     : {len(chunks)}")
        print(f"words/chunk: min {min(words)} median {sorted(words)[len(words)//2]} max {max(words)}")
        print("\nfirst five citations:")
        for chunk in chunks[:5]:
            print(f"  {chunk.citation:24s} {chunk.text[:70]}...")
        return 0

    settings = get_settings()
    missing = settings.missing_required()
    if missing:
        print(f"ERROR: missing required settings: {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.example to .env and fill it in.", file=sys.stderr)
        return 2

    store = build_store(settings)
    if not store.ping():
        print(f"ERROR: cannot reach Qdrant at {settings.qdrant_url}", file=sys.stderr)
        return 3

    report = ingest(store, build_embedder(settings), args.corpus, recreate=args.recreate)
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
