#!/usr/bin/env python
"""Measure retrieval quality against the golden set.

Runs the real retrieval path -- sub-query planning, embedding, vector search,
fusion, and optionally reranking -- and reports recall, MRR, and precision.
No LLM assessment: this isolates retrieval so a regression can be attributed
without the model's variance confounding it.

Usage:
    python scripts/eval_retrieval.py                 # current configuration
    python scripts/eval_retrieval.py --no-rerank     # force reranking off
    python scripts/eval_retrieval.py --top-k 10      # override per-sub-query k
    python scripts/eval_retrieval.py --verbose       # show every case
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes import AgentDeps, plan_retrieval, retrieve  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.ingest import build_embedder, build_store  # noqa: E402
from evals.golden_set import (  # noqa: E402
    ADVERSARIAL,
    COVERAGE,
    GOLDEN_SET,
    NEAR_MISS,
    OUT_OF_SCOPE,
)


def evaluate(deps: AgentDeps, verbose: bool = False) -> dict[str, float]:
    hits_at_1 = hits_at_3 = hits_any = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    failures: list[str] = []
    by_kind: dict[str, list[int]] = {}

    for gold in GOLDEN_SET:
        state = {
            "query": gold.query,
            "retrieval_round": 0,
            "node_path": [],
            "errors": [],
            "features": [],
            "concepts": [],
        }
        started = time.perf_counter()
        plan_retrieval(state, deps)
        retrieve(state, deps)
        latencies.append(time.perf_counter() - started)

        docs = [h.doc_id for h in state.get("hits", [])]
        relevant = [d in gold.expected for d in docs]

        found = any(relevant)
        if found:
            hits_any += 1
            rank = relevant.index(True) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
        else:
            reciprocal_ranks.append(0.0)
            failures.append(
                f"[{gold.kind}] {gold.query[:58]}  want {sorted(gold.expected)} got {docs[:4]}"
            )

        tally = by_kind.setdefault(gold.kind, [0, 0])
        tally[0] += int(found)
        tally[1] += 1

        precisions.append(sum(relevant) / len(relevant) if relevant else 0.0)

        if verbose:
            mark = "OK  " if found else "MISS"
            print(f"  {mark} [{gold.kind}] {gold.query[:58]}")
            print(f"       want {sorted(gold.expected)}  got {docs}")

    # Out-of-scope queries should score low enough to trip the coverage rule.
    oos_scores = []
    for query in OUT_OF_SCOPE:
        state = {
            "query": query, "retrieval_round": 0, "node_path": [], "errors": [],
            "features": [], "concepts": [],
        }
        plan_retrieval(state, deps)
        retrieve(state, deps)
        oos_scores.append(state.get("top_score", 0.0))

    n = len(GOLDEN_SET)
    results = {
        "recall_any": hits_any / n,
        "recall_at_1": hits_at_1 / n,
        "recall_at_3": hits_at_3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "precision": sum(precisions) / n,
        "latency_s": sum(latencies) / n,
        "oos_max_score": max(oos_scores) if oos_scores else 0.0,
    }
    for kind, (found, total) in by_kind.items():
        results[f"recall_{kind}"] = found / total if total else 0.0

    if failures:
        print("\n  misses:")
        for f in failures:
            print(f"    - {f}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if settings.missing_required():
        print(f"ERROR: missing {settings.missing_required()}", file=sys.stderr)
        return 2

    rerank = settings.rerank_enabled and not args.no_rerank
    top_k = args.top_k or settings.retrieval_top_k

    deps = AgentDeps(
        llm=None,
        embedder=build_embedder(settings),
        store=build_store(settings),
        top_k=top_k,
        score_threshold=settings.retrieval_score_threshold,
        reranker=_build_reranker(settings) if rerank else None,
    )

    print(f"cases      : {len(GOLDEN_SET)} golden + {len(OUT_OF_SCOPE)} out-of-scope")
    print(f"top_k      : {top_k} per sub-query")
    print(f"rerank     : {settings.rerank_model if rerank else 'OFF'}")
    print()

    results = evaluate(deps, verbose=args.verbose)

    print()
    print(f"  recall (any rank) : {results['recall_any']:.3f}")
    print(f"  recall@1          : {results['recall_at_1']:.3f}")
    print(f"  recall@3          : {results['recall_at_3']:.3f}")
    print(f"  MRR               : {results['mrr']:.3f}")
    print(f"  precision         : {results['precision']:.3f}")
    print(f"  latency/query     : {results['latency_s']:.2f}s")
    print(f"  out-of-scope max  : {results['oos_max_score']:.3f} "
          f"(threshold {settings.retrieval_score_threshold})")

    print("\n  recall by kind:")
    for kind in (COVERAGE, NEAR_MISS, ADVERSARIAL):
        key = f"recall_{kind}"
        if key in results:
            n = sum(1 for c in GOLDEN_SET if c.kind == kind)
            print(f"    {kind:12} : {results[key]:.3f}  ({n} cases)")
    return 0


def _build_reranker(settings):  # noqa: ANN001, ANN201
    from app.rag.rerank import OpenRouterReranker

    return OpenRouterReranker(
        api_key=settings.openrouter_api_key,
        model=settings.rerank_model,
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
