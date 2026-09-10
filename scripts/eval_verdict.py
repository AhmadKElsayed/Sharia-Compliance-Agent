#!/usr/bin/env python
"""Measure verdict quality against the verdict eval set.

Unlike the retrieval eval this runs the *whole* graph, so it costs real LLM
calls and real time. It is the only thing that measures the claim the product
makes, so it is worth both.

Usage:
    python scripts/eval_verdict.py                     # whole set
    python scripts/eval_verdict.py --stratum adversarial
    python scripts/eval_verdict.py --repeat 3          # also measure stability
    python scripts/eval_verdict.py --workers 8         # more concurrency
    python scripts/eval_verdict.py --json out.json     # machine-readable result

The headline number is the **false-COMPLIANT rate**: the fraction of prohibited
or adversarial cases the system approved. It is reported separately from
accuracy rather than averaged into it, because approving a prohibition and
over-referring a permissible product are not errors of the same kind and must
never cancel each other out in a single figure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import build_graph, initial_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from evals.verdict_set import (  # noqa: E402
    CLEAR_COMPLIANT,
    COMPLIANT,
    NEEDS_REVIEW,
    NON_COMPLIANT,
    OUT_OF_SCOPE,
    SCHOLAR_REVIEWED,
    STRATA,
    VERDICT_SET,
    VerdictCase,
)


@dataclass
class Outcome:
    case: VerdictCase
    verdicts: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    cited: set[str] = field(default_factory=set)
    retrieved: bool = False
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """The modal verdict across runs; ties resolve to the first seen."""
        return Counter(self.verdicts).most_common(1)[0][0] if self.verdicts else "ERROR"

    @property
    def accepted(self) -> bool:
        return self.verdict in self.case.accept

    @property
    def is_false_compliant(self) -> bool:
        return self.verdict in self.case.forbidden

    @property
    def detected(self) -> bool:
        """Named the prohibition, rather than merely declining to approve it.

        The false-COMPLIANT rate counts NEEDS_REVIEW as a pass, which is right
        for safety and wrong for utility: a system that referred every
        prohibition to a human would score a perfect 0.000 while telling the
        reviewer nothing. This separates the two.
        """
        return bool(self.case.forbidden) and self.verdict == NON_COMPLIANT

    @property
    def strictly_permissible(self) -> bool:
        """COMPLIANT is the only defensible answer, so NEEDS_REVIEW is a miss.

        Borderline cases are excluded deliberately: NEEDS_REVIEW is an accepted
        answer there, and counting it as over-flagging would score correct
        conservatism as a defect.
        """
        return self.case.accept == {COMPLIANT}

    @property
    def over_flagged(self) -> bool:
        """A plainly permissible product referred to a human anyway."""
        return self.strictly_permissible and self.verdict == NEEDS_REVIEW

    @property
    def grounded(self) -> bool:
        """At least one surviving citation came from a governing standard."""
        return bool(self.case.must_cite) and bool(self.cited & self.case.must_cite)

    @property
    def stable(self) -> bool:
        return len(set(self.verdicts)) <= 1

    def scorecard(self) -> list[tuple[str, bool, str]]:
        """One point per criterion the agent satisfied.

        Two points rather than one, because a verdict can be right in two
        different ways and only rewarding the label hides the difference. A
        prohibition that comes back NEEDS_REVIEW is not approved -- it earns the
        verdict point -- but it did not tell the reviewer anything, so it does
        not earn the second. The mirror holds for a clear permission referred to
        a human.

        What counts as decisive is stratum-specific, because the *desired*
        behaviour differs: name a prohibition, approve a clear permission, and
        decline to resolve a borderline case.
        """
        rows = [("verdict", self.accepted, "returned an acceptable verdict")]

        if self.case.stratum == OUT_OF_SCOPE:
            rows.append(("no retrieval", not self.retrieved, "answered without retrieving"))
            return rows

        if self.case.forbidden:
            rows.append(("named it", self.verdict == NON_COMPLIANT, "named the prohibition"))
        elif self.case.stratum == CLEAR_COMPLIANT:
            rows.append(("decided", self.verdict == COMPLIANT, "approved a clear permission"))
        else:
            rows.append(("deferred", self.verdict == NEEDS_REVIEW, "left a borderline case open"))

        rows.append(("grounded", self.grounded, "cited a governing standard"))
        return rows

    @property
    def points(self) -> int:
        return sum(1 for _, earned, _ in self.scorecard() if earned)

    @property
    def possible(self) -> int:
        return len(self.scorecard())


def run_case(case: VerdictCase, graph, repeat: int) -> Outcome:
    outcome = Outcome(case=case)
    for attempt in range(repeat):
        trace_id = f"eval-{abs(hash(case.query)) % 10**8}-{attempt}"
        started = time.perf_counter()
        try:
            state = graph.invoke(initial_state(case.query, trace_id))
        except Exception as exc:  # noqa: BLE001
            outcome.errors.append(f"{type(exc).__name__}: {exc}")
            continue
        outcome.latencies.append(time.perf_counter() - started)

        decision = state.get("decision")
        outcome.verdicts.append(decision.verdict.value if decision else "ERROR")
        outcome.rules.append(decision.rule if decision else "none")
        for finding in state.get("findings", []) or []:
            for citation in finding.citations:
                outcome.cited.add(citation.doc_id)
        if "retrieve" in (state.get("node_path") or []):
            outcome.retrieved = True
        outcome.errors.extend(state.get("errors") or [])
    return outcome


def evaluate(cases: list[VerdictCase], repeat: int, workers: int) -> list[Outcome]:
    settings = get_settings()
    missing = settings.missing_required()
    if missing:
        raise SystemExit(f"configuration incomplete: {', '.join(missing)}")

    from app.main import build_dependencies

    # Built exactly as the API builds them, so the eval cannot drift from what
    # production actually runs.
    graph = build_graph(build_dependencies(settings))

    done = 0
    total = len(cases)

    def work(case: VerdictCase) -> Outcome:
        nonlocal done
        outcome = run_case(case, graph, repeat)
        done += 1
        print(
            f"  [{done:>2}/{total}] {outcome.verdict:<14} {case.query[:58]}",
            file=sys.stderr,
            flush=True,
        )
        return outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, cases))


def report(outcomes: list[Outcome], repeat: int, verbose: bool) -> dict:
    by_stratum: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        by_stratum[outcome.case.stratum].append(outcome)

    risky = [o for o in outcomes if o.case.forbidden]
    false_compliant = [o for o in risky if o.is_false_compliant]
    permissible = [o for o in outcomes if o.strictly_permissible]
    over_flagged = [o for o in permissible if o.over_flagged]
    citable = [o for o in outcomes if o.case.must_cite]
    grounded = [o for o in citable if o.grounded]
    oos = by_stratum.get(OUT_OF_SCOPE, [])
    leaked = [o for o in oos if o.retrieved]
    latencies = [x for o in outcomes for x in o.latencies]

    print()
    print("=" * 72)
    print("VERDICT EVAL")
    if not SCHOLAR_REVIEWED:
        print("  !! expectations are engineer-authored and NOT scholar-reviewed.")
        print("     Treat these numbers as a regression signal, not as evidence")
        print("     of Sharia correctness. See evals/verdict_set.py.")
    print("=" * 72)

    print(f"\ncases {len(outcomes)}   runs/case {repeat}")
    print(
        f"\n  FALSE COMPLIANT      {len(false_compliant)}/{len(risky)} "
        f"= {len(false_compliant) / max(1, len(risky)):.3f}   <-- the number that matters"
    )
    detected = [o for o in risky if o.detected]
    print(
        f"  prohibition detected {len(detected)}/{len(risky)} "
        f"= {len(detected) / max(1, len(risky)):.3f}   (named it, not just declined to approve)"
    )
    print(
        f"  accepted verdict     {sum(o.accepted for o in outcomes)}/{len(outcomes)} "
        f"= {sum(o.accepted for o in outcomes) / max(1, len(outcomes)):.3f}"
    )
    print(
        f"  over-flagged         {len(over_flagged)}/{len(permissible)} "
        f"= {len(over_flagged) / max(1, len(permissible)):.3f}   (clear permissions sent to review)"
    )
    print(
        f"  citation grounding   {len(grounded)}/{len(citable)} "
        f"= {len(grounded) / max(1, len(citable)):.3f}   (cited a governing standard)"
    )
    if oos:
        print(f"  out-of-scope leaked  {len(leaked)}/{len(oos)}   (reached retrieval)")
    if repeat > 1:
        stable = sum(o.stable for o in outcomes)
        print(
            f"  verdict stability    {stable}/{len(outcomes)} "
            f"= {stable / max(1, len(outcomes)):.3f}"
        )
    if latencies:
        ordered = sorted(latencies)
        print(
            f"\n  latency  p50 {statistics.median(ordered):.1f}s   "
            f"p95 {ordered[int(len(ordered) * 0.95) - 1]:.1f}s   "
            f"max {ordered[-1]:.1f}s"
        )

    earned = sum(o.points for o in outcomes)
    possible = sum(o.possible for o in outcomes)
    print(f"\n  SCORE                {earned}/{possible} = {earned / max(1, possible):.3f}")

    print("\n  score by stratum")
    header = f"    {'stratum':<18} {'n':>3} {'score':>10} {'pct':>7} {'false-COMPL':>13}"
    print(header)
    for stratum in STRATA:
        group = by_stratum.get(stratum, [])
        if not group:
            continue
        got = sum(o.points for o in group)
        top = sum(o.possible for o in group)
        bad = sum(o.is_false_compliant for o in group)
        risk = sum(1 for o in group if o.case.forbidden)
        risk_cell = f"{bad}/{risk}" if risk else "-"
        pct = got / max(1, top)
        print(f"    {stratum:<18} {len(group):>3} {got:>5}/{top:<4} {pct:>7.3f} {risk_cell:>13}")

    print("\n  points by criterion")
    criteria: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for outcome in outcomes:
        for name, won, _ in outcome.scorecard():
            criteria[name][1] += 1
            criteria[name][0] += int(won)
    for name, (won, total) in criteria.items():
        print(f"    {name:<14} {won:>3}/{total:<3} = {won / max(1, total):.3f}")

    print("\n  every case")
    for stratum in STRATA:
        group = by_stratum.get(stratum, [])
        if not group:
            continue
        print(f"\n    -- {stratum} --")
        for outcome in sorted(group, key=lambda o: -o.points):
            missed = [n for n, won, _ in outcome.scorecard() if not won]
            tail = f"   missed: {', '.join(missed)}" if missed else ""
            print(f"    {outcome.points}/{outcome.possible}  {outcome.verdict:<14} {outcome.case.query[:62]}{tail}")

    failures = [o for o in outcomes if not o.accepted]
    if failures:
        print(f"\n  {len(failures)} case(s) outside the accepted set:")
        for outcome in failures:
            marker = "FALSE-COMPLIANT" if outcome.is_false_compliant else "miss"
            print(f"\n    [{marker}] {outcome.case.stratum}")
            print(f"      query    {outcome.case.query[:100]}")
            print(f"      got      {outcome.verdict}  (rule: {outcome.rules[0] if outcome.rules else '-'})")
            print(f"      expected {'|'.join(sorted(outcome.case.accept))}")
            if outcome.case.authority:
                print(f"      governs  {outcome.case.authority} -- {outcome.case.rationale}")
            if outcome.errors:
                print(f"      errors   {outcome.errors[0][:120]}")

    if verbose:
        print("\n  all cases")
        for outcome in outcomes:
            flag = "ok " if outcome.accepted else "FAIL"
            print(f"    {flag} {outcome.verdict:<14} {outcome.case.query[:70]}")

    return {
        "score": earned,
        "score_possible": possible,
        "case_count": len(outcomes),
        "repeat": repeat,
        "scholar_reviewed": SCHOLAR_REVIEWED,
        "false_compliant": len(false_compliant),
        "prohibition_detected": len(detected),
        "false_compliant_of": len(risky),
        "accepted": sum(o.accepted for o in outcomes),
        "over_flagged": len(over_flagged),
        "citation_grounding": len(grounded) / max(1, len(citable)),
        "out_of_scope_leaked": len(leaked),
        "by_stratum": {
            s: {
                "n": len(g),
                "accepted": sum(o.accepted for o in g),
                "false_compliant": sum(o.is_false_compliant for o in g),
            }
            for s, g in by_stratum.items()
        },
        "cases": [
            {
                "query": o.case.query,
                "stratum": o.case.stratum,
                "authority": o.case.authority,
                "expected": sorted(o.case.accept),
                "verdicts": o.verdicts,
                "rules": o.rules,
                "cited": sorted(o.cited),
                "accepted": o.accepted,
                "detected": o.detected,
                "false_compliant": o.is_false_compliant,
            }
            for o in outcomes
        ],
        "failures": [
            {
                "query": o.case.query,
                "stratum": o.case.stratum,
                "got": o.verdict,
                "expected": sorted(o.case.accept),
                "rule": o.rules[0] if o.rules else None,
                "authority": o.case.authority,
            }
            for o in failures
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stratum", choices=STRATA, help="run one stratum only")
    parser.add_argument("--limit", type=int, help="run the first N cases")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--workers", type=int, default=4, help="concurrent assessments")
    parser.add_argument("--json", help="write the result to this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = VERDICT_SET
    if args.stratum:
        cases = [c for c in cases if c.stratum == args.stratum]
    if args.limit:
        cases = cases[: args.limit]

    runs = len(cases) * args.repeat
    print(
        f"running {len(cases)} case(s) x {args.repeat} = {runs} assessments "
        f"at {args.workers} workers",
        file=sys.stderr,
    )
    print(
        f"expect roughly {runs * 16 / args.workers / 60:.0f} min and "
        f"{runs * 2:,} LLM calls",
        file=sys.stderr,
    )

    started = time.perf_counter()
    outcomes = evaluate(cases, args.repeat, args.workers)
    elapsed = time.perf_counter() - started

    result = report(outcomes, args.repeat, args.verbose)
    result["wall_seconds"] = round(elapsed, 1)
    print(f"\n  wall clock {elapsed / 60:.1f} min")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    return 1 if result["false_compliant"] else 0


if __name__ == "__main__":
    sys.exit(main())
