"""Tests for deterministic verdict aggregation and citation verification.

These encode the safety property the whole design rests on: the system must
never return COMPLIANT on weak evidence, and must never return NON_COMPLIANT on
fabricated evidence.
"""

from __future__ import annotations

import pytest

from app.agent.verdict import (
    COVERAGE_THRESHOLD,
    Citation,
    Finding,
    Severity,
    Verdict,
    decide,
    verify_citations,
)

STRONG = 0.7  # a retrieval score comfortably above the coverage threshold


def cite(chunk_id: str = "SFS-001#3.1-3.3") -> Citation:
    return Citation(
        chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], section="3.1-3.3", title="T"
    )


def finding(severity: Severity, *, cited: bool = True, issue: str = "riba") -> Finding:
    return Finding(
        issue=issue,
        severity=severity,
        explanation="...",
        citations=[cite()] if cited else [],
    )


# --- core rule table -----------------------------------------------------


@pytest.mark.parametrize(
    ("severities", "expected", "rule"),
    [
        ([Severity.PROHIBITED], Verdict.NON_COMPLIANT, "grounded_prohibition"),
        (
            [Severity.PROHIBITED, Severity.PERMISSIBLE],
            Verdict.NON_COMPLIANT,
            "grounded_prohibition",
        ),
        ([Severity.UNRESOLVED], Verdict.NEEDS_REVIEW, "unresolved_findings"),
        ([Severity.CONDITIONAL], Verdict.NEEDS_REVIEW, "conditional_findings"),
        (
            [Severity.CONDITIONAL, Severity.PERMISSIBLE],
            Verdict.NEEDS_REVIEW,
            "conditional_findings",
        ),
        ([Severity.PERMISSIBLE], Verdict.COMPLIANT, "all_findings_permissible"),
    ],
)
def test_severity_combinations(
    severities: list[Severity], expected: Verdict, rule: str
) -> None:
    decision = decide([finding(s) for s in severities], STRONG)
    assert decision.verdict is expected
    assert decision.rule == rule


def test_prohibited_outranks_unresolved_and_conditional() -> None:
    """A clear violation is not softened by co-occurring uncertainty."""
    decision = decide(
        [
            finding(Severity.UNRESOLVED),
            finding(Severity.CONDITIONAL),
            finding(Severity.PROHIBITED),
        ],
        STRONG,
    )
    assert decision.verdict is Verdict.NON_COMPLIANT


def test_unresolved_outranks_conditional() -> None:
    decision = decide(
        [finding(Severity.CONDITIONAL), finding(Severity.UNRESOLVED)], STRONG
    )
    assert decision.rule == "unresolved_findings"


# --- fail-safe paths -----------------------------------------------------


def test_out_of_scope_short_circuits() -> None:
    decision = decide([finding(Severity.PROHIBITED)], STRONG, out_of_scope=True)
    assert decision.verdict is Verdict.NEEDS_REVIEW
    assert decision.rule == "out_of_scope"


def test_weak_retrieval_forces_review_even_with_findings() -> None:
    decision = decide([finding(Severity.PERMISSIBLE)], top_score=0.16)
    assert decision.verdict is Verdict.NEEDS_REVIEW
    assert decision.rule == "insufficient_corpus_coverage"


def test_threshold_boundary_is_inclusive_above() -> None:
    assert decide([finding(Severity.PERMISSIBLE)], COVERAGE_THRESHOLD).verdict is (
        Verdict.COMPLIANT
    )
    assert decide(
        [finding(Severity.PERMISSIBLE)], COVERAGE_THRESHOLD - 0.01
    ).verdict is Verdict.NEEDS_REVIEW


def test_no_findings_is_not_compliant() -> None:
    """Silence is not evidence of permissibility."""
    decision = decide([], STRONG)
    assert decision.verdict is Verdict.NEEDS_REVIEW
    assert decision.rule == "no_findings"


def test_prohibition_without_citations_downgrades() -> None:
    """Refusing on fabricated evidence is as wrong as approving on it."""
    decision = decide([finding(Severity.PROHIBITED, cited=False)], STRONG)
    assert decision.verdict is Verdict.NEEDS_REVIEW
    assert decision.rule == "prohibited_findings_lost_all_citations"


def test_permissible_without_citations_does_not_reach_compliant() -> None:
    decision = decide([finding(Severity.PERMISSIBLE, cited=False)], STRONG)
    assert decision.verdict is Verdict.NEEDS_REVIEW
    assert decision.rule == "permissible_but_ungrounded"


def test_partially_grounded_prohibition_still_non_compliant() -> None:
    decision = decide(
        [
            finding(Severity.PROHIBITED, cited=False, issue="gharar"),
            finding(Severity.PROHIBITED, cited=True, issue="riba"),
        ],
        STRONG,
    )
    assert decision.verdict is Verdict.NON_COMPLIANT
    assert "riba" in decision.rationale
    assert "gharar" not in decision.rationale


# --- confidence ----------------------------------------------------------


def test_confidence_is_bounded() -> None:
    many = Finding(
        issue="riba",
        severity=Severity.PROHIBITED,
        explanation="...",
        citations=[cite(f"SFS-001#{i}") for i in range(20)],
    )
    assert 0.0 <= decide([many], 1.0).confidence <= 0.97


def test_confidence_rises_with_retrieval_strength() -> None:
    weak = decide([finding(Severity.PROHIBITED)], 0.40).confidence
    strong = decide([finding(Severity.PROHIBITED)], 0.95).confidence
    assert strong > weak


def test_low_confidence_on_every_fail_safe_path() -> None:
    for decision in (
        decide([], STRONG),
        decide([finding(Severity.PROHIBITED)], STRONG, out_of_scope=True),
        decide([finding(Severity.PERMISSIBLE)], 0.1),
        decide([finding(Severity.PROHIBITED, cited=False)], STRONG),
    ):
        assert decision.confidence <= 0.5, decision.rule


# --- citation verification -----------------------------------------------


def test_hallucinated_citations_are_dropped() -> None:
    findings = [
        Finding(
            issue="riba",
            severity=Severity.PROHIBITED,
            explanation="...",
            citations=[cite("real#1"), cite("invented#9")],
        )
    ]
    cleaned, rejected = verify_citations(findings, {"real#1"})
    assert [c.chunk_id for c in cleaned[0].citations] == ["real#1"]
    assert rejected == ["invented#9"]


def test_verification_preserves_finding_content() -> None:
    findings = [finding(Severity.CONDITIONAL, issue="gharar")]
    cleaned, _ = verify_citations(findings, {cite().chunk_id})
    assert cleaned[0].issue == "gharar"
    assert cleaned[0].severity is Severity.CONDITIONAL


def test_verification_does_not_mutate_input() -> None:
    original = [finding(Severity.PROHIBITED)]
    verify_citations(original, set())
    assert len(original[0].citations) == 1, "input findings must not be mutated"


def test_end_to_end_fabricated_prohibition_cannot_convict() -> None:
    """The property that matters most: invented evidence never yields a refusal."""
    findings = [
        Finding(
            issue="riba",
            severity=Severity.PROHIBITED,
            explanation="cites a clause that does not exist",
            citations=[cite("SFS-999#1.1")],
        )
    ]
    cleaned, rejected = verify_citations(findings, {"SFS-001#3.1-3.3"})
    decision = decide(cleaned, STRONG)
    assert rejected == ["SFS-999#1.1"]
    assert decision.verdict is Verdict.NEEDS_REVIEW
