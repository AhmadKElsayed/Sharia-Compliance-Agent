"""Deterministic verdict aggregation.

The LLM produces *findings*; this module produces the *verdict*. Keeping the
decision in plain Python means it is auditable, unit-testable, and identical for
identical findings — none of which is true of asking a model for a label.

The governing principle is asymmetric cost. In compliance review a false
COMPLIANT ships a prohibited product; a false NEEDS_REVIEW costs a reviewer an
hour. Every ambiguous path therefore resolves toward NEEDS_REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Severity(StrEnum):
    """How a single finding bears on the overall verdict."""

    PROHIBITED = "PROHIBITED"
    """A clear violation, e.g. a guaranteed return on a deposit."""

    CONDITIONAL = "CONDITIONAL"
    """Permissible only if conditions hold that the query does not establish."""

    UNRESOLVED = "UNRESOLVED"
    """The corpus does not settle the question."""

    PERMISSIBLE = "PERMISSIBLE"
    """Explicitly allowed by a cited clause."""


@dataclass(frozen=True)
class Citation:
    """A pointer into the corpus supporting a finding."""

    chunk_id: str
    doc_id: str
    section: str
    title: str
    quote: str = ""

    def label(self) -> str:
        return f"{self.doc_id} §{self.section}" if self.section else self.doc_id


@dataclass
class Finding:
    """One compliance issue identified in the proposal."""

    issue: str
    severity: Severity
    explanation: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True if at least one citation survived verification."""
        return bool(self.citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "section": c.section,
                    "title": c.title,
                    "quote": c.quote,
                }
                for c in self.citations
            ],
        }


@dataclass
class VerdictDecision:
    """The verdict, plus the rule that produced it."""

    verdict: Verdict
    rule: str
    rationale: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rule": self.rule,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 3),
        }


# Retrieval below this score means the corpus does not really cover the query.
# Calibrated against live measurements: in-scope queries score 0.53-0.74 and an
# out-of-scope query scores 0.16.
COVERAGE_THRESHOLD = 0.35


GREETING_REPLY = (
    "Hello. I assess whether a proposed financial product or transaction is "
    "Sharia-compliant, against the AAOIFI Shari'ah Standards. Describe a product "
    "and I will review it — for example: \"Can we offer a savings account paying "
    "a fixed 4% annual return?\""
)


def decide(
    findings: list[Finding],
    top_score: float,
    *,
    out_of_scope: bool = False,
    greeting: bool = False,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> VerdictDecision:
    """Aggregate findings into a verdict.

    Rules are evaluated in order; the first match wins, and the rule name is
    recorded so a reviewer can see exactly why the verdict came out as it did.
    """
    if greeting:
        # A greeting is not an assessment, and saying "the corpus does not cover
        # this" to someone who said hello is a non-answer. It keeps the
        # NEEDS_REVIEW label because the response schema requires one of the
        # three verdicts, and inventing a fourth would break every client; the
        # rule name and rationale are what carry the real meaning here.
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "greeting",
            GREETING_REPLY,
            0.0,
        )

    if out_of_scope:
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "out_of_scope",
            "The query does not describe a financial product or transaction that "
            "this corpus covers.",
            0.2,
        )

    if top_score < coverage_threshold:
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "insufficient_corpus_coverage",
            f"Best retrieval score {top_score:.2f} is below the {coverage_threshold} "
            "coverage threshold, so the corpus does not adequately address this query.",
            0.25,
        )

    if not findings:
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "no_findings",
            "No compliance issues were identified, but neither was any clause found "
            "that affirmatively permits the proposal.",
            0.3,
        )

    prohibited = [f for f in findings if f.severity is Severity.PROHIBITED]
    if prohibited:
        grounded = [f for f in prohibited if f.is_grounded]
        if not grounded:
            # Every supporting citation was stripped as hallucinated. Refusing on
            # unsupported grounds is as wrong as approving on them.
            return VerdictDecision(
                Verdict.NEEDS_REVIEW,
                "prohibited_findings_lost_all_citations",
                f"{len(prohibited)} prohibition(s) were raised but none retained a "
                "verifiable citation after checking them against retrieved text.",
                0.3,
            )
        issues = ", ".join(sorted({f.issue for f in grounded}))
        return VerdictDecision(
            Verdict.NON_COMPLIANT,
            "grounded_prohibition",
            f"Prohibited under: {issues}.",
            _confidence(0.75, grounded, top_score),
        )

    unresolved = [f for f in findings if f.severity is Severity.UNRESOLVED]
    if unresolved:
        issues = ", ".join(sorted({f.issue for f in unresolved}))
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "unresolved_findings",
            f"The corpus does not settle: {issues}.",
            0.4,
        )

    conditional = [f for f in findings if f.severity is Severity.CONDITIONAL]
    if conditional:
        issues = ", ".join(sorted({f.issue for f in conditional}))
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "conditional_findings",
            f"Permissible only if conditions are met that the proposal does not "
            f"establish: {issues}.",
            0.5,
        )

    permissible = [f for f in findings if f.severity is Severity.PERMISSIBLE]
    grounded = [f for f in permissible if f.is_grounded]
    if not grounded:
        return VerdictDecision(
            Verdict.NEEDS_REVIEW,
            "permissible_but_ungrounded",
            "No prohibition was identified, but no verifiable citation supports "
            "permissibility either.",
            0.35,
        )

    return VerdictDecision(
        Verdict.COMPLIANT,
        "all_findings_permissible",
        f"All {len(grounded)} identified issue(s) are supported by cited clauses "
        "permitting the structure.",
        _confidence(0.7, grounded, top_score),
    )


def _confidence(base: float, findings: list[Finding], top_score: float) -> float:
    """Blend citation density and retrieval strength into a 0-1 score.

    Deliberately coarse. It signals how much support the verdict has, and is not
    a probability — presenting it as one would overstate what this can know.
    """
    citation_count = sum(len(f.citations) for f in findings)
    citation_bonus = min(citation_count, 4) * 0.04
    retrieval_bonus = max(0.0, min(top_score, 1.0)) * 0.12
    return round(min(base + citation_bonus + retrieval_bonus, 0.97), 3)


def verify_citations(
    findings: list[Finding], retrieved_chunk_ids: set[str]
) -> tuple[list[Finding], list[str]]:
    """Drop citations that point at chunks the retriever never returned.

    A model asked for citations will sometimes invent plausible-looking ones. An
    unverifiable citation is worse than none: it looks like evidence. This is
    the gate that stops fabricated support reaching a reviewer.

    Returns the cleaned findings and the list of rejected chunk IDs.
    """
    rejected: list[str] = []
    cleaned: list[Finding] = []

    for finding in findings:
        kept = []
        for citation in finding.citations:
            if citation.chunk_id in retrieved_chunk_ids:
                kept.append(citation)
            else:
                rejected.append(citation.chunk_id)
        cleaned.append(
            Finding(
                issue=finding.issue,
                severity=finding.severity,
                explanation=finding.explanation,
                citations=kept,
            )
        )

    return cleaned, rejected
