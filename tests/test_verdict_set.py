"""Structural checks on the verdict eval set.

No network and no LLM: these assert the set is well-formed, so a malformed case
fails in CI in milliseconds rather than 20 minutes into a paid run.
"""

from __future__ import annotations

import re

import pytest

from app.agent.verdict import Verdict
from evals.verdict_set import (
    ADVERSARIAL,
    CLEAR_COMPLIANT,
    CLEAR_PROHIBITED,
    COMPLIANT,
    OUT_OF_SCOPE,
    STRATA,
    VERDICT_SET,
    by_stratum,
)

AUTHORITY_RE = re.compile(r"^SS-\d{2}( \d+(/\d+)*)?$")


def test_every_stratum_is_populated() -> None:
    for stratum in STRATA:
        assert by_stratum(stratum), f"{stratum} has no cases"


def test_queries_are_unique() -> None:
    queries = [c.query for c in VERDICT_SET]
    assert len(set(queries)) == len(queries)


def test_accepted_verdicts_are_real_verdicts() -> None:
    valid = {v.value for v in Verdict}
    for case in VERDICT_SET:
        assert case.accept, f"{case.query!r} accepts nothing"
        assert case.accept <= valid, f"{case.query!r} accepts {case.accept - valid}"


def test_prohibited_and_adversarial_cases_forbid_compliant() -> None:
    """The set's whole point: approving these is the error that ships a product."""
    for stratum in (CLEAR_PROHIBITED, ADVERSARIAL):
        for case in by_stratum(stratum):
            assert case.forbidden == {COMPLIANT}
            assert COMPLIANT not in case.accept


def test_only_risky_strata_forbid_anything() -> None:
    for stratum in (CLEAR_COMPLIANT, OUT_OF_SCOPE):
        for case in by_stratum(stratum):
            assert case.forbidden == frozenset()


def test_in_scope_cases_name_a_governing_authority() -> None:
    for case in VERDICT_SET:
        if case.stratum == OUT_OF_SCOPE:
            continue
        assert case.authority, f"{case.query!r} names no authority"
        assert AUTHORITY_RE.match(case.authority), f"malformed: {case.authority!r}"
        assert case.rationale, f"{case.query!r} gives no rationale"
        assert case.must_cite, f"{case.query!r} names no expected standard"


def test_authority_standard_is_among_the_expected_standards() -> None:
    """A case whose authority sits outside must_cite would score itself wrong."""
    for case in VERDICT_SET:
        if not case.authority:
            continue
        assert case.authority.split()[0] in case.must_cite, case.query


def test_out_of_scope_cases_expect_irrelevant_only() -> None:
    for case in by_stratum(OUT_OF_SCOPE):
        assert case.accept == {Verdict.IRRELEVANT.value}
        assert not case.must_cite


def test_must_cite_uses_well_formed_standard_ids() -> None:
    for case in VERDICT_SET:
        for doc_id in case.must_cite:
            assert re.match(r"^SS-\d{2}$", doc_id), doc_id


@pytest.mark.parametrize("stratum", [CLEAR_PROHIBITED, CLEAR_COMPLIANT])
def test_clear_strata_accept_exactly_one_verdict(stratum: str) -> None:
    """A 'clear' case that accepts several verdicts is not measuring anything.

    One exception is allowed and asserted explicitly: the same-day sale and
    leaseback, where NEEDS_REVIEW is a defensible reading because the standard
    turns on a period long enough for the asset's value to change, and 'same
    day' is a fact the reviewer would want confirmed rather than assumed.
    """
    multi = [c for c in by_stratum(stratum) if len(c.accept) > 1]
    assert len(multi) <= 1, [c.query for c in multi]
    for case in multi:
        assert "leaseback" in case.query or "lease it straight back" in case.query
