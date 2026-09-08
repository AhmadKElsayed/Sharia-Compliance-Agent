"""Golden retrieval set: queries paired with the standards that govern them.

This exists because retrieval quality is the binding constraint on verdict
quality, and until now it was evidenced only by spot checks. A production query
-- whether a savings account may guarantee a fixed return -- silently retrieved
the wrong clauses and was caught only because it happened to be tested by hand.

Each case names the AAOIFI standard(s) a competent reviewer would expect to be
consulted. Deliberately *standard*-level rather than clause-level: several
clauses within a standard can legitimately answer a question, so demanding an
exact clause would measure luck rather than retrieval.

Scope and honesty about limits: these pairs were written by an engineer reading
the corpus, not by a Sharia scholar. They are adequate for detecting retrieval
regressions and comparing retrieval strategies. They are **not** sufficient to
certify verdict correctness, which needs scholar-authored cases -- see
DOCUMENTATION.md §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    query: str
    #: Standards any of which counts as a correct retrieval.
    expected: frozenset[str]
    note: str = ""


def case(query: str, *expected: str, note: str = "") -> GoldenCase:
    return GoldenCase(query=query, expected=frozenset(expected), note=note)


GOLDEN_SET: list[GoldenCase] = [
    # --- deposits and returns -------------------------------------------
    case(
        "Can Mal offer a savings account guaranteeing depositors a fixed 4% annual return?",
        "SS-40", "SS-13", "SS-19",
        note="The production failure that motivated this set.",
    ),
    case(
        "May the bank guarantee the capital of investment account holders?",
        "SS-40", "SS-13", "SS-45",
    ),
    case(
        "How should profit be shared between the bank and depositors in a Mudarabah account?",
        "SS-40", "SS-13",
    ),
    case(
        "Can we smooth returns to investors using a profit equalisation reserve?",
        "SS-40", "SS-13",
    ),
    # --- murabaha --------------------------------------------------------
    case(
        "Is it permissible to sell an asset in Murabaha before the bank owns it?",
        "SS-08",
    ),
    case(
        "Can we charge a late payment penalty on a Murabaha and keep it as bank income?",
        "SS-08", "SS-03",
    ),
    case(
        "Must the bank disclose its actual cost and profit margin in a Murabaha?",
        "SS-08",
    ),
    case(
        "Can the customer act as the bank's agent to buy the asset in a Murabaha?",
        "SS-08", "SS-23",
    ),
    # --- leasing ---------------------------------------------------------
    case(
        "Can the lessee be made responsible for all maintenance and insurance of the leased asset?",
        "SS-09",
    ),
    case(
        "Does rental continue to accrue after the leased asset is destroyed?",
        "SS-09",
    ),
    # --- insurance and risk ---------------------------------------------
    case(
        "Is conventional insurance permissible, or must the bank use takaful?",
        "SS-26", "SS-41",
        note="A known miss under single-stage retrieval.",
    ),
    case(
        "Can the takaful operator retain the underwriting surplus as its own profit?",
        "SS-26", "SS-41",
    ),
    # --- liquidity and structuring --------------------------------------
    case(
        "Can we use tawarruq to provide a customer with cash?",
        "SS-30", "SS-44",
    ),
    case(
        "Is it permissible to sell a commodity we bought and resold instantly with no delivery?",
        "SS-30", "SS-20", "SS-08",
    ),
    # --- investment and screening ---------------------------------------
    case(
        "What debt ratio disqualifies a company from a Sharia-compliant equity fund?",
        "SS-21",
    ),
    case(
        "How is income from prohibited sources purified in an investment portfolio?",
        "SS-21", "SS-35",
    ),
    case(
        "What are the requirements for issuing investment sukuk?",
        "SS-17",
    ),
    # --- other contracts -------------------------------------------------
    case(
        "Can we charge a fee for providing a financial guarantee to a customer?",
        "SS-05",
    ),
    case(
        "Is it permissible to trade currencies with deferred settlement?",
        "SS-01",
    ),
    case(
        "Can the bank charge interest-based fees on a credit card?",
        "SS-02",
    ),
    case(
        "What conditions apply to a Salam contract where payment is made in advance?",
        "SS-10",
    ),
    case(
        "Can earnest money (Arboun) be retained if the buyer withdraws?",
        "SS-53",
    ),
]


OUT_OF_SCOPE: list[str] = [
    "What is the capital of France?",
    "How do I reset my online banking password?",
    "What is the weather forecast for Dubai tomorrow?",
]
