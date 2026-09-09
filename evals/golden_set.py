"""Golden retrieval set: queries paired with the standards that govern them.

This exists because retrieval quality is the binding constraint on verdict
quality, and until now it was evidenced only by spot checks. A production query
-- whether a savings account may guarantee a fixed return -- silently retrieved
the wrong clauses and was caught only because it happened to be tested by hand.

Each case names the AAOIFI standard(s) a competent reviewer would expect to be
consulted. Deliberately *standard*-level rather than clause-level: several
clauses within a standard can legitimately answer a question, so demanding an
exact clause would measure luck rather than retrieval.

Cases carry a ``kind`` so failures can be attributed:

``coverage``     an ordinary question a compliance officer would ask. A miss
                 here means the corpus or the embedding is not reaching a topic.
``near_miss``    two surface-similar queries that must retrieve *different*
                 standards. A miss here means retrieval is keying on vocabulary
                 rather than on the transaction being described.
``adversarial``  a prohibited structure described in euphemistic or neutral
                 language, as a proposer hoping for approval would phrase it. A
                 miss here is the most dangerous kind: the governing prohibition
                 never reaches the model, so nothing downstream can catch it.

Scope and honesty about limits: these pairs were written by an engineer reading
the corpus, not by a Sharia scholar. They are adequate for detecting retrieval
regressions and comparing retrieval strategies. They are **not** sufficient to
certify verdict correctness, which needs scholar-authored cases -- see
DOCUMENTATION.md §3.2.

A note on the target. This set is not meant to sit at recall 1.000. A suite that
always passes has stopped measuring anything; the adversarial and near-miss
cases are deliberately at the edge of what the current retrieval can do, and
some are expected to fail. Track the trend and the per-kind breakdown, not a
green tick.
"""

from __future__ import annotations

from dataclasses import dataclass

COVERAGE = "coverage"
NEAR_MISS = "near_miss"
ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class GoldenCase:
    query: str
    #: Standards any of which counts as a correct retrieval.
    expected: frozenset[str]
    note: str = ""
    kind: str = COVERAGE


def case(query: str, *expected: str, note: str = "", kind: str = COVERAGE) -> GoldenCase:
    return GoldenCase(query=query, expected=frozenset(expected), note=note, kind=kind)


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

    # === coverage: partnership, agency and construction ==================
    case(
        "Can one partner in a Musharakah guarantee the capital contributed by another partner?",
        "SS-12",
    ),
    case(
        "How does a diminishing Musharakah home purchase transfer ownership to the customer over time?",
        "SS-12", "SS-09",
    ),
    case(
        "Can the bank finance construction of a factory under Istisna'a and pay the manufacturer in instalments?",
        "SS-11",
    ),
    case(
        "In an investment agency (Wakalah bi al-Istithmar), can the agent be promised a fixed return on the invested funds?",
        "SS-46",
    ),
    case(
        "What fee structure is permissible for a Wakalah investment agent?",
        "SS-46",
    ),
    case(
        "Can the bank appoint an employee on a salary tied to a share of the profits he generates?",
        "SS-34", "SS-13",
    ),

    # === coverage: banking operations ====================================
    case(
        "Can the bank charge a commitment fee on an undrawn credit facility?",
        "SS-37", "SS-19",
    ),
    case(
        "What charges may the bank take for issuing a documentary credit?",
        "SS-14",
    ),
    case(
        "Is discounting a bill of exchange before maturity permissible?",
        "SS-16", "SS-07",
    ),
    case(
        "Can a debt owed to the bank be transferred to a third party through Hawalah?",
        "SS-07",
    ),
    case(
        "Can the bank set off a customer's deposit against a debt he owes in a different currency?",
        "SS-04", "SS-01",
    ),
    case(
        "What fees can the bank charge for safe deposit vault and collection services?",
        "SS-28",
    ),
    case(
        "Can an Islamic bank participate in a syndicated facility alongside conventional banks?",
        "SS-24",
    ),
    case(
        "How should the assets of an insolvent debtor be distributed among creditors?",
        "SS-43", "SS-03",
    ),

    # === coverage: contract mechanics ====================================
    case(
        "Is a customer's promise to purchase binding on him in a Murabaha to the purchase orderer?",
        "SS-49", "SS-08",
    ),
    case(
        "Can a single agreement bundle a sale, a lease and a guarantee into one contract?",
        "SS-25",
    ),
    case(
        "What makes uncertainty in a contract excessive enough to invalidate it?",
        "SS-31",
    ),
    case(
        "Can a contract be concluded online by the customer clicking an accept button?",
        "SS-38",
    ),
    case(
        "Can a customer's investment account be mortgaged as security for a facility?",
        "SS-39",
    ),
    case(
        "Does constructive possession satisfy the requirement to take possession of a commodity?",
        "SS-18", "SS-20",
    ),
    case(
        "Can a buyer return goods that turn out to be defective after the sale is concluded?",
        "SS-51", "SS-48",
    ),
    case(
        "Can the parties agree a cooling-off period during which either may revoke the sale?",
        "SS-52",
    ),
    case(
        "What happens to a contract when performance becomes impossible due to force majeure?",
        "SS-36", "SS-11",
    ),
    case(
        "Can the bank sell its rights under a financing contract to another institution for cash?",
        "SS-42", "SS-16",
    ),

    # === coverage: institutional ========================================
    case(
        "How should a conventional bank converting to Islamic banking deal with its existing interest-bearing loans?",
        "SS-06",
    ),
    case(
        "Can we benchmark our product returns against a conventional market index?",
        "SS-27", "SS-47",
    ),
    case(
        "What is the permissible basis for determining the profit rate in a financing transaction?",
        "SS-47",
    ),
    case(
        "Can the government grant a concession to operate a utility under an Islamic structure?",
        "SS-22",
    ),

    # === near-miss pairs =================================================
    # Each pair is surface-similar but governed by different standards.
    # Retrieval that keys on vocabulary rather than on the transaction will
    # collapse them together.
    case(
        "We want to buy a commodity on the metal exchange and sell it immediately to give the customer cash.",
        "SS-30", "SS-44",
        note="Pairs with the next case: this is monetisation, not ordinary trading.",
        kind=NEAR_MISS,
    ),
    case(
        "We want to buy a commodity on the metal exchange to take delivery and resell it to an industrial buyer.",
        "SS-20", "SS-18",
        note="Pairs with the previous case: genuine commodity trade, not tawarruq.",
        kind=NEAR_MISS,
    ),
    case(
        "The customer pays the full price now and we deliver the wheat in six months.",
        "SS-10",
        note=(
            "Salam, paired with the next case; both are advance payment. The one "
            "standing miss in this set. Naming the contract retrieves 6/8 hits from "
            "SS-10; describing the identical transaction in plain business English "
            "retrieves none, landing on SS-44, SS-11 and SS-20 instead. In the full "
            "pipeline parse_query labels it 'salam sale' and SS-10 returns at rank 2, "
            "so production is not currently broken -- but that shows retrieval is "
            "carried by the LLM's contract-type labelling rather than by the "
            "description, which makes a parse error a silent retrieval failure. Kept "
            "failing on purpose as the regression witness for that dependency."
        ),
        kind=NEAR_MISS,
    ),
    case(
        "The customer pays in stages while we manufacture the equipment to his specification.",
        "SS-11",
        note="Istisna'a, not Salam, because the subject is manufactured to order.",
        kind=NEAR_MISS,
    ),
    case(
        "The bank charges the customer a fee for standing behind his obligation to a third party.",
        "SS-05",
        note="Guarantee fee. Pairs with the next case.",
        kind=NEAR_MISS,
    ),
    case(
        "The bank charges the customer a fee for managing his funds and investing them on his behalf.",
        "SS-46", "SS-23",
        note="Agency fee, permissible in a way a guarantee fee is not.",
        kind=NEAR_MISS,
    ),
    case(
        "We lend the customer money and he repays the same amount later with no increase.",
        "SS-19",
        note="Qard. Pairs with the next case: no profit entitlement.",
        kind=NEAR_MISS,
    ),
    case(
        "The customer places funds with us to invest and shares in whatever profit results.",
        "SS-13", "SS-40",
        note="Mudarabah, not a loan; the distinction decides who bears loss.",
        kind=NEAR_MISS,
    ),

    # === adversarial =====================================================
    # Prohibited structures described the way a proposer hoping for approval
    # would describe them. The governing prohibition must still be retrieved.
    case(
        "We charge a fixed administrative service fee on the outstanding balance, recalculated monthly, "
        "to cover the cost of maintaining the account.",
        "SS-19", "SS-08", "SS-02",
        note="Interest renamed as a service fee; the tell is that it scales with balance and time.",
        kind=ADVERSARIAL,
    ),
    case(
        "Our capital protection feature ensures clients never receive back less than they invested, "
        "which we consider prudent risk management rather than a guarantee.",
        "SS-45", "SS-13", "SS-40",
        note="A guarantee of principal, described as prudence.",
        kind=ADVERSARIAL,
    ),
    case(
        "We sell the asset to the customer on deferred payment and repurchase it from him immediately "
        "for a lower cash price, as an efficiency measure.",
        "SS-30", "SS-08",
        note="'Inah. The repurchase by the same party is the prohibited element.",
        kind=ADVERSARIAL,
    ),
    case(
        "Participants contribute a fixed monthly amount to a mutual protection pool, and any surplus "
        "accrues to the shareholders as the operator's return.",
        "SS-26", "SS-41",
        note="Conventional insurance economics under a takaful label.",
        kind=ADVERSARIAL,
    ),
    case(
        "The profit rate on our financing is benchmarked to the prevailing interbank offered rate, "
        "purely as a market reference for pricing.",
        "SS-47", "SS-27", "SS-08",
        note="Benchmarking is treated differently from charging interest; both standards should surface.",
        kind=ADVERSARIAL,
    ),
    case(
        "The customer commits in advance to buy the asset from us, and forfeits his deposit if he does "
        "not, so the bank carries no inventory risk at any point.",
        "SS-08", "SS-49", "SS-53",
        note="Eliminating ownership risk is what makes a Murabaha a financing in disguise.",
        kind=ADVERSARIAL,
    ),
    case(
        "We offer clients exposure to an index of global equities through a contract settled in cash "
        "against the index level, with no purchase of underlying shares.",
        "SS-27", "SS-21", "SS-31",
        note="Cash-settled derivative exposure, framed as index investing.",
        kind=ADVERSARIAL,
    ),
    case(
        "Late payers are asked to pay an additional amount as a deterrent, which the bank collects and "
        "records as other operating income.",
        "SS-03", "SS-08",
        note="A late-payment charge retained by the bank rather than given to charity.",
        kind=ADVERSARIAL,
    ),
]


OUT_OF_SCOPE: list[str] = [
    "What is the capital of France?",
    "How do I reset my online banking password?",
    "What is the weather forecast for Dubai tomorrow?",
]
