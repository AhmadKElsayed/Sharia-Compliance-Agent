"""Verdict eval set: queries paired with the outcome a reviewer would expect.

The retrieval set measures whether the governing clause reaches the model. This
set measures what the system does with it, which is the claim the product
actually makes and the one that was previously evidenced only by spot checks.

**Provenance, stated first because it bounds everything below.** These cases were
written by an engineer reading the AAOIFI text, not by a Sharia scholar. Every
``authority`` below is a clause verified to exist in the published English
edition and to say what the case asserts it says -- that much is checkable. What
is *not* certified is the harder question of whether a scholar would agree with
the expected outcome on the borderline cases. ``SCHOLAR_REVIEWED`` is False and
the runner prints it on every report, because a verdict eval that quietly implies
scholarly authority is worse than none.

Treat this as the scaffold the scholar-authored set drops into: the strata, the
runner and the metrics are the reusable part; the expectations are the part that
needs replacing.

Cases carry a ``stratum`` so failures can be attributed:

``clear_prohibited``  a structure the standards prohibit outright. COMPLIANT here
                      is a false COMPLIANT -- the only error that ships a
                      prohibited product, and the metric that governs this set.
``clear_compliant``   a structure the standards permit, with the permitting
                      condition stated in the query. NEEDS_REVIEW here is not
                      dangerous but it is degenerate: an agent that refers
                      everything informs nobody.
``borderline``        genuinely unsettled on the facts given. More than one
                      verdict is defensible, so several are accepted; the point
                      is that the system must not resolve these confidently.
``adversarial``       a prohibited structure in the euphemistic language a
                      proposer hoping for approval would use. Same failure as
                      clear_prohibited, but the prohibition has to survive the
                      phrasing.
``out_of_scope``      not a compliance question. Must return IRRELEVANT without
                      retrieving anything.

On the ``accept`` sets. A single expected verdict would measure luck on the
borderline cases, where a competent reviewer could land in more than one place.
What is never acceptable is a confident approval of a prohibition, so that is
scored separately and exactly rather than averaged into an accuracy number that
hides it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: No scholar has signed off on these expectations. The runner prints this.
SCHOLAR_REVIEWED = False

CLEAR_PROHIBITED = "clear_prohibited"
CLEAR_COMPLIANT = "clear_compliant"
BORDERLINE = "borderline"
ADVERSARIAL = "adversarial"
OUT_OF_SCOPE = "out_of_scope"

COMPLIANT = "COMPLIANT"
NON_COMPLIANT = "NON_COMPLIANT"
NEEDS_REVIEW = "NEEDS_REVIEW"
IRRELEVANT = "IRRELEVANT"

STRATA = (CLEAR_PROHIBITED, CLEAR_COMPLIANT, BORDERLINE, ADVERSARIAL, OUT_OF_SCOPE)


@dataclass(frozen=True)
class VerdictCase:
    query: str
    #: Verdicts a competent reviewer would accept for this query.
    accept: frozenset[str]
    stratum: str
    #: A clause verified to exist in the English edition and to govern this case.
    authority: str
    #: Why that clause settles it, in the reviewer's terms.
    rationale: str
    #: Standards at least one surviving citation should come from.
    must_cite: frozenset[str] = frozenset()
    note: str = ""

    @property
    def forbidden(self) -> frozenset[str]:
        """Verdicts that are a safety failure here, not merely a miss."""
        if self.stratum in (CLEAR_PROHIBITED, ADVERSARIAL):
            return frozenset({COMPLIANT})
        return frozenset()


def case(
    query: str,
    accept: tuple[str, ...],
    stratum: str,
    authority: str,
    rationale: str,
    *must_cite: str,
    note: str = "",
) -> VerdictCase:
    return VerdictCase(
        query=query,
        accept=frozenset(accept),
        stratum=stratum,
        authority=authority,
        rationale=rationale,
        must_cite=frozenset(must_cite),
        note=note,
    )


VERDICT_SET: list[VerdictCase] = [
    # --- clear prohibitions ----------------------------------------------
    case(
        "Can Mal offer a savings account that guarantees depositors a fixed 4% annual return?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-47 7/2",
        "It is not permissible to guarantee the capital, the profit, or both, in a "
        "Mudarabah-based account.",
        "SS-40", "SS-13", "SS-45", "SS-47",
        note="The production failure that motivated the whole eval effort.",
    ),
    case(
        "We want to sign the Murabaha sale with the customer first and buy the asset "
        "from the supplier afterwards. Is that acceptable?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-08 3/2/1",
        "Actual or constructive possession by the Institution must be ascertained "
        "before the onward sale to the customer.",
        "SS-08",
    ),
    case(
        "The customer sells us equipment he already owns, and we immediately sell it "
        "back to him on deferred Murabaha terms. Is this permissible?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-08 2/2/3",
        "The supplier must be a third party; a customer selling an item to the "
        "Institution and repurchasing it is 'Inah.",
        "SS-08",
    ),
    case(
        "In our Mudarabah investment account, can the bank's share be set at 2% of "
        "the capital contributed rather than a share of the profit?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-13 8/1",
        "Profit must be distributed as an agreed percentage of profit, never as a "
        "lump sum or a percentage of the capital.",
        "SS-13", "SS-40",
    ),
    case(
        "Can the bank keep the late payment penalty it charges Murabaha customers as "
        "fee income?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-08 5/6",
        "A late-payment undertaking is permissible only as a donation to charitable "
        "causes verified by the SSB, never for the Institution.",
        "SS-08", "SS-03",
    ),
    case(
        "Our Ijarah contract requires the lessee to carry out and pay for major "
        "structural maintenance of the leased building. Is that allowed?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-09 5/1/7",
        "The lessor may not stipulate that the lessee undertakes the major "
        "maintenance needed to keep the asset delivering the contractual benefit.",
        "SS-09",
    ),
    case(
        "A third party will guarantee the capital of our investment fund in exchange "
        "for an annual fee. Is that structure acceptable?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-45 5/4",
        "An undertaking by a third party to guarantee capital for a fee is a form of "
        "conventional insurance, listed as non-compliant.",
        "SS-45", "SS-05",
    ),
    case(
        "Can we use a debt the farmer already owes us as the capital for a Salam "
        "contract instead of paying cash?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-10 3/1/4",
        "A debt may not be recognised as the capital of Salam.",
        "SS-10",
    ),
    case(
        "We buy the components from the manufacturer for cash and sell them back to "
        "that same manufacturer on deferred terms at a markup. Is this a valid "
        "Istisna'a?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-11 2/2/4",
        "Istisna'a may not be a legal device for mere interest-based financing, and "
        "this is the example the clause itself names.",
        "SS-11",
    ),
    case(
        "In our Musharakah, can the silent partner be entitled to 8% of his "
        "contributed capital each year?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-12 4/3/2/2",
        "A sleeping partner may not take profit as a percentage of capital or a "
        "lump sum.",
        "SS-12",
    ),
    case(
        "Can we require an investment agency client to provide a personal guarantee "
        "covering any investment loss?",
        (NON_COMPLIANT,),
        CLEAR_PROHIBITED,
        "SS-05 2/2/1",
        "A guarantee may not be stipulated in a trust contract except to cover "
        "misconduct, negligence or breach of conditions.",
        "SS-05", "SS-46",
    ),
    case(
        "We buy the property from the customer and lease it straight back to him "
        "under Ijarah Muntahia Bittamleek, transferring ownership at the end. Can "
        "the sale and the leaseback happen the same day?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        CLEAR_PROHIBITED,
        "SS-09 8/5",
        "A period long enough for the asset or its value to change must separate the "
        "purchase from the leaseback, to avoid 'Inah.",
        "SS-09",
    ),

    # --- clear permissions ------------------------------------------------
    case(
        "We offer a Mudarabah investment account where profit is shared at an agreed "
        "70:30 ratio, losses fall on the capital provider unless we are negligent, "
        "and no return is guaranteed. Is this compliant?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-13 8/1",
        "Profit as an agreed percentage of profit, with no capital or profit "
        "guarantee, is the compliant form.",
        "SS-13", "SS-40",
    ),
    case(
        "Mal purchases the machinery from the supplier, takes delivery at its own "
        "warehouse, and only then sells it to the customer on Murabaha at a disclosed "
        "markup. Is this compliant?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-08 3/2/1",
        "Possession ascertained before the onward sale is exactly what the standard "
        "requires.",
        "SS-08",
    ),
    case(
        "Our Murabaha has the customer undertake to donate a percentage of the debt "
        "to charity if he delays payment, with the Shari'ah Supervisory Board "
        "confirming the money reaches charitable causes and not the bank. Is that "
        "permissible?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-08 5/6",
        "The standard permits precisely this undertaking, conditioned on SSB "
        "oversight of the charitable destination.",
        "SS-08", "SS-03",
    ),
    case(
        "In our Ijarah the lessor remains liable for major maintenance but delegates "
        "the work to the lessee and reimburses the lessee's cost. Is that acceptable?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-09 5/1/7",
        "The lessor may delegate the task of major maintenance to the lessee at the "
        "lessor's own cost.",
        "SS-09",
    ),
    case(
        "The customer pays the full Salam price in cash at contract signing and we "
        "deliver a specified quantity and grade of wheat in six months. Is this "
        "compliant?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-10 3/2/9",
        "Salam requires capital paid when the contract is concluded and a delivery "
        "date known without ambiguity; both hold here.",
        "SS-10",
    ),
    case(
        "We exchange dirhams for dollars with delivery and acceptance completed in "
        "the same contract session. Is that compliant?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-18 3/7",
        "Reciprocal possession within the contract session is what Sarf requires.",
        "SS-01", "SS-18",
    ),
    case(
        "Can we take Takaful cover on the leased assets underlying our Sukuk against "
        "destruction and for major maintenance?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-45 4/2/2",
        "Listed explicitly among the permissible methods of protecting capital.",
        "SS-45", "SS-17",
    ),
    case(
        "In our Musharakah the partners agree to set aside a proportion of the profit "
        "as a charitable donation to a non-partner. Is that allowed?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-12 3/1/5/14",
        "Permitted expressly: a proportion of profit may be set aside for "
        "non-partners as charity.",
        "SS-12",
    ),
    case(
        "For our charge card we hold the cardholder's security deposit and invest it "
        "for his benefit on a Mudarabah basis, sharing profit at a stated ratio. Is "
        "that compliant?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-02 3/2/2",
        "The standard sets out this exact treatment for a blocked cardholder deposit.",
        "SS-02",
    ),
    case(
        "In a Murabaha, may Mal authorise a shipping agent to take delivery of the "
        "goods from the supplier on its behalf?",
        (COMPLIANT,),
        CLEAR_COMPLIANT,
        "SS-08 3/2/5",
        "The Institution may authorise another party to take delivery on its behalf; "
        "risk transfers on that possession.",
        "SS-08",
    ),

    # --- genuinely borderline --------------------------------------------
    case(
        "We arrange for the customer to buy platinum from us on deferred payment and "
        "sell it through a broker in the organised market for immediate cash. Is this "
        "permissible?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-30",
        "Organised Tawarruq is tightly constrained; permissibility turns on facts the "
        "query leaves open, such as whether the Institution acts as the customer's "
        "selling agent.",
        "SS-30", "SS-20",
    ),
    case(
        "Can we maintain a profit equalisation reserve to smooth returns paid to "
        "investment account holders across periods?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-40",
        "Permitted subject to disclosure, account-holder consent and the reserve's "
        "ownership on liquidation, none of which the query establishes.",
        "SS-40", "SS-13",
    ),
    case(
        "Our Ijarah rental floats at a benchmark rate plus a margin, reset annually, "
        "with a stated cap and floor. Is that acceptable?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-09 5/2",
        "A variable rental tied to a known benchmark is permitted where each period's "
        "rental becomes determinate; the cap and floor are what make it arguable.",
        "SS-09",
    ),
    case(
        "Is a binding promise from the customer to purchase the asset, taken before we "
        "acquire it, acceptable in Murabaha to the purchase orderer?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-49",
        "A unilateral binding promise is permitted; binding both sides would convert "
        "the promise into a sale of what the Institution does not yet own.",
        "SS-49", "SS-08",
    ),
    case(
        "We want one document combining the Murabaha sale, the agency appointment and "
        "the security arrangement. Is combining them permissible?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-25",
        "Combination is permitted unless the contracts are made conditional on one "
        "another or the combination is a device for riba; the query says neither.",
        "SS-25", "SS-08",
    ),
    case(
        "Our Sukuk carries an undertaking from the originator to buy the assets back "
        "at face value on maturity. Is that compliant?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-17",
        "A purchase undertaking at nominal value guarantees the holders' capital; "
        "whether it is at market or net value instead is what decides it.",
        "SS-17", "SS-45",
    ),
    case(
        "May our equity fund hold shares in a listed manufacturer that earns a small "
        "amount of interest income on its cash balances?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-21",
        "Permitted within quantitative screening thresholds and with purification of "
        "the tainted income; the query gives no ratios.",
        "SS-21",
    ),
    case(
        "If the buyer walks away, may we keep the earnest money he paid when the sale "
        "was agreed?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-53",
        "'Arboun is permitted in specified contracts with a defined waiting period; it "
        "turns on the contract type the query does not name.",
        "SS-53",
    ),
    case(
        "To hedge currency exposure, can we execute back-to-back interest-free loans "
        "in two currencies with a counterparty?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-01 2/4/1",
        "Permitted only where neither party takes extra benefit and the two loans are "
        "not contractually connected, which the query does not establish.",
        "SS-01",
    ),
    case(
        "In an investment agency, may we tell the client an expected profit rate and "
        "keep anything earned above it as an incentive?",
        (COMPLIANT, NEEDS_REVIEW),
        BORDERLINE,
        "SS-46",
        "An indicative rate plus a performance incentive is permitted where it is not "
        "a guarantee; the wording of the undertaking decides it.",
        "SS-46", "SS-47",
    ),

    # --- adversarial: prohibitions in approving language -------------------
    case(
        "Our new deposit product includes a capital protection feature so customers "
        "always get their money back at maturity. Please confirm it is acceptable.",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-47 7/2",
        "'Capital protection' on an investment account is a capital guarantee, which "
        "the standard prohibits.",
        "SS-45", "SS-40", "SS-13", "SS-47",
    ),
    case(
        "We charge an administrative fee that increases with each day a payment is "
        "late, to cover our collection costs. That is a fee, not interest, so it "
        "should be fine?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-08 5/6",
        "A charge scaling with elapsed time on a debt is riba however it is labelled; "
        "only a charitable donation undertaking is permitted.",
        "SS-08", "SS-03",
    ),
    case(
        "The customer nominates the supplier for the Murabaha. The supplier is a "
        "company the customer happens to own 80% of. That is still a third party, "
        "correct?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-08 2/2/3",
        "The standard names majority ownership -- more than 50% -- as disqualifying "
        "the supplier from counting as a third party.",
        "SS-08",
    ),
    case(
        "We purchase the metal and our broker sells it on within the same minute, so "
        "the customer receives cash. It is a genuine commodity sale, so is it "
        "permissible?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-30",
        "Immediate resale through the Institution's own broker is the organised "
        "Tawarruq the standard constrains, not a genuine commodity trade.",
        "SS-30", "SS-20",
    ),
    case(
        "On our benevolent loan product we levy a monthly service charge calculated "
        "as a small percentage of the outstanding balance. It covers administration, "
        "so it is not interest, is it?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-19",
        "A charge proportional to the outstanding balance is a benefit drawn from a "
        "loan; only actual administrative cost may be recovered.",
        "SS-19",
    ),
    case(
        "Our fund tells investors their principal is returned at maturity, backed by a "
        "guarantee our own subsidiary provides for an annual fee. Acceptable?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-45 5/4",
        "A guarantee of capital for a fee is a non-compliant means of protecting "
        "capital; a related-party guarantor does not change that.",
        "SS-45", "SS-05",
    ),
    case(
        "We quote an all-inclusive rent under which the lessee handles every repair, "
        "structural ones included, so the tenant has cost certainty. Fine?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-09 5/1/7",
        "Bundling major maintenance into the lessee's obligation is the stipulation "
        "the standard forbids, whatever the rent is called.",
        "SS-09",
    ),
    case(
        "Our silent investor receives a predictable annual distribution of 8% of what "
        "he put in, which keeps his planning simple. Is that structure acceptable?",
        (NON_COMPLIANT, NEEDS_REVIEW),
        ADVERSARIAL,
        "SS-12 4/3/2/2",
        "A distribution set as a percentage of contributed capital is prohibited "
        "however convenient its predictability is.",
        "SS-12", "SS-13",
    ),

    # --- out of scope ------------------------------------------------------
    case("Hi", (IRRELEVANT,), OUT_OF_SCOPE, "", "A greeting is not a compliance question."),
    case(
        "Assalamu alaikum, how are you today?",
        (IRRELEVANT,),
        OUT_OF_SCOPE,
        "",
        "A salutation, including one phrased as a question.",
    ),
    case(
        "What is the weather forecast for Dubai this weekend?",
        (IRRELEVANT,),
        OUT_OF_SCOPE,
        "",
        "Off-topic; nothing in the corpus bears on it.",
    ),
    case(
        "Can you write me a Python script that parses a CSV file?",
        (IRRELEVANT,),
        OUT_OF_SCOPE,
        "",
        "A software request, not a product to assess.",
    ),
    case(
        "Who won the football World Cup in 2022?",
        (IRRELEVANT,),
        OUT_OF_SCOPE,
        "",
        "General knowledge; must not be dressed up as a compliance answer.",
    ),
]


def by_stratum(stratum: str) -> list[VerdictCase]:
    return [c for c in VERDICT_SET if c.stratum == stratum]
