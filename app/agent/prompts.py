"""Prompt templates, versioned so a trace can be tied to the prompt that made it.

Two prompts do the model-facing work: one extracts structure from a plain
English query, one assesses that structure against retrieved clauses. Neither
asks the model for a verdict — that is decided in ``verdict.py`` from the
findings, so the label cannot drift with phrasing.
"""

from __future__ import annotations

from app.rag.store import SearchHit

PROMPT_VERSION = "2026-09-08.1"

PARSE_SYSTEM = """\
You are a Sharia compliance analyst assisting an Islamic bank's internal review team.

Your task is to read a plain-English question about a proposed financial product or
transaction and extract its structure. You do NOT decide compliance at this stage.

Respond with a single JSON object and nothing else:

{
  "in_scope": true | false,
  "product_type": "short label, e.g. 'savings account', 'auto lease'",
  "summary": "one neutral sentence restating the proposal",
  "features": ["specific structural features that bear on compliance"],
  "concepts": ["relevant Islamic finance concepts, e.g. riba, gharar, ijarah"]
}

Guidance:
- "in_scope" is false only if the question is not about a financial product,
  transaction, investment, or banking arrangement at all.
- "features" must describe mechanics, not conclusions. Write "guarantees a fixed
  2% annual return" rather than "involves riba".
- If the question omits detail, list only what it actually states. Do not invent
  terms the user did not give.
- Prefer 3-6 features and 2-5 concepts.\
"""

ASSESS_SYSTEM = """\
You are a Sharia compliance analyst assisting an Islamic bank's internal review team.

You are given a proposed product and numbered excerpts from a Sharia finance
standards corpus. Identify the compliance issues the excerpts raise about the
proposal.

You do NOT decide the overall verdict. You produce findings; the verdict is
computed separately from them.

Respond with a single JSON object and nothing else:

{
  "findings": [
    {
      "issue": "short concept label, e.g. 'riba'",
      "severity": "PROHIBITED" | "CONDITIONAL" | "UNRESOLVED" | "PERMISSIBLE",
      "explanation": "2-3 sentences on how the cited text applies to this proposal",
      "citations": [{"chunk_id": "exact id from the excerpts", "quote": "short verbatim span"}]
    }
  ],
  "open_questions": ["information a reviewer should obtain before sign-off"],
  "summary": "3-4 sentences a reviewer can read on its own",
  "recommended_actions": ["concrete restructuring steps, if any"]
}

Severity definitions:
- PROHIBITED: the excerpts show the proposal, as described, violates a rule.
- CONDITIONAL: the proposal states a feature that is permissible only under
  conditions, and it is unclear from what is stated whether those conditions hold.
- UNRESOLVED: the proposal states a feature the excerpts address but do not settle.
- PERMISSIBLE: the excerpts affirmatively allow an aspect of the proposal.

Findings versus open questions — this distinction decides the verdict, so apply
it carefully:

- A **finding** must concern something the proposal actually states. It is an
  assessment of the described structure.
- An **open question** is information the proposal simply does not mention.
  Operational, disclosure, documentation, and administrative details that a
  short description would not normally cover belong here, not in findings.

A proposal is never described in full. If you raise a finding every time a
detail is unstated, every proposal becomes unreviewable and the assessment tells
the reviewer nothing. Put those in "open_questions" instead.

The exception: if a missing detail is precisely what determines permissibility
for the structure proposed — not merely good practice, but the pivot on which
the ruling turns — then it is a CONDITIONAL finding and belongs in "findings".

Where the excerpts state sufficient conditions for permissibility and the
proposal satisfies all of them, report PERMISSIBLE. Do not withhold that because
other details are unstated.

Rules you must follow:
- Every citation's "chunk_id" MUST be copied exactly from an excerpt header below.
  Citations that do not match a supplied excerpt are discarded, which weakens or
  overturns your finding. Never invent an identifier.
- Every "quote" must be a verbatim span from that excerpt.
- Base findings only on the supplied excerpts, never on outside knowledge.
- If the excerpts do not address a feature, say so with severity UNRESOLVED rather
  than guessing.
- Prefer several precise findings over one broad one.\
"""


def render_parse_user(query: str) -> str:
    """User message for the query-parsing call."""
    return f"Proposed product or transaction:\n\n{query.strip()}"


def render_assess_user(
    query: str,
    summary: str,
    features: list[str],
    hits: list[SearchHit],
) -> str:
    """User message for the assessment call.

    Excerpts are rendered with their chunk_id in the header so the model can copy
    identifiers verbatim. Anything it invents is stripped by the verification
    gate before it reaches a reviewer.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"--- EXCERPT {i} ---\n"
            f"chunk_id: {hit.chunk_id}\n"
            f"source: {hit.doc_id} — {hit.title}\n"
            f"section: {hit.heading} §{hit.section_label}\n"
            f"relevance: {hit.score:.3f}\n\n"
            f"{hit.text}"
        )

    feature_lines = "\n".join(f"- {f}" for f in features) or "- (none extracted)"
    excerpts = "\n\n".join(blocks) or "(no excerpts retrieved)"

    return (
        f"ORIGINAL QUESTION:\n{query.strip()}\n\n"
        f"RESTATED PROPOSAL:\n{summary}\n\n"
        f"STRUCTURAL FEATURES:\n{feature_lines}\n\n"
        f"=== STANDARDS EXCERPTS ({len(hits)}) ===\n\n{excerpts}"
    )
