# Implementation Plan & Build Log — Sharia Compliance Agent

**Status:** delivered. All ten phases complete, deployed at
<https://sharia-compliance-agent.onrender.com>, 174 tests green.
**Planned:** 2026-09-08 · **Completed:** 2026-09-09

This document is kept as a record of what was planned and **where reality
diverged from it**. Section 2 is the honest part: eight decisions in the original
plan turned out to be wrong, and the reasons are more informative than the plan
itself. Deep architectural rationale lives in `DOCUMENTATION.md`; usage lives in
`README.md`.

---

## 1. Objective

A production-grade AI agent for Mal's internal compliance team that assesses
whether a proposed financial product or transaction is Sharia-compliant,
returning a structured verdict (`COMPLIANT` / `NON_COMPLIANT` / `NEEDS_REVIEW`)
with cited reasoning, exposed over a public REST API.

**Delivered.** Unchanged from the original objective, with one addition: a
message that is not a compliance question returns `IRRELEVANT` rather than being
forced into one of the three assessment verdicts (divergence 7).

---

## 2. Where the plan changed

| # | Planned | Actually built | Why it changed |
|---|---|---|---|
| 1 | **Synthesised corpus** — 6 in-house documents modelled on AAOIFI themes | **The real AAOIFI *Shari'ah Standards*** — 1,264 pages, 54 standards, 1,973 clauses, parsed from the published PDF | The user supplied the real PDFs mid-build and asked to "build something real using the real docs". The synthesised corpus was a trap: paraphrased rules produce fluent output whose citations point at documents that exist only in this repo, making the system's central claim unverifiable. |
| 2 | `text-embedding-3-small`, 1536 dims | **`text-embedding-3-large`, 3072 dims** | Retrieval quality is the binding constraint on verdict quality, and the cost difference on a 147K-token corpus is under two cents. |
| 3 | **Single-stage vector search**, top_k=5 | **Grounded sub-queries → RRF fusion → cross-encoder rerank**, top_k=10 | A live query retrieved the general riba prohibition instead of the deposit rules that govern it. Diagnosis via trace showed a retrieval failure, not a reasoning one. See §4. |
| 4 | **Docker image + `render.yaml` blueprint** | **Native Python runtime, configured in the Render dashboard** | Render builds Python natively from `requirements.txt`; a Dockerfile added a build step and an image to maintain for no benefit at this size. Neither file exists in the repo. |
| 5 | **Retrieval smoke tests** — a handful of golden query/document pairs inside the pytest suite | **A standalone golden eval set** (now 66 cases) with its own runner (`scripts/eval_retrieval.py`) | Retrieval quality needed to be *measured and compared across configurations*, not merely asserted. Folding it into pytest would have made it a pass/fail gate rather than a metric, and it needs network access the offline suite deliberately avoids. |
| 6 | Verdict rules: `CONDITIONAL` **or** `UNRESOLVED` → `NEEDS_REVIEW`, flat | **Findings vs open questions split**, with an ordered rule table | With a 1,973-clause corpus the model raised `UNRESOLVED` findings about anything the query was silent on — and every query is silent on something. The rule as planned made nothing ever `COMPLIANT`. See §5. |
| 7 | Three verdicts, per the brief | **A fourth, `IRRELEVANT`** | Greetings and off-topic messages returned `NEEDS_REVIEW`, which is a work queue: it told a reviewer a hello needed their judgement and corrupted the `NEEDS_REVIEW` rate the eval treats as a degeneracy alarm. The three assessment outcomes are unchanged; `IRRELEVANT` records that nothing was assessed. |
| 8 | Scope checked once, implicitly | **An LLM intent router on the first node** | `parse_query` classifies GREETING / ASSESSMENT / OTHER and only ASSESSMENT reaches retrieval. A keyword list was built first and discarded — see §4. |

Two planned items were **not built** and are recorded as cuts in
`DOCUMENTATION.md` §6: durable trace storage and the offline replay harness.

Two things the plan never contemplated were added after it: **batched sub-query
embeddings** (§4, worth ~1.3s per assessment) and the **near-miss and adversarial
eval categories** (§9), the second of which immediately found a retrieval
dependency the plan had no concept of.

---

## 3. Locked decisions — as built

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Task constraint. 3.11 rather than 3.13 for widest wheel availability. |
| LLM gateway | **OpenRouter** via the `openai` SDK | Paid subscription. OpenAI-compatible, so no vendor lock-in in code. Chat, embeddings **and** reranking all traverse one key. |
| Chat model | **`deepseek/deepseek-v4-flash-0731`**, with `OPENROUTER_FALLBACK_MODELS` | Reliable JSON adherence at low cost. The fallback chain covers provider outages and model retirement. |
| Embeddings | **`openai/text-embedding-3-large`** (3072 dims) via OpenRouter `/embeddings` | Same key, same SDK, same bill as chat. ~$0.02 to index the whole corpus. |
| Reranking | **`voyageai/rerank-2.5`** via OpenRouter `/rerank` | Cross-encoder over the fused candidates. No extra credential — though the endpoint is live but undocumented, which is a supply-chain caveat. |
| Vector store | **Qdrant Cloud free tier**, behind a `VectorStore` protocol | Credentials supplied by the user. Index lives outside the container, so re-ingesting never requires a redeploy. |
| Agent framework | **LangGraph** `StateGraph` | Task constraint. All node logic is plain Python in this repo — no black-box abstraction. |
| API | **FastAPI** + Uvicorn | Native Pydantic schemas give the structured JSON contract for free. |
| Deploy | **Render** free web service, **native Python** | 750 hours/month, no credit card, public HTTPS URL requiring no sign-in. |

### Reasoning calibration

The chat model is a reasoning model, and how much it is allowed to reason turned
out to change *correctness*, not just latency. Benchmarked over four queries —
two with obvious answers, two where the corpus says "refer for review" rather
than yes or no:

| Effort | Verdicts correct | Latency | Completion tokens |
|---|---|---|---|
| off | 4 / 4 | 8–12s | 850–1,500 |
| **low** | **4 / 4** | 16–30s | 2,000–5,600 |
| high | **3 / 4** | 28–71s | 4,900–13,600 |

`high` is not a safer setting, it is a worse one. On a Mudarabah account that is
genuinely compliant it invented two CONDITIONAL findings — "loss allocation",
"profit realisation timing" — about details the query never raised, which
downgraded a correct COMPLIANT to NEEDS_REVIEW. The same over-flagging appeared
on the SOFR query, where it raised four UNRESOLVED findings about unstated asset
details, disclosure, and late-payment terms.

That failure mode is specifically damaging here. The verdict rules already
resolve ambiguity toward NEEDS_REVIEW; a model that manufactures conditions on
top of that drives every verdict to NEEDS_REVIEW, at which point the agent tells
a reviewer nothing.

`low` is the default. It matched `off` on every verdict while producing better
output: it mapped the benchmark-pricing clause to UNRESOLVED rather than
CONDITIONAL, matching the corpus language "referred for review", and it
consolidated findings that `off` emitted as four restatements of a single issue
(riba / guaranteed principal / substance over form / lack of risk-sharing).

Two related settings exist because of this. `LLM_MAX_TOKENS` defaults to 12,000:
at 3,072 the reasoning pass consumed the entire budget (3,072 of 3,072 tokens
were reasoning) and the call returned empty content. `OPENROUTER_PROVIDER_SORT`
defaults to `throughput`: OpenRouter serves this model from 29 providers, and
default routing picked one running at 15.6 tok/s where throughput-sorted routing
picked one at 130.3 tok/s.

Caveat on method: one run per cell, four queries. Enough to reject `high`, whose
failure has a clear structural mechanism, but not a precise quality ranking of
`low` against `off`.

### Cost profile — as built

| Item | Where it runs | Cost |
|---|---|---|
| Chat completions (2 per assessment) | OpenRouter | Paid subscription |
| Embeddings (corpus ~147K tokens once; queries trivial) | OpenRouter | **≈ $0.02** to index |
| Reranking (24 candidates per assessment) | OpenRouter → Voyage | Negligible at demo volume; **~28% of the bill** at 50k/day (see `DOCUMENTATION.md` §2.1) |
| Vector storage (1,973 vectors × 3072 dims ≈ **24 MB**) | Qdrant Cloud | Free tier, ~0.6% of the 4 GB disk |
| API hosting | Render | Free tier, 750 h/month |

The corpus ended up ~16× the planned vector count and ~32× the planned storage
(the vectors are also twice as wide), and it is still well inside the free tier —
the plan's caution about storage was misplaced. The real constraint is 0.5 shared
vCPU serving ~4 searches per assessment.

### Memory budget

The container holds no model weights and no index, only client code. Never became
a binding constraint, as expected.

---

## 4. Architecture — as built

```
                    POST /assess   { "query": "..." }
                                |
                +---------------v----------------+
                |  FastAPI                       |
                |   - trace-id middleware        |
                |   - Pydantic validation        |
                +---------------+----------------+
                                |
                +---------------v----------------+
                |      LangGraph StateGraph      |
                +--------------------------------+

  1. parse_query        LLM #1: extract product structure and features,
                        check scope.
                                |
  1b.[conditional]      Intent GREETING or OTHER? Straight to decide_verdict
                        as IRRELEVANT, skipping planning, embedding, search
                        and reranking entirely.
                                |
  2. plan_retrieval     Derive 2-4 sub-queries, each grounded in the
                        product name (pure Python + templates).
                                |
  3. retrieve           Embed all sub-queries in one batched call, search
                        Qdrant top_k=10 each, fuse by reciprocal rank
                        (K=60), rerank 24 -> 8.
                                |
  4. [conditional]      Weak retrieval? Broaden queries, loop once (max 1).
                                |
  5. assess             LLM #2: findings (riba, gharar, maysir, asset
                        backing, prohibited sector) with explicit chunk
                        citations, plus open questions kept separate.
                                |
  6. verify_citations   Drop any citation ID absent from the retrieved
                        set. Hallucinated support cannot survive.
                                |
  7. decide_verdict     Deterministic Python rules, not LLM whim.
                                |
                +---------------v----------------+
                |     Structured JSON response   |
                +--------------------------------+

External:  OpenRouter (chat + embeddings + rerank)  |  Qdrant Cloud (search)
```

### The retrieval rebuild (divergence #3)

The plan's single-stage vector search shipped and then failed in production on
the query *"Can Mal offer a savings account guaranteeing depositors a fixed 4%
annual return?"* — it returned the general riba prohibition rather than the
deposit rules that actually govern the case. The trace showed why: the planner
had emitted the bare sub-query `"riba"`.

Four fixes, in order — the first two for correctness, the third for latency,
the fourth for ranking quality:

1. **Grounded sub-queries.** Anchor each sub-query to the product —
   `"savings account guaranteed return"`, not `"riba"`.
2. **Rank fusion instead of score merging.** Similarity scores from different
   sub-queries are not on a common scale, so averaging them is arithmetic on
   incomparable numbers. RRF (K=60) uses only within-query rank.
3. **Batched embeddings.** All sub-queries in one call rather than four
   sequential ones: 1,925 ms → 594 ms measured live, ~1.3s off every
   assessment. Purely latency — embeddings bill per token, so the cost is
   unchanged.
4. **Cross-encoder reranking over a wider pool.** top_k 5 → 10 with
   `voyageai/rerank-2.5` reranking 24 candidates down to 8.

Measured on the golden set, the fourth fix is only worth it paired with the
wider pool (numbers from the 22-case set that existed then; the comparison
between rows holds, the absolutes are a snapshot):

| Configuration | Recall | MRR | Latency/query |
|---|---|---|---|
| top_k=5, no rerank | 0.955 | 0.898 | 0.72s |
| top_k=10, no rerank | 0.955 | 0.898 | 0.66s |
| top_k=5, rerank | 0.955 | 0.895 | 1.18s |
| **top_k=10, rerank** | **1.000** | **0.924** | 1.26s |

Neither half helps alone. Shipped separately, either would have looked like a
no-op and been reverted — which is the clearest argument in this project for
building the eval set *before* tuning.

### The intent router — unplanned, and it changed the response contract

The plan had no concept of a message that is not a question. Everything was an
assessment, so a greeting was parsed, planned, embedded, searched, reranked and
then labelled NEEDS_REVIEW.

Two changes followed. `parse_query` became a router, classifying each message as
GREETING, ASSESSMENT or OTHER, with only ASSESSMENT continuing to retrieval. A
deterministic greeting word list was built first and discarded: it cannot cover
other languages, transliterated salutations, typos or informal phrasing without
growing without limit, and each word added widens the chance of swallowing a real
query. The LLM classification rides on the parse call that already runs, so it
costs nothing extra; the word list only ever saved one call on the cheapest path
in the system.

Then the verdict enum gained `IRRELEVANT`, a deliberate deviation from the
brief's three verdicts. `NEEDS_REVIEW` is a work queue, and putting greetings in
it mixes noise into the exact population a reviewer works from. The three
assessment outcomes are untouched; `IRRELEVANT` says no assessment happened.

### The scope short-circuit — the plan was right, the build was not

The plan said `parse_query` "rejects out-of-scope input early". It did not. The
scope check was implemented only inside `should_broaden`, the routing function
attached to `retrieve`, so an out-of-scope query was embedded, searched and
reranked before anything read the flag — the greeting reached the verdict with
eight irrelevant clauses attached at a top score of 0.11.

Fixed by adding the conditional edge the plan implied, after `parse_query`
rather than after `retrieve`. A greeting now costs 1.6s and one LLM call instead
of 4.9s, three network calls and a rerank charge.

Recorded here rather than quietly patched because of what it says about reading
one's own design: the sentence "rejects out-of-scope input early" appeared in the
plan, in the graph docstring, and in the README, and was true of the *verdict*
while being false of the *work done to reach it*. Nothing contradicted it, and no
test asserted it, so it survived four rounds of documentation.

### Verdict rules — as built (divergence #6)

Aggregation is an **ordered rule table** in `app/agent/verdict.py`; the first
match wins and the rule name is recorded in the response, so a reviewer sees
exactly why the verdict came out as it did:

1. Out of scope → `NEEDS_REVIEW` (`out_of_scope`)
2. Top score below the coverage threshold → `NEEDS_REVIEW`
   (`insufficient_corpus_coverage`)
3. No findings at all → `NEEDS_REVIEW` (`no_findings`) — silence is not evidence
4. `PROHIBITED` findings, all citations stripped → `NEEDS_REVIEW`
   (`prohibited_findings_lost_all_citations`) — never convict on fabricated
   evidence
5. `PROHIBITED` with a surviving citation → `NON_COMPLIANT`
   (`grounded_prohibition`)
6. `UNRESOLVED` → `NEEDS_REVIEW` · 7. `CONDITIONAL` → `NEEDS_REVIEW`
8. `PERMISSIBLE` but ungrounded → `NEEDS_REVIEW`
9. All findings permissible and grounded → `COMPLIANT`

**The findings/open-questions split** is what the plan missed. With a
1,973-clause corpus the model began raising `UNRESOLVED` findings about details
the query was simply silent on — commingling disclosure, and so forth. That
logic is degenerate: *every* query is silent on something a corpus this size
mentions, so "silent on X → `NEEDS_REVIEW`" makes nothing ever `COMPLIANT`.
Findings now concern what the proposal actually states and drive the verdict;
open questions reach the reviewer but never change it. The exception is explicit
in the prompt: where a missing detail *is* the pivot on which permissibility
turns, it stays a `CONDITIONAL` finding.

---

## 5. Corpus — as built (divergence #1)

The real **AAOIFI *Shari'ah Standards*, English edition**: 1,264 pages, 54
standards, **1,973 normative clauses**, ~103,500 words, extracted from the
published PDF into one chunk per numbered clause, each embedded under its full
heading lineage.

The synthesised corpus survives as `corpus/demo/` — six documents (SFS-001
through SFS-006) — solely so the test suite and a fresh clone run without the
licensed PDFs. It is no longer what the deployed system reasons from.

**Extraction was the unplanned work.** Three problems, two of which corrupted
data silently:

- **A duplicated text layer.** The PDF fakes bold by drawing every span twice.
  Fixed exactly, not heuristically, by dropping any span already drawn at the
  same coordinates (66 spans per page, 33 unique).
- **A one-character regex bug.** Standards 42–44 and 46–48 are headed
  `"No (44)"` while the rest use `"No. (8)"`. Requiring the period made six
  standards invisible and their clauses silently inherited the preceding
  standard's number. Nothing failed; it surfaced only because Standard 41 had an
  implausible 106 clauses.
- **Headings that look like rules.** `2/1` is a heading where clauses nest
  beneath it and a rule where they do not — resolved per standard, 336 headings
  excluded.

The second of those is the strongest argument in this project for
extraction-level assertions at ingest, which are **not** built
(`DOCUMENTATION.md` §3.3).

---

## 6. Repository layout — as built

```
app/
  main.py             FastAPI app, endpoints, lifespan
  config.py           pydantic-settings, env loading
  schemas.py          Request/response Pydantic models
  observability/
    logging_setup.py  JSON-lines logger, stdout and file
    trace.py          contextvar trace ID, middleware
    traces.py         Bounded in-process trace store for replay
  rag/
    aaoifi.py         AAOIFI PDF extractor (span dedup, clause hierarchy)
    pdf_loader.py     PDF text extraction
    chunking.py       Chunk model, contextual headers
    embedder.py       Embedder protocol, OpenRouter + offline hashing
    store.py          VectorStore protocol, Qdrant + in-memory
    rerank.py         Reranker protocol, OpenRouter cross-encoder
    ingest.py         Load, chunk, embed, upsert
  agent/
    state.py          AgentState TypedDict
    graph.py          StateGraph wiring and conditional edges
    nodes.py          The six node functions
    prompts.py        Versioned prompt templates (PROMPT_VERSION)
    verdict.py        Deterministic aggregation rules
    llm.py            OpenRouter chat client, retries, fallback, JSON repair
corpus/
  pdf/                The real AAOIFI standards (EN + AR)
  demo/               Synthesised corpus, so a clone runs without them
  source/             Markdown sources for the demo corpus
evals/golden_set.py   66 cases: coverage, near-miss, adversarial
scripts/              ingest.py, eval_retrieval.py, build_corpus_pdfs.py
tests/                174 tests
requirements.txt      Runtime only
requirements-ingest.txt   Adds PyMuPDF + ReportLab for offline extraction
.env.example
```

**Not built:** `Dockerfile`, `render.yaml` (divergence #4).

`requirements.txt` splitting into three was unplanned: PyMuPDF is AGPL and is
used only by the offline extractor, so keeping it out of the runtime image was
worth a separate file.

### Qdrant specifics — as built

- Collection created with `size=3072`, `distance=COSINE`, matching
  `text-embedding-3-large`. `scripts/ingest.py` supports `--recreate` and
  validates the existing collection's dimension, failing loudly on mismatch
  rather than silently returning nonsense.
- Payload carries `chunk_id`, `doc_id`, `title`, `topic`, `heading`,
  `section_label`, `citation`, `text`, `breadcrumb`, `lang` — so a retrieval
  result is directly citable without a second lookup, and an Arabic edition
  becomes a filter rather than a re-ingest.
- Ingest is idempotent: deterministic UUIDs derived from `chunk_id`.
- **Unplanned:** upserts are batched at 128 points. The full corpus in one
  request is ~24 MB, and the managed endpoint closes the connection rather than
  returning an error.

---

## 7. Observability — as built

Structured JSON lines to stdout **and** a file. A trace ID is minted or adopted
per request by middleware and stored in a `contextvar`, so every module picks it
up without threading it through call signatures. It is returned in the
`X-Trace-Id` response header and in the JSON body.

Logged per assessment, all sharing one `trace_id`:

- `request.received` — path, query
- `retrieval.result` — sub-queries, chunk IDs, similarity scores, top score
- `llm.request` — model, fully resolved prompt text, token estimate
- `llm.response` — raw completion, latency, finish reason, model actually served
- `verdict.decided` — verdict, the rule that fired, confidence
- `request.completed` — verdict, latency, rejected citations, errors

`GET /traces/{trace_id}` replays a stored trace. Prompt logging is toggleable via
`LOG_PROMPTS`.

**Two refinements the plan did not anticipate:**

- `llm.request` fires **before** the API call, not after, so a call that times
  out or fails still leaves behind exactly what was sent.
- Traces must record **token composition**, not just totals. The
  `max_tokens=3072` failure presented as "the model is bad at JSON" and was only
  diagnosable once reasoning-vs-content token counts were visible.

Trace storage is bounded and in-process, so traces are lost on restart and are
invisible across instances. Recorded as a cut.

---

## 8. API contract — as built

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/assess` | Submit a plain-English query, receive a structured verdict |
| `GET` | `/health` | Liveness and readiness: Qdrant reachable, collection populated, config complete |
| `GET` | `/corpus` | What is indexed — counts and identifiers only |
| `GET` | `/traces/{trace_id}` | Replay a stored trace for audit |
| `GET` | `/docs` | OpenAPI UI |

`/health` returns **503** when Qdrant is unreachable or empty or configuration is
incomplete, rather than reporting a false green.

**Changed from the planned response shape:** `summary` became `rationale`
(generated by the verdict rule, not the model), `rule` was added so the reviewer
can see which rule fired, `open_questions` was added (divergence #6), and
`rejected_citations` was added so a stripped hallucination is visible rather than
silent. See `README.md` for a real captured response.

**`/corpus` deliberately returns counts and identifiers only**, never bulk clause
text — the standards are licensed, and a public endpoint should not serve them
back. The plan did not consider this.

Errors return the same envelope shape with an `error` object — never a bare
string.

---

## 9. Testing — as built

174 tests, ~8s, no network and no credentials. The in-memory store implements the
same protocol as Qdrant, a deterministic hashing embedder stands in for the API,
and the LLM is scripted.

- **Chunking** — anchors survive, contextual prefixes are applied, no chunk
  exceeds the cap
- **Extraction (demo corpus only)** — document IDs read from the control table,
  running headers stripped, clause numbering preserved, clauses rejoined across
  page breaks. Note this exercises `pdf_loader.py`, **not** the AAOIFI extractor
- **Verdict rules** — table-driven across every finding-severity combination,
  including the citation-stripped downgrade path
- **Citation verifier** — fabricated chunk IDs are dropped and the verdict
  downgrades
- **Graph end to end** — stubbed LLM and in-memory store; asserts node order and
  that the retry edge fires exactly once
- **API contract** — `TestClient`; schema, status codes, `X-Trace-Id`, and
  `/health` reporting 503 honestly
- **Store contract** — the same suite runs against both implementations
- **Reranking** — reorders results, sees the breadcrumb, and **degrades rather
  than fails** when the reranker raises or the transport errors

**Divergence #5:** the planned "retrieval smoke tests" became a standalone eval
set (`evals/golden_set.py`, 66 cases + 3 out-of-scope) run by
`scripts/eval_retrieval.py`, outside pytest. Retrieval quality needed to be a
*measurement comparable across configurations*, not a boolean gate — and it needs
network access the offline suite deliberately avoids.

Current: **recall 0.985, recall@1 0.833, MRR 0.890, precision 0.733**,
out-of-scope max score 0.255 against a 0.35 threshold. Broken out by kind:
coverage 1.000 (50 cases), near-miss 0.875 (8), adversarial 1.000 (8).

**Unplanned:** `tests/conftest.py` redirects the suite's logging to a temp path.
Tests shared the production log, so each run appended ~168 lines of scripted
output to `logs/app.jsonl` until the file no longer resembled the system it was
meant to demonstrate. A fixture now asserts the production log is untouched.

**Not built:** the live smoke test against real endpoints, and
`tests/test_aaoifi.py` — the highest-risk module is still exercised only
indirectly.

---

## 10. Phases — all complete

| # | Phase | Status |
|---|---|---|
| 1 | Scaffold — repo, requirements, config, `.env.example`, `.gitignore` | done |
| 2 | Corpus | done — **replaced** by the real AAOIFI standards mid-build |
| 3 | RAG pipeline — embedder, chunker, store protocol, ingest | done |
| 4 | LLM client — fallback chain, defensive JSON parsing, repair retry | done |
| 5 | Agent — state, six nodes, verdict rules, graph wiring | done |
| 6 | API — endpoints, error envelope, readiness check, lifespan | done |
| 7 | Observability — JSON logging, trace contextvar, middleware, replay | done |
| 8 | Tests | done — 156 green |
| 9 | Deploy | done — **native Python on Render**, not Docker |
| 10 | README | done |
| — | *Unplanned:* AAOIFI PDF extraction | done |
| — | *Unplanned:* retrieval rebuild — grounding, RRF, reranking | done |
| — | *Unplanned:* golden eval set and runner | done |
| — | *Unplanned:* architecture & trade-offs document | done |

---

## 11. Known limitations

Superseded by `README.md` § Known limitations and `DOCUMENTATION.md` §6, which
are maintained. The plan's original list was written before the corpus changed
and is no longer accurate — in particular, the corpus is **not** synthesised, and
the first item below replaces it.

- Output is **decision support for a qualified reviewer**, never a substitute for
  a Sharia Supervisory Board ruling.
- **Verdict quality is unmeasured.** Retrieval is measured; verdict correctness
  is not, and needs a scholar-authored eval set.
- Render's free tier spins down after 15 minutes idle, so the first request after
  an idle period takes about 50s.
- The Qdrant free cluster shares resources, so p99 retrieval latency can spike.
- Three external calls are on the request path (chat, embeddings, rerank), all
  through OpenRouter. Reranking degrades gracefully; the other two do not.
- No authentication on the public endpoint — it is a demonstration deployment.
- Secrets live in Render's environment settings and a local `.env`;
  `.env.example` documents every variable and no real key is committed.
