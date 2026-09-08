# Implementation Plan — Sharia Compliance Agent

**Status:** planning complete, implementation not started
**Date:** 2026-09-08

---

## 1. Objective

A production-grade AI agent for Mal's internal compliance team that assesses whether a
proposed financial product or transaction is Sharia-compliant, returning a structured
verdict (`COMPLIANT` / `NON_COMPLIANT` / `NEEDS_REVIEW`) with cited reasoning, exposed
over a public REST API.

---

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Task constraint. 3.11 rather than 3.13 for widest wheel availability. |
| LLM gateway | **OpenRouter** via the `openai` SDK with `base_url=https://openrouter.ai/api/v1` | User has a paid subscription. OpenAI-compatible, so no vendor lock-in in code. |
| Chat model | Env var `OPENROUTER_MODEL` plus an `OPENROUTER_FALLBACK_MODELS` list | Paid sub means we can use a strong model with reliable JSON adherence. The fallback chain covers provider outages and model retirement. |
| Embeddings | **OpenRouter `/embeddings`**, default `openai/text-embedding-3-small` (1536 dims) | Same key, same SDK, same bill as chat. Removes the need to host any model. ~$0.02/M tokens, so indexing the whole corpus costs well under a cent. |
| Vector store | **Qdrant Cloud free tier**, behind a `VectorStore` protocol | User's choice; credentials supplied. Free forever at 0.5 vCPU / 1 GB RAM / 4 GB disk. Index lives outside the container, so re-ingesting never requires a redeploy. |
| Agent framework | **LangGraph** `StateGraph` | Task constraint. All node logic is plain Python in this repo — no black-box abstraction. |
| API | **FastAPI** + Uvicorn | Task-suggested. Native Pydantic schemas give the structured JSON contract for free. |
| Deploy | **Render** free web service, Docker | 750 hours/month, no credit card, public HTTPS URL requiring no sign-in. |

### What the paid OpenRouter subscription changes

The earlier draft of this plan was shaped around the free tier's 50 requests/day ceiling
and the absence of an embeddings endpoint. Both constraints are now gone:

- No meaningful throughput cap, so the two-LLM-call design needs no rationing. The LRU
  cache stays, but as a latency optimisation rather than a quota defence.
- A stronger chat model can be used, so strict-JSON adherence is reliable. The defensive
  parser and repair retry remain as a safety net, not a load-bearing workaround.
- Embeddings come from the same key. No local model, no `torch`, no `onnxruntime`, and no
  second provider account.

### Cost and quota profile

| Item | Where it runs | Cost |
|---|---|---|
| Chat completions (2 per assessment) | OpenRouter | Paid subscription |
| Embeddings (corpus ~25K tokens once; queries trivial) | OpenRouter | Under $0.01 total to index |
| Vector storage (~120 vectors × 1536 dims ≈ 740 KB) | Qdrant Cloud | Free tier, ~0.02% of the 4 GB disk |
| API hosting | Render | Free tier, 750 h/month |

### Memory budget

No longer a binding constraint. Dropping local embeddings and the embedded vector store
removes roughly 290 MB from the earlier estimate:

| Component | Estimated resident |
|---|---|
| FastAPI + Uvicorn + LangGraph | ~80 MB |
| `openai` + `qdrant-client` | ~25 MB |
| Headroom on a 512 MB instance | ~400 MB |

The container is now thin: it holds no model weights and no index, only client code.

---

## 3. Architecture

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
                        check scope. Rejects out-of-scope input early.
                                |
  2. plan_retrieval     Derive 2-4 targeted sub-queries from the
                        extracted features (pure Python + templates).
                                |
  3. retrieve           Embed sub-queries via OpenRouter, search Qdrant,
                        dedupe, rerank, cap the context window.
                                |
  4. [conditional]      Weak retrieval? Broaden queries, loop once (max 1).
                                |
  5. assess             LLM #2: per-issue findings (riba, gharar, maysir,
                        asset backing, prohibited sector), each carrying
                        explicit chunk citations.
                                |
  6. verify_citations   Drop any citation ID absent from the retrieved
                        set. Hallucinated support cannot survive.
                                |
  7. decide_verdict     Deterministic Python rules, not LLM whim.
                                |
                +---------------v----------------+
                |     Structured JSON response   |
                +--------------------------------+

External services:  OpenRouter (chat + embeddings)  |  Qdrant Cloud (vector search)
```

### Why a graph rather than a chain

Two things genuinely need graph semantics: the **conditional retry edge** (weak retrieval
broadens the sub-queries and re-retrieves once) and the **citation-verification gate**
(which can downgrade a verdict and short-circuit). A linear chain would hide both.

### Verdict rules (deterministic, in `agent/verdict.py`)

Aggregation is plain Python so the outcome is auditable and testable. The LLM produces
*findings*; the code produces the *verdict*:

- Any finding of severity `PROHIBITED` that survives citation verification → `NON_COMPLIANT`
- Any `CONDITIONAL` or `UNRESOLVED` finding, or a top retrieval score below threshold, or
  a corpus that lacks coverage for an identified feature → `NEEDS_REVIEW`
- All identified features cleared with grounded citations → `COMPLIANT`
- A `NON_COMPLIANT` verdict whose citations were all stripped as hallucinated is
  downgraded to `NEEDS_REVIEW` — fail safe, never fail confident

Ambiguity resolves toward `NEEDS_REVIEW`. For a compliance tool, a false `COMPLIANT` costs
far more than an unnecessary human review.

---

## 4. Corpus

Five to six short documents, **synthesized in-house** and modeled on the themes of AAOIFI
Shari'ah Standards. Every document carries a header stating that it is an educational
synthesis and **not** authentic AAOIFI text — the agent must never present fabricated
passages as a real standards body's wording.

| ID | Topic |
|---|---|
| SFS-001 | Riba: prohibition of interest and guaranteed returns on deposits |
| SFS-002 | Murabaha: cost-plus sale, ownership and disclosure requirements |
| SFS-003 | Ijarah: leasing, risk of ownership, maintenance obligations |
| SFS-004 | Gharar and maysir: excessive uncertainty, speculation, conventional insurance and derivatives |
| SFS-005 | Mudarabah and Musharakah: profit sharing, loss attribution, profit equalization reserves |
| SFS-006 | Sector and financial screening criteria for permissible investment |

Each document uses stable section anchors so that citations resolve to a real location —
`SFS-001 §3.2`, not a bare filename.

---

## 5. Repository layout

```
app/
  main.py             FastAPI app, route handlers, lifespan
  config.py           pydantic-settings, env loading
  schemas.py          Request/response Pydantic models
  observability/
    logging_setup.py  JSON-lines logger, stdout and file
    trace.py          contextvar trace ID, middleware
  rag/
    embedder.py       Embedder protocol, OpenRouter implementation, batching
    chunking.py       Section-aware chunker that preserves anchors
    store.py          VectorStore protocol, Qdrant implementation, in-memory fake
    ingest.py         Load, chunk, embed, upsert to Qdrant
  agent/
    state.py          AgentState TypedDict
    graph.py          StateGraph wiring and conditional edges
    nodes.py          The six node functions
    prompts.py        Versioned prompt templates
    verdict.py        Deterministic aggregation rules
    llm.py            OpenRouter chat client, retries, fallback chain, JSON repair
corpus/               The source documents
scripts/ingest.py     CLI entry point for building the Qdrant collection
tests/                pytest suite
Dockerfile            Thin runtime image; no model weights, no index
render.yaml           Render blueprint
requirements.txt
.env.example
```

### Qdrant specifics

- Collection created with `size=1536`, `distance=COSINE`, matching
  `openai/text-embedding-3-small`. Changing the embedding model changes the vector
  dimension, so `scripts/ingest.py` supports `--recreate` and validates the existing
  collection's dimension on startup, failing loudly on mismatch rather than silently
  returning nonsense.
- Point payload carries `doc_id`, `section`, `title`, `text`, and `chunk_id` so a retrieval
  result is directly citable without a second lookup.
- Ingest is idempotent: deterministic point IDs derived from `chunk_id`, so re-running
  upserts in place rather than duplicating.
- Retrieval uses `query_points` with a score threshold; the threshold feeds the
  `NEEDS_REVIEW` coverage rule.

---

## 6. Observability

Structured JSON lines to stdout **and** a rotating file. A trace ID (UUID4) is minted per
request by middleware and stored in a `contextvar`, so every module picks it up without
threading it through call signatures. It is returned in the `X-Trace-Id` response header
and in the JSON body.

Logged per assessment, all sharing one `trace_id`:

- `request.received` — method, path, client, query
- `agent.node.start` / `agent.node.end` — node name, latency
- `retrieval.result` — sub-queries, chunk IDs, similarity scores, source document and section
- `llm.request` — model, fully resolved prompt text, token estimate
- `llm.response` — raw completion, latency, finish reason, model actually served
- `verdict.decided` — verdict, the rule that fired, surviving citations
- `request.completed` — status, total latency

`GET /traces/{trace_id}` replays a stored trace, so a reviewer can audit exactly what the
agent saw and sent. Prompt logging is toggleable via `LOG_PROMPTS` for environments where
prompt text is sensitive.

---

## 7. API contract

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/assess` | Submit a plain-English query, receive a structured verdict |
| `GET` | `/health` | Liveness and readiness: Qdrant reachable, collection present, vector count, model configured |
| `GET` | `/corpus` | List ingested documents and section anchors |
| `GET` | `/traces/{trace_id}` | Replay a stored trace for audit |
| `GET` | `/docs` | OpenAPI UI (FastAPI built-in) |

Because retrieval now depends on a network service, `/health` distinguishes **liveness**
(process up) from **readiness** (Qdrant reachable and the collection populated), and
returns 503 when the dependency is down rather than reporting a false green.

`POST /assess` response shape:

```json
{
  "trace_id": "uuid",
  "query": "Can Mal offer a fixed-return savings account?",
  "verdict": "NON_COMPLIANT",
  "confidence": 0.86,
  "summary": "A guaranteed fixed return on a deposit constitutes riba ...",
  "findings": [
    {
      "issue": "riba",
      "severity": "PROHIBITED",
      "explanation": "...",
      "citations": [
        { "doc_id": "SFS-001", "section": "3.2", "title": "...", "quote": "..." }
      ]
    }
  ],
  "recommended_actions": ["Restructure as a Mudarabah profit-sharing account ..."],
  "retrieved_chunks": [{ "chunk_id": "...", "score": 0.71, "doc_id": "SFS-001" }],
  "model": "...",
  "latency_ms": 4210
}
```

Errors return the same envelope shape with an `error` object — never a bare string.

---

## 8. Testing

- **Chunking** — anchors survive, no chunk exceeds the token cap, overlap is correct
- **Verdict rules** — table-driven across every finding-severity combination, including the
  citation-stripped downgrade path
- **Citation verifier** — fabricated chunk IDs are dropped and the verdict downgrades
- **Graph end to end** — stubbed LLM and in-memory fake store, no network; asserts node
  order and that the retry edge fires
- **API contract** — `TestClient`; schema, status codes, `X-Trace-Id` header present
- **Store contract** — the same test suite runs against both the Qdrant implementation and
  the in-memory fake, so the protocol is genuinely honoured
- **Retrieval smoke** — golden query/document pairs (for example, "fixed return deposit"
  must retrieve SFS-001) to guard against embedding or chunking regressions
- **Live smoke** — one opt-in test against the real OpenRouter and Qdrant endpoints, marked
  and skipped by default so CI needs no credentials

The in-memory fake store is what keeps the default suite fast and credential-free — no
test outside the live-smoke marker touches the network.

---

## 9. Phases

1. **Scaffold** — repo, git init, `requirements.txt`, `config.py`, `.env.example`, `.gitignore`
2. **Corpus** — author the documents with stable section anchors
3. **RAG pipeline** — OpenRouter embedder, chunker, store protocol with Qdrant and fake
   implementations, ingest script, collection bootstrap
4. **LLM client** — OpenRouter chat wrapper, fallback chain, defensive JSON parsing, repair retry
5. **Agent** — state, the six nodes, deterministic verdict rules, graph wiring with retry edge
6. **API** — FastAPI app, endpoints, error envelope, readiness check, lifespan warm-up
7. **Observability** — JSON logging, trace contextvar, middleware, trace replay
8. **Tests** — the suite above, green
9. **Docker and deploy** — thin image, Render blueprint, env vars set in the dashboard, live URL
10. **README** — setup, architecture diagram, curl examples, known limitations

---

## 10. Known limitations (to carry into the README)

- The corpus is **synthesized for demonstration**, not authentic AAOIFI text. Output is
  decision support for a qualified reviewer, never a substitute for a Sharia board ruling.
- Retrieval is limited to the ingested corpus. A question outside its coverage should return
  `NEEDS_REVIEW`, and the corpus-coverage check is what enforces that.
- Render's free tier spins down after 15 minutes idle, so the first request after an idle
  period takes about a minute.
- The Qdrant free cluster shares resources, so p99 retrieval latency can spike under load.
  Retrieval adds a network round trip that an embedded store would not.
- Two external dependencies are now on the request path (OpenRouter and Qdrant). Both are
  wrapped with timeouts and retries, and `/health` reports readiness honestly.
- There is no authentication on the public endpoint — it is a demonstration deployment.
  Rate limiting is per-process only.
- Secrets live in Render's environment settings and a local `.env`; `.env.example` documents
  every variable and no real key is ever committed.
