# Sharia Compliance Agent

An AI agent that helps Mal's internal compliance team assess whether a proposed financial
product or transaction is Sharia-compliant. It retrieves from a corpus of Sharia finance
standards, reasons over them with a LangGraph agent, and returns a structured verdict —
`COMPLIANT`, `NON_COMPLIANT`, or `NEEDS_REVIEW` — with cited reasoning.

> **Status: in development.** Scaffold and corpus complete; RAG pipeline in progress.
> See [PLAN.md](PLAN.md) for the full design, and the progress tracker below.

---

## Progress tracker

### Phase 0 — Planning
- [x] Read and break down the task requirements
- [x] Choose the stack and lock architectural decisions
- [x] Verify service constraints (OpenRouter embeddings, Qdrant free tier, Render limits)
- [x] Write [PLAN.md](PLAN.md)
- [x] Write this tracker

### Phase 1 — Scaffold
- [x] `git init`, `.gitignore`, initial commit
- [x] `requirements.txt` pinned for Python 3.11
- [x] `app/config.py` — pydantic-settings, env loading
- [x] `.env.example` documenting `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`,
      `OPENROUTER_EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`
- [ ] Verify a bare `uvicorn app.main:app` boots

### Phase 2 — Corpus
- [x] SFS-001 Riba and guaranteed returns
- [x] SFS-002 Murabaha
- [x] SFS-003 Ijarah
- [x] SFS-004 Gharar and maysir
- [x] SFS-005 Mudarabah and Musharakah
- [x] SFS-006 Sector and financial screening
- [x] Provenance header on every document (synthesized, not authentic AAOIFI text)

### Phase 3 — RAG pipeline
- [ ] `rag/embedder.py` — Embedder protocol, OpenRouter implementation with batching
- [x] `rag/chunking.py` — section-aware chunker preserving `§` anchors
- [ ] `rag/store.py` — VectorStore protocol, Qdrant implementation
- [ ] `rag/store.py` — in-memory fake for credential-free tests
- [ ] Collection bootstrap: 1536 dims, cosine, dimension-mismatch guard
- [ ] `rag/ingest.py` + `scripts/ingest.py` — idempotent upsert with deterministic IDs
- [ ] Confirm retrieval quality on golden queries

### Phase 4 — LLM client
- [ ] `agent/llm.py` — OpenRouter chat client via the `openai` SDK
- [ ] Model fallback chain for provider outages and retired models
- [ ] Defensive JSON extraction with a single repair retry
- [ ] Timeouts and retry with backoff
- [ ] In-process LRU cache keyed on the normalized query

### Phase 5 — Agent
- [ ] `agent/state.py` — AgentState
- [ ] `agent/prompts.py` — versioned prompt templates
- [ ] Node: `parse_query`
- [ ] Node: `plan_retrieval`
- [ ] Node: `retrieve`
- [ ] Node: `assess`
- [ ] Node: `verify_citations`
- [ ] Node: `decide_verdict` — deterministic rules
- [ ] `agent/graph.py` — wiring plus the conditional retry edge

### Phase 6 — API
- [ ] `POST /assess`
- [ ] `GET /health` — liveness vs. readiness, 503 when Qdrant is unreachable
- [ ] `GET /corpus`
- [ ] `GET /traces/{trace_id}`
- [ ] Consistent error envelope
- [ ] Lifespan warm-up and client reuse

### Phase 7 — Observability
- [ ] JSON-lines logger to stdout and a rotating file
- [ ] Trace ID contextvar plus middleware, returned as `X-Trace-Id`
- [ ] Log retrieved chunks with scores
- [ ] Log the fully resolved LLM prompt, behind `LOG_PROMPTS`
- [ ] Log the verdict and the rule that produced it
- [ ] Trace replay endpoint working end to end

### Phase 8 — Tests
- [ ] Chunking tests
- [ ] Verdict rule tests, table-driven
- [ ] Citation verifier tests
- [ ] Graph end-to-end test with stubbed LLM and fake store
- [ ] API contract tests
- [ ] Store contract tests run against both implementations
- [ ] Retrieval smoke tests on golden pairs
- [ ] Opt-in live smoke test, skipped by default so CI needs no credentials
- [ ] Full suite green

### Phase 9 — Deploy
- [ ] `Dockerfile` — thin runtime image, no model weights, no bundled index
- [ ] Local container run verified end to end
- [ ] Qdrant Cloud cluster created and corpus ingested
- [ ] `render.yaml` blueprint, secrets set in the Render dashboard
- [ ] Deployed to Render, public URL live
- [ ] **Verified in an incognito window with no sign-in**

### Phase 10 — Documentation
- [ ] Setup instructions
- [ ] Architecture diagram
- [ ] Example `curl` requests with real responses
- [ ] Known limitations
- [ ] Public repository, verified in an incognito window

---

## Architecture

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

  1. parse_query        LLM #1: extract product structure and features
  2. plan_retrieval     Derive 2-4 targeted sub-queries
  3. retrieve           Embed via OpenRouter, search Qdrant, dedupe, rerank
  4. [conditional]      Weak retrieval? Broaden and loop once
  5. assess             LLM #2: per-issue findings with citations
  6. verify_citations   Drop citations absent from the retrieved set
  7. decide_verdict     Deterministic Python rules
                                |
                +---------------v----------------+
                |     Structured JSON response   |
                +--------------------------------+

External services:  OpenRouter (chat + embeddings)  |  Qdrant Cloud (vector search)
```

The LLM produces *findings*; deterministic Python code produces the *verdict*. Ambiguity
resolves toward `NEEDS_REVIEW` — for a compliance tool, a false `COMPLIANT` costs far more
than an unnecessary human review.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Agent | LangGraph (`StateGraph`) — all node logic visible in this repo |
| LLM | OpenRouter, via the OpenAI-compatible SDK |
| Embeddings | OpenRouter `/embeddings` — same key and bill as chat, no self-hosted model |
| Vector store | Qdrant Cloud free tier, behind a swappable `VectorStore` protocol |
| API | FastAPI + Uvicorn |
| Hosting | Render free tier, Docker |

The runtime container ships no model weights and no index, so it stays thin and starts fast.

## Setup

_To be completed in Phase 10. Planned shape:_

```bash
git clone <repo-url> && cd Sharia-Compliance-Agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY
python scripts/ingest.py  # chunk, embed, and upsert the corpus into Qdrant (run once)
uvicorn app.main:app --reload
```

Ingestion is idempotent and runs against the hosted Qdrant cluster, so it is a one-time
step rather than part of the deploy. Re-running it updates the corpus without redeploying
the API.

## Example request

_To be replaced with a real captured response in Phase 10._

```bash
curl -X POST https://<deployed-url>/assess \
  -H "Content-Type: application/json" \
  -d '{"query": "Can Mal offer a fixed-return savings account?"}'
```

## Known limitations

- The corpus is **synthesized for demonstration** and is not authentic AAOIFI text.
  Output is decision support for a qualified reviewer, never a substitute for a Sharia
  board ruling.
- Retrieval is limited to the ingested corpus. Questions outside its coverage return
  `NEEDS_REVIEW` by design.
- Render's free tier spins down after 15 minutes idle; the first request afterwards takes
  about a minute.
- The Qdrant free cluster shares resources, so p99 retrieval latency can spike under load,
  and retrieval adds a network round trip an embedded store would not.
- Two external services sit on the request path (OpenRouter and Qdrant). Both are wrapped
  with timeouts and retries, and `/health` reports readiness honestly rather than a false
  green when a dependency is down.
- The public endpoint is unauthenticated, and rate limiting is per-process only.
