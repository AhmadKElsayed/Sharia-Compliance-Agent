Build a production-grade AI agent that helps Mal's internal compliance team assess whether a proposed financial product or transaction is Sharia-compliant.

**What to build:**

1. A RAG pipeline that ingests a small corpus of Sharia finance documents (you may use publicly available AAOIFI standards excerpts, or synthesize 3–5 short documents yourself). Chunk, embed, and store them in a vector store of your choice (e.g., Chroma, Pinecone free tier, pgvector).
2. An AI agent (Langgraph) that accepts a plain-English query like _"Can Mal offer a fixed-return savings account?"_ and returns a structured compliance verdict: `COMPLIANT`, `NON_COMPLIANT`, or `NEEDS_REVIEW`, with a cited reasoning summary.
3. A REST API (FastAPI or equivalent) with at least two endpoints: `POST /assess` to submit a query and `GET /health`. The API must return structured JSON responses.
4. Basic observability: log each request with a trace ID, the retrieved chunks used, the LLM prompt sent, and the verdict returned. Logs can write to stdout/file — no external APM required.
5. A deployed instance accessible via public URL (Railway, Render, Fly.io, or similar free tier).

**Constraints:**

- Use Python
- Do not use OpenAI's Assistants API or any black-box agent abstraction — the agent logic must be visible in your code
- Include a `.env.example` and never commit real API keys
- README must include: setup instructions, architecture diagram (ASCII or image), example `curl` request, and known limitations

**Deliverables:**

GitHub Repository with README

**Make sure anyone can open your link without signing in.** Test in an incognito window first.

Deployed API URL

**Make sure anyone can open your link without signing in.** Test in an incognito window first.