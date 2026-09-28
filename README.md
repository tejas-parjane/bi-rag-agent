# Business Intelligence RAG Agent

A production-shaped starter that answers business questions using three evidence lanes:

1. **Business documents** – chunked reports with source citations.
2. **SQL data** – governed revenue facts, queried through approved templates.
3. **KPI definitions** – a semantic layer that explains the metric and grain.

The demo answers: **“Why did revenue decline in Q3?”** by comparing Q2 and Q3, identifying contributing segments, retrieving an analyst report, and returning evidence with SQL and document citations.

## Architecture

```text
Question -> intent classifier -> metric resolver
         -> SQL evidence + document/KPI retrieval
         -> grounded reasoning -> answer, citations, evaluation trace
```

The database has `documents`, `document_chunks`, `metric_definitions`, and `revenue_facts` tables. PostgreSQL is bundled with pgvector so document embeddings can be added without redesigning the schema. The starter uses a real **LangChain LCEL pipeline** (`RunnableParallel` + `RunnableLambda`) to assemble evidence, use a `BaseRetriever` implementation for report chunks, and optionally invoke `ChatOpenAI` for constrained synthesis. The deterministic fallback lets the project demo without a paid API key.

### Stack signal for a job description

- **LangChain**: LCEL orchestration, custom retriever, structured evidence context, optional `ChatOpenAI` synthesis.
- **FastAPI + Pydantic**: typed API contract and service boundary.
- **SQLAlchemy + PostgreSQL/pgvector**: governed SQL and a vector-ready document schema.
- **Streamlit + Docker**: usable UI and reproducible local deployment.

LlamaIndex is intentionally not included: it overlaps with LangChain for this scope. A focused LangChain implementation is easier to explain in an interview than using both frameworks without a clear responsibility split.

## Run it

1. Copy `.env.example` to `.env` and set your existing `OPENAI_API_KEY` locally if you want LLM synthesis. The service remains useful without it.
2. Start the stack:

   ```bash
   docker compose up --build
   ```

3. Open Streamlit at http://localhost:8501, or FastAPI docs at http://localhost:8000/docs.

For a local API-only demo, run `pip install -r requirements.txt` then `uvicorn app.main:app --reload`. It uses SQLite automatically when `DATABASE_URL` is absent.

## Deploy to Render

This repository includes [`render.yaml`](render.yaml), which creates two services: the FastAPI evidence API and the Streamlit UI. In Render, select **New +** → **Blueprint**, connect this repository, and approve the detected `render.yaml`. The hosted version works without `OPENAI_API_KEY`; leave that variable blank for a no-cost, deterministic portfolio demo. The UI will be available at the `bi-rag-agent-ui` service URL after deployment.

## Safety and governance

- The API never executes model-generated SQL. `QueryService` selects reviewed query templates from structured intent.
- Every response carries query, KPI, and document evidence.
- The LLM prompt is evidence-only and explicitly forbids invented drivers.
- `/evaluate` checks citation coverage and whether claims remain grounded in supplied evidence.

## Next production increments

- Replace lexical retrieval with OpenAI embeddings + pgvector cosine search.
- Connect S3 ingestion and your warehouse read replica.
- Add role-based data policies, query cost limits, audit persistence, and human review for new metric definitions.
