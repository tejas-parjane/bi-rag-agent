import os
from datetime import date
from fastapi import Depends, FastAPI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from .models import Base, Document, DocumentChunk, MetricDefinition, RevenueFact
from .schemas import AskRequest, AskResponse, Citation
from .langchain_pipeline import retrieve_evidence, synthesize_with_langchain
from .services import QueryService, classify, evaluate, grounded_answer

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bi_rag.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
app = FastAPI(title="Business Intelligence RAG Agent", version="0.1.0")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed(db: Session):
    if db.scalar(select(MetricDefinition.id).limit(1)):
        return
    db.add(MetricDefinition(name="Revenue", definition="Recognized subscription revenue, excluding tax and credits.", grain="Quarter x customer segment", source_table="revenue_facts"))
    db.add_all([
        RevenueFact(period="Q2", segment="Enterprise", revenue=700000), RevenueFact(period="Q3", segment="Enterprise", revenue=500000),
        RevenueFact(period="Q2", segment="Mid-market", revenue=300000), RevenueFact(period="Q3", segment="Mid-market", revenue=280000),
        RevenueFact(period="Q2", segment="SMB", revenue=200000), RevenueFact(period="Q3", segment="SMB", revenue=160000),
    ])
    report = Document(title="Q3 Commercial Performance Review", source_uri="s3://bi-rag-demo/reports/q3-commercial-review.pdf", published_on=date(2026, 10, 5))
    db.add(report); db.flush()
    db.add(DocumentChunk(document_id=report.id, chunk_index=0, content="Enterprise revenue fell as several large renewals moved into Q4 after procurement delays. SMB new-logo conversion softened after paid-search efficiency declined. Mid-market remained broadly stable."))
    db.commit()


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)):
    intent = classify(request.question)
    # LangChain LCEL graph retrieves the three evidence lanes in parallel.
    evidence = retrieve_evidence(db, request.question)
    data, docs, definition = evidence["data"], evidence["docs"], evidence["definition"]
    citations = [
        Citation(kind="sql", title="Governed revenue comparison", locator="revenue_facts: Q2 vs Q3 by segment", excerpt="Reviewed template; no generated SQL executed."),
        Citation(kind="metric", title=definition.name, locator=f"{definition.source_table} | {definition.grain}", excerpt=definition.definition),
    ]
    for doc in docs:
        citations.append(Citation(kind="document", title="Q3 Commercial Performance Review", locator=f"document_chunks:{doc['id']}", excerpt=doc["content"]))
    fallback = grounded_answer(data, docs)
    answer = synthesize_with_langchain(evidence, fallback)
    return AskResponse(answer=answer, intent=intent.name, citations=citations, sql=QueryService.QOQ_REVENUE_SQL, data=data, evaluation=evaluate(answer, [c.model_dump() for c in citations]))


@app.post("/evaluate")
def evaluation(request: AskRequest, db: Session = Depends(get_db)):
    return ask(request, db).evaluation
