"""LangChain orchestration for the BI RAG evidence path.

The pipeline deliberately separates governed SQL computation from LLM synthesis:
the model receives evidence, but cannot execute SQL or choose an unapproved query.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import Field
from sqlalchemy.orm import Session

from .models import DocumentChunk
from .services import QueryService, metric, retrieve


class BusinessReportRetriever(BaseRetriever):
    """Custom LangChain retriever backed by ingested business-report chunks."""

    db: Any = Field(exclude=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[LCDocument]:
        chunks = retrieve(self.db, query)
        return [
            LCDocument(
                page_content=chunk["content"],
                metadata={"source": "business_document", "chunk_id": chunk["id"], "document_id": chunk["document_id"]},
            )
            for chunk in chunks
        ]


def build_evidence_chain(db: Session):
    """Build an LCEL graph that retrieves documents, metrics, and governed SQL data in parallel."""
    retriever = BusinessReportRetriever(db=db)
    return RunnableParallel(
        question=RunnableLambda(lambda question: question),
        documents=retriever,
        revenue_data=RunnableLambda(lambda _: QueryService().revenue_comparison(db)),
        metric_definition=RunnableLambda(lambda _: metric(db, "Revenue")),
    )


def retrieve_evidence(db: Session, question: str) -> dict:
    """Invoke the LangChain graph and convert its output to API-safe primitives."""
    result = build_evidence_chain(db).invoke(question)
    return {
        "data": result["revenue_data"],
        "definition": result["metric_definition"],
        "docs": [
            {"id": document.metadata["chunk_id"], "content": document.page_content, "document_id": document.metadata["document_id"]}
            for document in result["documents"]
        ],
    }


def synthesize_with_langchain(evidence: dict, fallback: str) -> str:
    """Optional ChatOpenAI synthesis with an evidence-only constraint.

    Offline/deterministic output remains the default when no API key is configured.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return fallback
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a BI analyst. Answer only from supplied evidence. Do not invent causes, numbers, or sources. State reported explanations as reported, not proven causality."),
            ("human", "Question: {question}\n\nStructured data: {data}\nMetric definition: {definition}\nBusiness-report evidence: {docs}\n\nWrite a concise grounded answer."),
        ])
        chain = prompt | ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
        response = chain.invoke({"question": "Why did revenue decline in Q3?", "data": evidence["data"], "definition": evidence["definition"].definition, "docs": [d["content"] for d in evidence["docs"]]})
        return str(response.content)
    except Exception:
        # The evidence-backed answer remains available if an optional provider is unavailable.
        return fallback
