"""Embedding, pgvector search, and document chunking primitives.

The semantic path is intentionally local-first: SentenceTransformers uses PyTorch,
so a portfolio demo can demonstrate embeddings without an external API bill.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import DocumentChunk

EMBEDDING_DIMENSIONS = 384


def semantic_enabled() -> bool:
    return os.getenv("SEMANTIC_RETRIEVAL_ENABLED", "false").lower() == "true"


@lru_cache(maxsize=1)
def _embedder():
    """Lazy-load the local PyTorch model only when semantic retrieval is requested."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Chunk with LangChain's splitter while preserving a testable empty-input behavior."""
    if not text.strip():
        return []
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def _is_postgres(db: Session) -> bool:
    return db.bind is not None and db.bind.dialect.name == "postgresql"


def save_embedding(vector: list[float], db: Session) -> object:
    """Keep SQLite demo compatibility while using native vector values in PostgreSQL."""
    return vector if _is_postgres(db) else json.dumps(vector)


def semantic_search(db: Session, question: str, limit: int = 3) -> list[DocumentChunk]:
    """Run cosine-distance search only against native pgvector columns."""
    if not semantic_enabled() or not _is_postgres(db):
        return []
    query_vector = embed_texts([question])[0]
    try:
        distance = DocumentChunk.embedding.cosine_distance(query_vector)  # type: ignore[attr-defined]
        return list(db.scalars(select(DocumentChunk).where(DocumentChunk.embedding.is_not(None)).order_by(distance).limit(limit)))
    except Exception:
        # Degrade safely during first-run migration/model availability issues; lexical retrieval remains available.
        return []
