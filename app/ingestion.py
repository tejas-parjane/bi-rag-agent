"""Document ingestion: text -> chunks + metadata -> PyTorch embeddings -> pgvector-ready rows."""
from datetime import date
from sqlalchemy.orm import Session
from .models import Document, DocumentChunk
from .vector_store import chunk_text, embed_texts, save_embedding, semantic_enabled


def ingest_text_document(db: Session, title: str, source_uri: str, content: str) -> tuple[Document, int, bool]:
    chunks = chunk_text(content)
    document = Document(title=title, source_uri=source_uri, published_on=date.today())
    db.add(document)
    db.flush()
    semantic = semantic_enabled()
    vectors = embed_texts(chunks) if semantic else [None] * len(chunks)
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        db.add(DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk,
            embedding=save_embedding(vector, db) if vector is not None else None,
        ))
    db.commit()
    return document, len(chunks), semantic
