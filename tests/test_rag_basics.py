from app.services import evaluate
from app.vector_store import chunk_text


def test_chunking_preserves_nonempty_document():
    chunks = chunk_text("Revenue declined because Enterprise renewals moved into Q4. " * 50, chunk_size=120, overlap=20)
    assert len(chunks) > 2
    assert all(chunks)


def test_evaluation_requires_citations_and_causal_caveat():
    assert evaluate("Reported factors are not proof of causality.", [{"kind": "sql"}])["grounded"]
    assert not evaluate("An unsupported statement", [])["grounded"]
