from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


class IngestRequest(BaseModel):
    title: str = Field(min_length=3, max_length=240)
    source_uri: str = Field(min_length=3, max_length=500, examples=["s3://company-reports/q3-review.md"])
    content: str = Field(min_length=50, max_length=100_000)


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    semantic_embeddings_created: bool


class Citation(BaseModel):
    kind: Literal["sql", "document", "metric"]
    title: str
    locator: str
    excerpt: str | None = None


class AskResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation]
    sql: str
    data: dict
    evaluation: dict
