from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


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

