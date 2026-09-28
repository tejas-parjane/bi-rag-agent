from datetime import date
from sqlalchemy import Date, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector
    EMBEDDING_TYPE = Vector(384)
except ImportError:  # Allows the zero-dependency SQLite demo to start before optional extras install.
    EMBEDDING_TYPE = Text


class Base(DeclarativeBase):
    pass


class MetricDefinition(Base):
    __tablename__ = "metric_definitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    definition: Mapped[str] = mapped_column(Text)
    grain: Mapped[str] = mapped_column(String(120))
    source_table: Mapped[str] = mapped_column(String(120))


class RevenueFact(Base):
    __tablename__ = "revenue_facts"
    id: Mapped[int] = mapped_column(primary_key=True)
    period: Mapped[str] = mapped_column(String(8), index=True)
    segment: Mapped[str] = mapped_column(String(80), index=True)
    revenue: Mapped[float] = mapped_column(Float)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(240))
    source_uri: Mapped[str] = mapped_column(String(500))
    published_on: Mapped[date] = mapped_column(Date)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # `vector(384)` in PostgreSQL when pgvector is installed; JSON text in demo fallback mode.
    embedding: Mapped[object | None] = mapped_column(EMBEDDING_TYPE, nullable=True)
