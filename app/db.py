"""Database utilities and ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = "sqlite:///data/app.db"


class Base(DeclarativeBase):
    """Declarative base class."""


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, future=True, echo=False
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class History(Base):
    """Single conversation history table."""

    __tablename__ = "history"

    id: int = Column(Integer, primary_key=True, index=True)
    role: str = Column(String, nullable=False)
    content: str = Column(String, nullable=False)
    intent: str = Column(String, nullable=True)
    created_at: datetime = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )


def init_db() -> None:
    """Create database tables if they do not exist."""

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for dependency injection."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

