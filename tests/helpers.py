from __future__ import annotations

from pathlib import Path

from briefgpt_arxiv.db import SessionLocal, engine
from briefgpt_arxiv.models import Base


FIXTURES = Path(__file__).parent / "fixtures"


def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
