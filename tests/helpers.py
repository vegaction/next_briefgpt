from __future__ import annotations

from pathlib import Path

import briefgpt_arxiv.db as db
from briefgpt_arxiv.models import Base


FIXTURES = Path(__file__).parent / "fixtures"
TEST_DB_PATH = Path(".tmp_test_briefgpt.db")


def _reset_test_engine() -> None:
    db.engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    db.engine = db.create_engine(
        f"sqlite:///{TEST_DB_PATH}",
        connect_args={"check_same_thread": False},
    )
    db.SessionLocal.configure(bind=db.engine)


def reset_database() -> None:
    _reset_test_engine()
    Base.metadata.drop_all(bind=db.engine)
    Base.metadata.create_all(bind=db.engine)
    db.engine.dispose()


def get_session():
    return db.SessionLocal()
