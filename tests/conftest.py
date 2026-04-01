from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from briefgpt_arxiv.models import Base


@pytest.fixture()
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine) -> Generator[Session, None, None]:
    session = sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False,
    )()
    yield session
    session.close()


@pytest.fixture()
def tempdir() -> Generator[TemporaryDirectory, None, None]:
    td = TemporaryDirectory()
    yield td
    td.cleanup()


_SETTINGS_MODULES = [
    "briefgpt_arxiv.config",
    "briefgpt_arxiv.db",
    "briefgpt_arxiv.llm_client",
    "briefgpt_arxiv.services.crawler",
    "briefgpt_arxiv.services.parser",
    "briefgpt_arxiv.services.extractor",
]


@contextmanager
def override_settings(**kwargs):
    """Temporarily override settings fields across all modules that import it."""
    import sys
    from dataclasses import fields as dataclass_fields

    import briefgpt_arxiv.config as cfg

    original = cfg.settings
    overrides = {f.name: kwargs.get(f.name, getattr(original, f.name)) for f in dataclass_fields(original)}
    new_settings = cfg.Settings(**overrides)

    # Replace the settings reference in every module that imported it.
    patched: list[tuple] = []
    for mod_name in _SETTINGS_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "settings"):
            patched.append((mod, getattr(mod, "settings")))
            mod.settings = new_settings

    try:
        yield new_settings
    finally:
        for mod, orig_ref in patched:
            mod.settings = orig_ref


@pytest.fixture()
def no_llm_keys():
    with override_settings(openrouter_api_key=None, gemini_api_key=None):
        yield


FIXTURES = Path(__file__).parent / "fixtures"
