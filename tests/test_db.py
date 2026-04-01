from __future__ import annotations

from sqlalchemy import inspect


def test_init_db_creates_only_current_runtime_tables(db_engine) -> None:
    inspector = inspect(db_engine)
    table_names = set(inspector.get_table_names())

    assert {
        "papers",
        "artifacts",
        "paper_references",
        "citation_blocks",
        "citation_mentions",
        "ingestion_jobs",
    }.issubset(table_names)
    assert not {"paper_versions", "citation_extractions", "references"} & table_names
