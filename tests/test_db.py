from __future__ import annotations

from unittest import TestCase

from sqlalchemy import inspect

from briefgpt_arxiv.db import engine, init_db
from tests.helpers import reset_database


class DatabaseSchemaTests(TestCase):
    def setUp(self) -> None:
        reset_database()

    def test_init_db_creates_only_current_runtime_tables(self) -> None:
        init_db()

        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())

        self.assertTrue(
            {
                "papers",
                "artifacts",
                "paper_references",
                "citation_blocks",
                "citation_mentions",
                "ingestion_jobs",
            }.issubset(table_names)
        )
        self.assertFalse({"paper_versions", "citation_extractions", "references"} & table_names)
