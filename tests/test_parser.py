from __future__ import annotations

import io
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.models import Artifact, CitationBlock, IngestionJob, Paper, PaperReference
from briefgpt_arxiv.services.parser import (
    LLMParserRepairClient,
    ParseRepairResult,
    ParserRepairClient,
    ParserService,
)
from tests.helpers import FIXTURES, get_session, reset_database


class FakeRepairClient(ParserRepairClient):
    def repair(self, raw_text: str, candidate_keys: list[str]) -> ParseRepairResult:
        repaired_keys = list(candidate_keys)
        if "\\mycite{BIBREF2}" in raw_text and "BIBREF2" not in repaired_keys:
            repaired_keys.append("BIBREF2")
        return ParseRepairResult(
            raw_citation_keys=repaired_keys,
            cleaned_text=raw_text.replace("\\mycite{BIBREF2}", "BIBREF2"),
            used_repair="\\mycite{BIBREF2}" in raw_text,
        )


class ParserServiceTests(TestCase):
    def setUp(self) -> None:
        reset_database()
        self.original_openrouter_api_key = settings.openrouter_api_key
        self.original_gemini_api_key = settings.gemini_api_key
        settings.openrouter_api_key = None
        settings.gemini_api_key = None
        self.session = get_session()
        self.paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A")
        self.session.add(self.paper)
        self.session.flush()
        self.paper.parse_status = "pending"
        self.session.commit()
        self.tempdir = TemporaryDirectory()

    def tearDown(self) -> None:
        settings.openrouter_api_key = self.original_openrouter_api_key
        settings.gemini_api_key = self.original_gemini_api_key
        self.session.close()
        self.tempdir.cleanup()

    def test_parse_structured_doc_creates_sections_references_and_blocks(self) -> None:
        structured_path = Path(self.tempdir.name) / "structured.json"
        structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="structured_parse", uri=str(structured_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session)
        result = service.parse_paper(self.paper.id)

        self.assertEqual((3, 3, 2), result.as_tuple())
        self.assertEqual(3, self.session.query(PaperReference).count())
        self.assertEqual(3, self.session.query(CitationBlock).count())
        self.assertEqual(2, self.session.query(CitationBlock).filter(CitationBlock.has_citations.is_(True)).count())
        job = self.session.query(IngestionJob).one()
        self.assertEqual("completed", job.status)
        self.assertEqual(1, job.attempt_count)
        self.assertIsNotNone(job.finished_at)

    def test_parse_result_includes_parse_contract_details(self) -> None:
        structured_path = Path(self.tempdir.name) / "structured.json"
        structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="structured_parse", uri=str(structured_path))
        self.session.add(artifact)
        self.session.commit()

        result = ParserService(self.session).parse_paper(self.paper.id)

        self.assertEqual(self.paper.id, result.paper_id)
        self.assertEqual("2603.15726", result.arxiv_id)
        self.assertEqual("v1", result.version)
        self.assertEqual("structured_parse", result.source_artifact_type)
        self.assertEqual(str(structured_path), result.source_artifact_uri)
        self.assertFalse(result.cleanup_performed)
        self.assertEqual("parsed", result.status)

    def test_parse_skip_reuses_existing_outputs_and_records_skipped_job(self) -> None:
        structured_path = Path(self.tempdir.name) / "structured.json"
        structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="structured_parse", uri=str(structured_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session)
        service.parse_paper(self.paper.id)
        skipped = service.parse_paper(self.paper.id, rerun=False)

        self.assertEqual((3, 3, 2), skipped.as_tuple())
        self.assertEqual("skipped", skipped.status)
        self.assertFalse(skipped.cleanup_performed)
        jobs = self.session.query(IngestionJob).order_by(IngestionJob.id).all()
        self.assertEqual(["completed", "skipped"], [job.status for job in jobs])
        self.assertEqual([1, 2], [job.attempt_count for job in jobs])

    def test_parse_structured_doc_requires_explicit_latex_parse_object(self) -> None:
        structured_path = Path(self.tempdir.name) / "structured.json"
        structured_path.write_text('{"body_text": [], "bib_entries": {}}')
        artifact = Artifact(paper_id=self.paper.id, artifact_type="structured_parse", uri=str(structured_path))
        self.session.add(artifact)
        self.session.commit()

        with self.assertRaisesRegex(ValueError, "latex_parse"):
            ParserService(self.session).parse_paper(self.paper.id)

    def test_parse_source_uses_repair_client_for_nonstandard_macros(self) -> None:
        source_path = Path(self.tempdir.name) / "source.tex"
        source_path.write_text((FIXTURES / "latex_sample.tex").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="source", uri=str(source_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session, repair_client=FakeRepairClient())
        _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(3, refs)
        self.assertEqual(2, blocks)
        repaired_block = (
            self.session.query(CitationBlock)
            .filter(CitationBlock.repair_used.is_(True))
            .one()
        )
        self.assertIn("BIBREF2", repaired_block.raw_citation_keys)

    def test_parse_source_skips_repair_for_paragraphs_without_citation_signal(self) -> None:
        source_path = Path(self.tempdir.name) / "source.tex"
        source_path.write_text(
            r"""
\documentclass{article}
\begin{document}
\section{Intro}
This paragraph has no references at all.

We compare against prior work \cite{alpha2024}.

\begin{thebibliography}{9}
\bibitem{alpha2024} Alpha Systems for Planning. 2024.
\end{thebibliography}
\end{document}
"""
        )
        artifact = Artifact(paper_id=self.paper.id, artifact_type="source", uri=str(source_path))
        self.session.add(artifact)
        self.session.commit()

        repair_client = Mock(spec=ParserRepairClient)
        repair_client.repair.return_value = ParseRepairResult(
            raw_citation_keys=["alpha2024"],
            cleaned_text="We compare against prior work alpha2024.",
            used_repair=True,
        )

        service = ParserService(self.session, repair_client=repair_client)
        _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(1, refs)
        self.assertEqual(1, blocks)
        self.assertEqual(1, repair_client.repair.call_count)
        self.assertIn(r"\cite{alpha2024}", repair_client.repair.call_args.args[0])

    def test_llm_repair_client_derives_used_repair_without_model_flag(self) -> None:
        settings.openrouter_api_key = "test-key"
        client = LLMParserRepairClient()

        class StubLLMClient:
            def generate_json(self, **_kwargs) -> dict:
                return {
                    "raw_citation_keys": ["BIBREF0", "BIBREF2"],
                    "cleaned_text": "Text with BIBREF2",
                }

        client.client = StubLLMClient()
        repaired = client.repair("Text with \\mycite{BIBREF2}", ["BIBREF0"])

        self.assertEqual(["BIBREF0", "BIBREF2"], repaired.raw_citation_keys)
        self.assertEqual("Text with BIBREF2", repaired.cleaned_text)
        self.assertTrue(repaired.used_repair)

    def test_parse_source_tar_extracts_references_from_external_bib(self) -> None:
        source_path = Path(self.tempdir.name) / "source.tar"
        tex_text = r"""
\documentclass{article}
\begin{document}
\section{Intro}
We follow prior work \cite{alpha2024,beta2023}.
\bibliography{main}
\end{document}
"""
        bib_text = r"""
@article{alpha2024,
  title={Alpha Systems for Planning},
  author={Ada Lovelace and Grace Hopper},
  journal={Journal of Agents},
  year={2024}
}

@inproceedings{beta2023,
  title={Beta Benchmarks for Search},
  author={Alan Turing},
  booktitle={Proceedings of SearchConf},
  year={2023}
}
"""
        with tarfile.open(source_path, "w") as archive:
            for name, payload in {"main.tex": tex_text, "main.bib": bib_text}.items():
                encoded = payload.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        artifact = Artifact(paper_id=self.paper.id, artifact_type="source", uri=str(source_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session, repair_client=FakeRepairClient())
        _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(2, refs)
        self.assertEqual(1, blocks)
        reference_rows = self.session.query(PaperReference).order_by(PaperReference.local_ref_id).all()
        self.assertEqual("alpha2024", reference_rows[0].local_ref_id)
        self.assertEqual("Alpha Systems for Planning", reference_rows[0].title)
        self.assertEqual(2024, reference_rows[0].year)
        self.assertEqual("Journal of Agents", reference_rows[0].venue)
        self.assertEqual("beta2023", reference_rows[1].local_ref_id)
        self.assertEqual("Beta Benchmarks for Search", reference_rows[1].title)
        self.assertEqual(2023, reference_rows[1].year)

    def test_pdf_text_fallback_parses_reference_blocks(self) -> None:
        pdf_text_path = Path(self.tempdir.name) / "paper.txt"
        pdf_text_path.write_text((FIXTURES / "pdf_fallback.txt").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="pdf_text", uri=str(pdf_text_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session)
        _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(2, refs)
        self.assertEqual(2, blocks)

    def test_pdf_text_fallback_reconstructs_paragraphs_and_citation_ranges(self) -> None:
        pdf_text_path = Path(self.tempdir.name) / "paper.txt"
        pdf_text_path.write_text((FIXTURES / "pdf_complex_sample.txt").read_text())
        artifact = Artifact(paper_id=self.paper.id, artifact_type="pdf_text", uri=str(pdf_text_path))
        self.session.add(artifact)
        self.session.commit()

        service = ParserService(self.session)
        _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(5, refs)
        self.assertEqual(2, blocks)
        block_rows = self.session.query(CitationBlock).order_by(CitationBlock.id).all()
        citation_rows = [row for row in block_rows if row.has_citations]
        self.assertEqual(["REF1", "REF2", "REF3"], citation_rows[0].raw_citation_keys)
        self.assertEqual(["REF4", "REF5"], citation_rows[1].raw_citation_keys)
        self.assertIn("robust planning behavior in agents.", citation_rows[0].raw_text)
        self.assertIn("API Match and Correctness.", citation_rows[1].raw_text)
        section_titles = [block.section_title for block in block_rows]
        self.assertIn("1. Introduction", section_titles)
        self.assertIn("2. Metrics", section_titles)

        reference_rows = self.session.query(PaperReference).order_by(PaperReference.id).all()
        self.assertEqual("Alpha systems for planning", reference_rows[0].title)
        self.assertEqual("Epsilon study of correctness", reference_rows[-1].title)

    def test_pdf_artifact_is_directly_parsed_and_persisted_as_pdf_text(self) -> None:
        pdf_path = Path(self.tempdir.name) / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        artifact = Artifact(paper_id=self.paper.id, artifact_type="pdf", uri=str(pdf_path))
        self.session.add(artifact)
        self.session.commit()

        class FakePage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class FakePdfReader:
            def __init__(self, _path: str) -> None:
                self.pages = [FakePage((FIXTURES / "pdf_fallback.txt").read_text())]

        with patch("pypdf.PdfReader", FakePdfReader):
            service = ParserService(self.session)
            _, refs, blocks = service.parse_paper(self.paper.id).as_tuple()

        self.assertEqual(2, refs)
        self.assertEqual(2, blocks)
        pdf_text_artifact = (
            self.session.query(Artifact)
            .filter(Artifact.paper_id == self.paper.id, Artifact.artifact_type == "pdf_text")
            .one()
        )
        self.assertTrue(Path(pdf_text_artifact.uri).exists())
