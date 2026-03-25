from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.gemini import GeminiClient
from briefgpt_arxiv.models import (
    Artifact,
    CitationBlock,
    CitationMention,
    Paper,
    PaperReference,
)
from briefgpt_arxiv.prompts import (
    PARSER_REPAIR_JSON_SCHEMA,
    PARSER_REPAIR_SYSTEM_TEMPLATE,
    PARSER_REPAIR_USER_TEMPLATE,
    PROMPT_TEMPLATE_ENV,
)
from briefgpt_arxiv.services.contracts import ParseInputSelection, ParseRunResult
from briefgpt_arxiv.services.jobs import JobTracker
from briefgpt_arxiv.utils import normalize_whitespace, utcnow_naive


@dataclass(slots=True)
class ReferencePayload:
    local_ref_id: str
    raw_text: str
    title: str | None
    authors: list[dict] | None
    year: int | None
    venue: str | None


@dataclass(slots=True)
class SectionPayload:
    section_title: str | None
    section_path: str | None
    chunk_index: int
    raw_text: str
    raw_citation_keys: list[str]
    repair_used: bool


@dataclass(slots=True)
class ParsedDocument:
    references: list[ReferencePayload]
    sections: list[SectionPayload]


@dataclass(slots=True)
class SourceBundle:
    tex_text: str
    bib_texts: list[str]
    bbl_texts: list[str]


@dataclass(slots=True)
class ParseRepairResult:
    raw_citation_keys: list[str]
    cleaned_text: str
    used_repair: bool


class ParserRepairClient:
    def repair(self, raw_text: str, candidate_keys: list[str]) -> ParseRepairResult:
        return ParseRepairResult(raw_citation_keys=candidate_keys, cleaned_text=raw_text, used_repair=False)


class GeminiParserRepairClient(ParserRepairClient):
    def __init__(self) -> None:
        self.client = GeminiClient()

    def repair(self, raw_text: str, candidate_keys: list[str]) -> ParseRepairResult:
        system_instruction = PROMPT_TEMPLATE_ENV.from_string(PARSER_REPAIR_SYSTEM_TEMPLATE).render()
        user_text = PROMPT_TEMPLATE_ENV.from_string(PARSER_REPAIR_USER_TEMPLATE).render(
            raw_text=raw_text,
            candidate_keys=candidate_keys,
        )
        payload = self.client.generate_json(
            system_instruction=system_instruction,
            user_text=user_text,
            response_json_schema=PARSER_REPAIR_JSON_SCHEMA,
        )
        return ParseRepairResult(
            raw_citation_keys=payload.get("raw_citation_keys", candidate_keys),
            cleaned_text=payload.get("cleaned_text", raw_text),
            used_repair=bool(payload.get("used_repair", False)),
        )


class ParserService:
    def __init__(self, session: Session, repair_client: ParserRepairClient | None = None):
        self.session = session
        self.job_tracker = JobTracker(session)
        if repair_client is not None:
            self.repair_client = repair_client
        else:
            self.repair_client = GeminiParserRepairClient() if settings.gemini_api_key else ParserRepairClient()

    def parse_paper(self, paper_id: int, *, rerun: bool = True) -> ParseRunResult:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")
        selection = self._select_parse_input(paper)
        if not rerun and paper.parse_status == "parsed":
            result = self._build_parse_result(
                paper=paper,
                selection=selection,
                sections_created=self._count_sections(paper_id),
                references_created=self._count_references(paper_id),
                citation_blocks_created=self._count_blocks(paper_id),
                cleanup_performed=False,
                status="skipped",
            )
            job = self.job_tracker.start(job_type="parse", target_id=paper_id)
            self.job_tracker.finish(job, status="skipped", error_message="Reused existing parse outputs.")
            self.session.commit()
            return result

        job = self.job_tracker.start(job_type="parse", target_id=paper_id)
        try:
            _, parsed = self._parse_from_best_artifact(paper)
            cleanup_performed = self._has_parse_outputs(paper_id)
            if cleanup_performed:
                self.clear_parse_outputs(paper_id)
            sections_created, citation_blocks_created = self._persist_blocks(paper, parsed.sections)
            references_created = self._persist_references(paper, parsed.references)
            paper.parse_status = "parsed"
            paper.parsed_at = utcnow_naive()
            paper.ingest_status = "parsed"
            self.job_tracker.finish(job)
            self.session.commit()
            return self._build_parse_result(
                paper=paper,
                selection=selection,
                sections_created=sections_created,
                references_created=references_created,
                citation_blocks_created=citation_blocks_created,
                cleanup_performed=cleanup_performed,
                status="parsed",
            )
        except Exception as exc:
            self.session.rollback()
            self.job_tracker.record_failure(job_type="parse", target_id=paper_id, error_message=str(exc))
            self.session.commit()
            raise

    def clear_parse_outputs(self, paper_id: int) -> None:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")
        self._clear_existing_parse_outputs(paper_id)
        paper.parse_status = "pending"
        paper.parsed_at = None
        if paper.ingest_status in {"parsed", "ready"}:
            paper.ingest_status = "fetched"

    def _select_parse_input(self, paper: Paper) -> ParseInputSelection:
        artifacts = {artifact.artifact_type: artifact for artifact in paper.artifacts}
        if "structured_parse" in artifacts:
            artifact = artifacts["structured_parse"]
            return ParseInputSelection(artifact_type="structured_parse", artifact_uri=artifact.uri)
        if "source" in artifacts:
            artifact = artifacts["source"]
            return ParseInputSelection(artifact_type="source", artifact_uri=artifact.uri)
        if "pdf_text" in artifacts:
            artifact = artifacts["pdf_text"]
            return ParseInputSelection(artifact_type="pdf_text", artifact_uri=artifact.uri)
        if "pdf" in artifacts:
            artifact = artifacts["pdf"]
            return ParseInputSelection(artifact_type="pdf", artifact_uri=artifact.uri)
        raise ValueError("No parseable artifact found.")

    def _parse_from_best_artifact(self, paper: Paper) -> tuple[ParseInputSelection, ParsedDocument]:
        selection = self._select_parse_input(paper)
        path = Path(selection.artifact_uri)
        if selection.artifact_type == "structured_parse":
            return selection, self._parse_doc2json(path)
        if selection.artifact_type == "source":
            if path.suffix == ".json":
                return selection, self._parse_doc2json(path)
            return selection, self._parse_source(path)
        if selection.artifact_type == "pdf_text":
            return selection, self._parse_pdf_text(path)
        if selection.artifact_type == "pdf":
            if path.suffix in {".txt", ".text"}:
                return selection, self._parse_pdf_text(path)
            return selection, self._parse_pdf(paper, path)
        raise ValueError(f"Unsupported parse input {selection.artifact_type}")

    def _parse_doc2json(self, path: Path) -> ParsedDocument:
        with path.open() as handle:
            payload = json.load(handle)
        latex_parse = payload.get("latex_parse", payload)
        references = []
        for local_ref_id, entry in latex_parse.get("bib_entries", {}).items():
            raw_text = entry.get("raw_text") or entry.get("title") or local_ref_id
            references.append(
                ReferencePayload(
                    local_ref_id=local_ref_id,
                    raw_text=normalize_whitespace(raw_text),
                    title=entry.get("title"),
                    authors=entry.get("authors"),
                    year=entry.get("year"),
                    venue=entry.get("venue"),
                )
            )
        sections = []
        body_items = latex_parse.get("body_text", [])
        for index, item in enumerate(body_items):
            raw_text = normalize_whitespace(item.get("text", ""))
            if not raw_text:
                continue
            cite_spans = item.get("cite_spans", [])
            raw_keys = []
            for span in cite_spans:
                ref_id = span.get("ref_id")
                if ref_id:
                    raw_keys.append(ref_id)
            raw_keys = list(dict.fromkeys(raw_keys))
            repaired = self.repair_client.repair(raw_text, raw_keys)
            sections.append(
                SectionPayload(
                    section_title=item.get("section"),
                    section_path=item.get("section"),
                    chunk_index=index,
                    raw_text=normalize_whitespace(repaired.cleaned_text),
                    raw_citation_keys=repaired.raw_citation_keys,
                    repair_used=repaired.used_repair,
                )
            )
        return ParsedDocument(references=references, sections=sections)

    def _parse_source(self, path: Path) -> ParsedDocument:
        if path.suffix == ".tex":
            source_text = path.read_text()
            references = self._extract_references_from_source(source_text)
        else:
            bundle = self._read_source_bundle_from_tar(path)
            source_text = bundle.tex_text
            references = self._extract_references_from_source(
                source_text,
                bib_texts=bundle.bib_texts,
                bbl_texts=bundle.bbl_texts,
            )
        sections = self._extract_sections_from_source(source_text)
        return ParsedDocument(references=references, sections=sections)

    def _parse_pdf_text(self, path: Path) -> ParsedDocument:
        text = path.read_text()
        return self._parse_pdf_text_content(text)

    def _parse_pdf(self, paper: Paper, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for direct PDF parsing.") from exc

        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        text_path = path.with_suffix(".txt")
        text_path.write_text(text)

        artifact = self.session.scalar(
            select(Artifact).where(
                Artifact.paper_id == paper.id,
                Artifact.artifact_type == "pdf_text",
            )
        )
        if artifact is None:
            artifact = Artifact(
                paper_id=paper.id,
                artifact_type="pdf_text",
                uri=str(text_path),
                size_bytes=text_path.stat().st_size,
            )
            self.session.add(artifact)
        else:
            artifact.uri = str(text_path)
            artifact.size_bytes = text_path.stat().st_size
        return self._parse_pdf_text_content(text)

    def _parse_pdf_text_content(self, text: str) -> ParsedDocument:
        sections: list[SectionPayload] = []
        references: list[ReferencePayload] = []
        chunk_index = 0
        current_section = "Unknown"
        in_references = False
        paragraph_lines: list[str] = []
        reference_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal chunk_index
            if not paragraph_lines:
                return
            paragraph = self._join_pdf_lines(paragraph_lines)
            paragraph_lines.clear()
            citation_keys = self._extract_pdf_citation_keys(paragraph)
            if not citation_keys:
                return
            sections.append(
                SectionPayload(
                    section_title=current_section,
                    section_path=current_section,
                    chunk_index=chunk_index,
                    raw_text=normalize_whitespace(paragraph),
                    raw_citation_keys=citation_keys,
                    repair_used=False,
                )
            )
            chunk_index += 1

        def flush_reference() -> None:
            if not reference_lines:
                return
            reference_text = normalize_whitespace(self._join_pdf_lines(reference_lines))
            reference_lines.clear()
            match = re.match(r"^\[(\d+)\]\s*(.+)$", reference_text)
            if not match:
                return
            ref_num = match.group(1)
            raw_reference = match.group(2).strip()
            references.append(
                ReferencePayload(
                    local_ref_id=f"REF{ref_num}",
                    raw_text=raw_reference,
                    title=self._extract_pdf_reference_title(raw_reference),
                    authors=None,
                    year=self._extract_year(raw_reference),
                    venue=None,
                )
            )

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if in_references:
                    flush_reference()
                else:
                    flush_paragraph()
                continue
            if self._is_pdf_noise_line(line):
                continue
            if self._is_references_heading(line):
                flush_paragraph()
                flush_reference()
                in_references = True
                current_section = "References"
                continue
            if in_references:
                if self._is_pdf_section_heading(line):
                    flush_reference()
                    in_references = False
                    current_section = self._normalize_pdf_section_title(line)
                    continue
                if self._is_pdf_reference_start(line):
                    flush_reference()
                    reference_lines.append(line)
                else:
                    reference_lines.append(line)
                continue
            if self._is_pdf_section_heading(line):
                flush_paragraph()
                current_section = self._normalize_pdf_section_title(line)
                continue
            paragraph_lines.append(line)

        flush_paragraph()
        flush_reference()
        return ParsedDocument(references=references, sections=sections)

    def _read_source_bundle_from_tar(self, path: Path) -> SourceBundle:
        with tarfile.open(path) as archive:
            tex_buffers: list[str] = []
            bib_buffers: list[str] = []
            bbl_buffers: list[str] = []
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read().decode("utf-8", errors="ignore")
                if member.name.endswith(".tex"):
                    tex_buffers.append(payload)
                elif member.name.endswith(".bib"):
                    bib_buffers.append(payload)
                elif member.name.endswith(".bbl"):
                    bbl_buffers.append(payload)
            if not tex_buffers:
                raise ValueError("No .tex files found in source artifact.")
            return SourceBundle(
                tex_text="\n".join(tex_buffers),
                bib_texts=bib_buffers,
                bbl_texts=bbl_buffers,
            )

    def _extract_sections_from_source(self, source_text: str) -> list[SectionPayload]:
        source_text = re.sub(
            r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
            "",
            source_text,
            flags=re.DOTALL,
        )
        fragments = re.split(r"(\\(?:sub)*section\{[^}]+\})", source_text)
        current_section = "Unknown"
        sections: list[SectionPayload] = []
        chunk_index = 0
        for fragment in fragments:
            if fragment.startswith("\\section") or fragment.startswith("\\subsection"):
                current_section = re.sub(r"\\(?:sub)*section\{([^}]+)\}", r"\1", fragment)
                continue
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", fragment) if part.strip()]
            for paragraph in paragraphs:
                keys = self._extract_citation_keys(paragraph)
                repaired = self.repair_client.repair(paragraph, keys)
                if not repaired.raw_citation_keys and not keys:
                    if "cite" not in paragraph.lower():
                        continue
                if not repaired.raw_citation_keys:
                    continue
                cleaned = self._clean_latex_text(repaired.cleaned_text)
                sections.append(
                    SectionPayload(
                        section_title=current_section,
                        section_path=current_section,
                        chunk_index=chunk_index,
                        raw_text=normalize_whitespace(cleaned),
                        raw_citation_keys=repaired.raw_citation_keys,
                        repair_used=repaired.used_repair,
                    )
                )
                chunk_index += 1
        return sections

    def _extract_references_from_source(
        self,
        source_text: str,
        *,
        bib_texts: list[str] | None = None,
        bbl_texts: list[str] | None = None,
    ) -> list[ReferencePayload]:
        references_by_key: dict[str, ReferencePayload] = {}
        for payload in self._extract_bibitem_references(source_text):
            references_by_key.setdefault(payload.local_ref_id, payload)
        for bbl_text in bbl_texts or []:
            for payload in self._extract_bibitem_references(bbl_text):
                references_by_key.setdefault(payload.local_ref_id, payload)
        for bib_text in bib_texts or []:
            for payload in self._extract_bibtex_references(bib_text):
                references_by_key.setdefault(payload.local_ref_id, payload)
        return list(references_by_key.values())

    def _extract_bibitem_references(self, source_text: str) -> list[ReferencePayload]:
        references: list[ReferencePayload] = []
        bib_matches = re.finditer(
            r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.+?)(?=(\\bibitem(?:\[[^\]]*\])?\{)|\\end\{thebibliography\}|$)",
            source_text,
            flags=re.DOTALL,
        )
        for match in bib_matches:
            local_ref_id = match.group(1)
            raw_text = self._clean_latex_text(match.group(2))
            references.append(
                ReferencePayload(
                    local_ref_id=local_ref_id,
                    raw_text=normalize_whitespace(raw_text),
                    title=normalize_whitespace(raw_text[:240]),
                    authors=None,
                    year=self._extract_year(raw_text),
                    venue=None,
                )
            )
        return references

    def _extract_bibtex_references(self, bib_text: str) -> list[ReferencePayload]:
        references: list[ReferencePayload] = []
        cursor = 0
        while cursor < len(bib_text):
            match = re.search(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", bib_text[cursor:])
            if match is None:
                break
            entry_start = cursor + match.start()
            entry_body_start = cursor + match.end()
            entry_end = self._find_bibtex_entry_end(bib_text, entry_body_start)
            if entry_end is None:
                break
            local_ref_id = match.group(2).strip()
            fields_text = bib_text[entry_body_start : entry_end - 1]
            title = self._clean_bibtex_value(self._extract_bibtex_field(fields_text, "title"))
            author_text = self._clean_bibtex_value(self._extract_bibtex_field(fields_text, "author"))
            venue = self._clean_bibtex_value(
                self._extract_bibtex_field(fields_text, "journal")
                or self._extract_bibtex_field(fields_text, "booktitle")
                or self._extract_bibtex_field(fields_text, "howpublished")
            )
            year_text = self._clean_bibtex_value(self._extract_bibtex_field(fields_text, "year"))
            year = self._extract_year(year_text or fields_text)
            raw_parts = [part for part in [author_text, title, venue, year_text] if part]
            raw_text = normalize_whitespace(". ".join(raw_parts)) or normalize_whitespace(
                self._clean_latex_text(fields_text)
            )
            references.append(
                ReferencePayload(
                    local_ref_id=local_ref_id,
                    raw_text=raw_text,
                    title=title or normalize_whitespace(raw_text[:240]),
                    authors=self._parse_bibtex_authors(author_text),
                    year=year,
                    venue=venue,
                )
            )
            cursor = entry_end
        return references

    @staticmethod
    def _find_bibtex_entry_end(text: str, start: int) -> int | None:
        depth = 1
        cursor = start
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        return None

    def _extract_bibtex_field(self, fields_text: str, field_name: str) -> str | None:
        match = re.search(rf"(?is)\b{re.escape(field_name)}\s*=", fields_text)
        if match is None:
            return None
        cursor = match.end()
        while cursor < len(fields_text) and fields_text[cursor].isspace():
            cursor += 1
        if cursor >= len(fields_text):
            return None
        opener = fields_text[cursor]
        if opener == "{":
            return self._read_balanced_bibtex_value(fields_text, cursor, "{", "}")
        if opener == '"':
            return self._read_balanced_bibtex_value(fields_text, cursor, '"', '"')
        end = cursor
        while end < len(fields_text) and fields_text[end] not in ",\n":
            end += 1
        return fields_text[cursor:end].strip() or None

    @staticmethod
    def _read_balanced_bibtex_value(text: str, start: int, opener: str, closer: str) -> str | None:
        if opener == '"':
            cursor = start + 1
            while cursor < len(text):
                if text[cursor] == closer and text[cursor - 1] != "\\":
                    return text[start + 1 : cursor]
                cursor += 1
            return None
        depth = 1
        cursor = start + 1
        while cursor < len(text):
            char = text[cursor]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start + 1 : cursor]
            cursor += 1
        return None

    def _clean_bibtex_value(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = normalize_whitespace(self._clean_latex_text(value))
        return cleaned or None

    @staticmethod
    def _parse_bibtex_authors(author_text: str | None) -> list[dict] | None:
        if not author_text:
            return None
        authors = []
        for item in re.split(r"\s+and\s+", author_text):
            name = normalize_whitespace(item)
            if name:
                authors.append({"full_name": name})
        return authors or None

    def _extract_citation_keys(self, raw_text: str) -> list[str]:
        keys: list[str] = []
        for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]+)\}", raw_text):
            keys.extend([part.strip() for part in match.group(1).split(",") if part.strip()])
        if not keys:
            keys.extend(re.findall(r"\bBIBREF\d+\b", raw_text))
        return list(dict.fromkeys(keys))

    def _extract_pdf_citation_keys(self, raw_text: str) -> list[str]:
        keys: list[str] = []
        for content in re.findall(r"\[([^\]]+)\]", raw_text):
            if not re.fullmatch(r"\s*\d+(?:\s*[-–,]\s*\d+)*\s*", content):
                continue
            for part in re.split(r"\s*,\s*", content.strip()):
                if not part:
                    continue
                range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if start <= end and end - start <= 25:
                        keys.extend([f"REF{number}" for number in range(start, end + 1)])
                    continue
                if part.isdigit():
                    keys.append(f"REF{int(part)}")
        return list(dict.fromkeys(keys))

    def _clean_latex_text(self, raw_text: str) -> str:
        text = re.sub(r"\\cite[a-zA-Z*]*\{([^}]+)\}", lambda m: " " + m.group(1).replace(",", " ") + " ", raw_text)
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z*]+", " ", text)
        text = re.sub(r"[{}]", " ", text)
        return text

    def _join_pdf_lines(self, lines: list[str]) -> str:
        merged: list[str] = []
        for line in lines:
            normalized = normalize_whitespace(line)
            if not normalized:
                continue
            if not merged:
                merged.append(normalized)
                continue
            previous = merged[-1]
            if previous.endswith("-") and normalized[:1].islower():
                merged[-1] = previous[:-1] + normalized
            else:
                merged[-1] = previous + " " + normalized
        return normalize_whitespace(" ".join(merged))

    def _is_pdf_noise_line(self, line: str) -> bool:
        lowered = line.lower()
        if re.fullmatch(r"\d+", line):
            return True
        if lowered.startswith("arxiv:"):
            return True
        if lowered.endswith("technical report"):
            return True
        return False

    def _is_references_heading(self, line: str) -> bool:
        return normalize_whitespace(line).lower() in {"references", "bibliography"}

    def _is_pdf_section_heading(self, line: str) -> bool:
        normalized = normalize_whitespace(line)
        if self._is_references_heading(normalized):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,/&()\-]+", normalized):
            return True
        return False

    def _normalize_pdf_section_title(self, line: str) -> str:
        return normalize_whitespace(line).rstrip(".")

    def _is_pdf_reference_start(self, line: str) -> bool:
        return re.match(r"^\[\d+\]\s*", line) is not None

    def _extract_pdf_reference_title(self, raw_reference: str) -> str:
        trimmed = normalize_whitespace(raw_reference)
        sentences = re.split(r"(?<=[.!?])\s+", trimmed)
        candidate = trimmed
        if len(sentences) >= 2:
            candidate = " ".join(sentences[1:3])
        candidate = re.split(r"https?://", candidate, maxsplit=1)[0]
        candidate = re.split(
            r"(?i)\b(arxiv preprint|official announcement|official release|official launch|official repository|official hugging face model card|tech report|in the)\b",
            candidate,
            maxsplit=1,
        )[0]
        candidate = normalize_whitespace(candidate.strip(" .,:;"))
        if candidate:
            return candidate[:240]
        return trimmed[:240]

    @staticmethod
    def _extract_year(raw_text: str) -> int | None:
        candidates = [int(value) for value in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", raw_text)]
        if not candidates:
            return None
        return candidates[-1]

    def _clear_existing_parse_outputs(self, paper_id: int) -> None:
        block_ids = list(self.session.scalars(select(CitationBlock.id).where(CitationBlock.paper_id == paper_id)))
        if block_ids:
            mention_ids = list(
                self.session.scalars(select(CitationMention.id).where(CitationMention.citation_block_id.in_(block_ids)))
            )
            if mention_ids:
                self.session.execute(delete(CitationMention).where(CitationMention.id.in_(mention_ids)))
            self.session.execute(delete(CitationBlock).where(CitationBlock.id.in_(block_ids)))
        self.session.execute(delete(PaperReference).where(PaperReference.paper_id == paper_id))

    def _has_parse_outputs(self, paper_id: int) -> bool:
        return any(
            [
                self._count_sections(paper_id),
                self._count_references(paper_id),
                self._count_blocks(paper_id),
            ]
        )

    def _count_sections(self, paper_id: int) -> int:
        return len(list(self.session.scalars(select(CitationBlock.id).where(CitationBlock.paper_id == paper_id))))

    def _count_references(self, paper_id: int) -> int:
        return len(list(self.session.scalars(select(PaperReference.id).where(PaperReference.paper_id == paper_id))))

    def _count_blocks(self, paper_id: int) -> int:
        return len(
            list(
                self.session.scalars(
                    select(CitationBlock.id).where(
                        CitationBlock.paper_id == paper_id,
                        CitationBlock.has_citations.is_(True),
                    )
                )
            )
        )

    def _build_parse_result(
        self,
        *,
        paper: Paper,
        selection: ParseInputSelection,
        sections_created: int,
        references_created: int,
        citation_blocks_created: int,
        cleanup_performed: bool,
        status: str,
    ) -> ParseRunResult:
        return ParseRunResult(
            paper_id=paper.id,
            arxiv_id=paper.arxiv_id,
            source_artifact_type=selection.artifact_type,
            source_artifact_uri=selection.artifact_uri,
            sections_created=sections_created,
            references_created=references_created,
            citation_blocks_created=citation_blocks_created,
            cleanup_performed=cleanup_performed,
            status=status,
        )

    def _persist_references(self, paper: Paper, references: list[ReferencePayload]) -> int:
        created = 0
        for payload in references:
            ref = PaperReference(
                paper_id=paper.id,
                local_ref_id=payload.local_ref_id,
                raw_text=payload.raw_text,
                title=payload.title,
                authors_json=payload.authors,
                year=payload.year,
                venue=payload.venue,
            )
            self.session.add(ref)
            created += 1
        self.session.flush()
        return created

    def _persist_blocks(self, paper: Paper, sections: list[SectionPayload]) -> tuple[int, int]:
        created = 0
        citation_blocks_created = 0
        for payload in sections:
            block = CitationBlock(
                paper_id=paper.id,
                section_title=payload.section_title,
                section_path=payload.section_path,
                chunk_index=payload.chunk_index,
                raw_text=payload.raw_text,
                raw_citation_keys=payload.raw_citation_keys,
                has_citations=bool(payload.raw_citation_keys),
                repair_used=payload.repair_used,
            )
            self.session.add(block)
            created += 1
            if payload.raw_citation_keys:
                citation_blocks_created += 1
        self.session.flush()
        return created, citation_blocks_created
