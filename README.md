# briefgpt-arxiv

An arXiv MVP for citation-aware paper ingestion with three independently testable modules:

- `crawler`: fetches paper metadata and artifacts
- `parser`: detects citation-bearing blocks and reference mappings
- `extractor`: uses an LLM interface to extract citation mentions and semantics

The current data model is paper-centric:

- `papers`: one row per arXiv paper, including `current_version`, `parse_status`, and `ingest_status`
- `artifacts`: downloaded or derived files for a paper, such as `pdf` and `pdf_text`
- `paper_references`: references detected inside a paper
- `citation_blocks`: parsed text chunks for a paper, including section metadata and citation-bearing subsets
- `citation_mentions`: extracted mention-level citation semantics

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv run --with pytest pytest
uvicorn briefgpt_arxiv.main:app --reload
```

## End-To-End Example

1. Crawl a paper from arXiv:

```bash
curl -X POST http://127.0.0.1:8000/crawl/arxiv \
  -H 'Content-Type: application/json' \
  -d '{"arxiv_ids":["2603.15726"]}'
```

2. Parse the crawled paper:

```bash
curl -X POST http://127.0.0.1:8000/parse/1
```

3. Extract citation mentions:

```bash
curl -X POST http://127.0.0.1:8000/extract/1
```

4. Inspect the paper and references:

```bash
curl http://127.0.0.1:8000/papers/2603.15726v1
curl http://127.0.0.1:8000/papers/2603.15726v1/references
curl "http://127.0.0.1:8000/citations/search?intent=comparison"
```

## API

- `POST /crawl/arxiv`
- `POST /parse/{paper_id}`
- `POST /extract/{paper_id}`
- `GET /papers/{arxiv_id}`
- `GET /papers/{arxiv_id}/references`
- `GET /citations/search`

## Inspect the database

```bash
uv run python scripts/inspect_db.py overview
uv run python scripts/inspect_db.py paper 2603.15726v1
uv run python scripts/inspect_db.py mentions --arxiv-id 2603.15726v1
uv run python scripts/inspect_db.py extractions --arxiv-id 2603.15726v1
uv run python scripts/inspect_db.py blocks --arxiv-id 2603.15726v1
uv run python scripts/inspect_db.py jobs --job-type parse
uv run python scripts/inspect_db.py dump --limit 20
uv run python scripts/inspect_db.py sql "SELECT id, arxiv_id, ingest_status FROM papers;"
```

To run the bundled local demo against the checked-in artifact for `2603.15726v1`:

```bash
bash scripts/run_demo.sh
bash scripts/run_demo.sh 2603.15726v1 --mode local-artifacts
bash scripts/run_demo.sh 2603.15726 --mode crawl
```

For the full pipeline runner with explicit rerun controls:

```bash
uv run python scripts/run_pipeline.py 2603.15726 --json
uv run python scripts/run_pipeline.py 2603.15726 --skip-parse-if-parsed --skip-extract-if-ready
uv run python scripts/run_pipeline.py 2603.15726v1 --mode local-artifacts --json
```

`paper_id` is the primary workflow identifier for parse and extract operations.
`arxiv_id` is used for read APIs such as `GET /papers/{arxiv_id}`.

If you want to iterate on extraction only, you can run it directly against an already parsed paper:

```bash
uv run python scripts/run_extractor.py 2603.15726v1
uv run python scripts/run_extractor.py 1 --skip-if-ready
```

## Environment

- `DATABASE_URL`: SQLAlchemy URL, defaults to `sqlite:///./briefgpt.db`
- `ARTIFACT_ROOT`: root directory for downloaded artifacts, defaults to `./artifacts`
- `GEMINI_API_KEY`: Gemini API key used by all LLM-backed flows
- `GEMINI_MODEL`: Gemini model name, defaults to `gemini-2.5-flash-lite`

If `GEMINI_API_KEY` is not set, parsing still works, but extraction is disabled because summary generation is LLM-only.
