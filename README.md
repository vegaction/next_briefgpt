# briefgpt-arxiv

An arXiv MVP for citation-aware paper ingestion with three independently testable modules:

- `crawler`: fetches paper metadata and artifacts
- `parser`: detects citation-bearing blocks and reference mappings
- `extractor`: uses an LLM interface to extract citation mentions and semantics

The current data model is paper-centric:

- `papers`: one row per arXiv paper version, keyed by `arxiv_id` + `version`
- `artifacts`: downloaded or derived files for a paper, such as `pdf` and `pdf_text`
- `paper_references`: references detected inside a paper, with optional `cited_arxiv_id` + `cited_version`
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
uv run python scripts/inspect_db.py sql "SELECT id, arxiv_id, version, ingest_status FROM papers;"
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
Read APIs accept either a canonical arXiv id such as `2603.15726` or a versioned id such as `2603.15726v1`.

If you want to iterate on extraction only, you can run it directly against an already parsed paper:

```bash
uv run python scripts/run_extractor.py 2603.15726v1
uv run python scripts/run_extractor.py 1 --skip-if-ready
```

## Summary Eval MVP

The repo includes a minimal summary evaluation harness for the question:
`did this summary distill the best available insight from the citation input without overreaching?`

1. Export starter rows for manual labeling:

```bash
uv run python scripts/export_summary_eval_candidates.py --limit 100
```

2. Run the MVP harness on a labeled JSONL file:

```bash
uv run python scripts/run_summary_eval.py \
  --samples data/summary_eval/mvp_examples.jsonl \
  --predictions data/summary_eval/mvp_predictions_example.jsonl \
  --judge-mode heuristic
```

If you already have `mention_id` values in the gold file, you can omit `--predictions` and score the current DB summaries directly.

The MVP reports only three core metrics:

- `Insight Correctness`
- `Insight Lift`
- `Overreach Rate`

## Environment

- `DATABASE_URL`: SQLAlchemy URL, defaults to `sqlite:///./briefgpt.db`
- `ARTIFACT_ROOT`: root directory for downloaded artifacts, defaults to `./artifacts`
- `OPEN_ROUTER_API_KEY`: OpenRouter API key for the `openai_compatible` provider
- `OPEN_ROUTER_MODEL`: default OpenRouter model when `config.yaml` does not specify one
- `GEMINI_API_KEY`: Gemini API key for the `gemini` provider
- `GEMINI_MODEL`: default Gemini model when `config.yaml` does not specify one
- `OPENROUTER_REASONING_ENABLED`: enables OpenRouter reasoning mode, defaults to `true`

LLM provider selection lives in `config.yaml`:

```yaml
llm:
  parser:
    provider: "openai_compatible"
    model_name: "nvidia/nemotron-3-super-120b-a12b:free"
  extractor:
    provider: "openai_compatible"
    model_name: "nvidia/nemotron-3-super-120b-a12b:free"
```

Supported providers are `openai_compatible` and `gemini`.

If the configured provider for extraction does not have credentials, parsing still works, but extraction is disabled because summary generation is LLM-only.
