# briefgpt-arxiv

`briefgpt-arxiv` is a paper-centric arXiv ingestion and citation extraction pipeline.
It fetches paper artifacts, parses references and citation-bearing text blocks, runs LLM-backed citation summarization, and stores every stage in a local SQL database so the pipeline is inspectable and rerunnable.

This repository is the knowledge-base layer for a future research system, not the full agent product.

## What the repository does

- Crawls arXiv metadata and artifacts for one or more papers
- Persists raw and derived artifacts under `artifacts/`
- Parses references and citation-bearing blocks from source, structured parse, or PDF input
- Extracts mention-level citation intent and short summaries with an LLM
- Exposes the resulting state through a FastAPI app and local inspection scripts
- Includes a small summary-eval harness for checking summary quality

## Pipeline at a glance

```text
arXiv id
  -> crawl
  -> papers + artifacts
  -> parse
  -> paper_references + citation_blocks
  -> extract
  -> citation_mentions
```

Each stage writes durable state before the next stage starts. That makes it easy to rerun one step, inspect intermediate outputs, and debug failures without starting over from scratch.

## Repository map

- `src/briefgpt_arxiv/main.py`: FastAPI entrypoint
- `src/briefgpt_arxiv/models.py`: SQLAlchemy models
- `src/briefgpt_arxiv/services/crawler.py`: arXiv ingestion
- `src/briefgpt_arxiv/services/parser.py`: reference and citation-block parsing
- `src/briefgpt_arxiv/services/extractor.py`: LLM-backed citation extraction
- `src/briefgpt_arxiv/services/orchestrator.py`: end-to-end pipeline composition
- `scripts/run_pipeline.py`: CLI pipeline runner
- `scripts/run_extractor.py`: extractor-only runner
- `scripts/inspect_db.py`: database inspection utility
- `scripts/run_demo.sh`: local demo wrapper
- `src/evaluation/summary_eval.py`: summary evaluation logic
- `ARCHITECTURE.md`: system architecture and data flow
- `docs/design.md`: product and design boundaries
- `docs/development.md`: development workflow and verification guidance

## Quick start

### 1. Create the environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure credentials

Create a `.env` file if you want LLM-backed extraction:

```bash
OPEN_ROUTER_API_KEY=...
# or
GEMINI_API_KEY=...
```

Parsing works without LLM credentials. Extraction does not.

### 3. Optional model selection

`config.yaml` controls which provider and model each LLM-backed stage uses:

```yaml
llm:
  parser:
    provider: "openai_compatible"
    model_name: "nvidia/nemotron-3-super-120b-a12b:free"
  extractor:
    provider: "openai_compatible"
    model_name: "nvidia/nemotron-3-super-120b-a12b:free"
```

Supported providers today:

- `openai_compatible`
- `gemini`

### 4. Run tests

```bash
uv run pytest
```

### 5. Start the API

```bash
uv run uvicorn briefgpt_arxiv.main:app --reload
```

## Common workflows

### Run the full pipeline from arXiv

```bash
uv run python scripts/run_pipeline.py 2603.15726
uv run python scripts/run_pipeline.py 2603.15726 --json
uv run python scripts/run_pipeline.py 2603.15726 --skip-parse-if-parsed --skip-extract-if-ready
```

### Run the pipeline from checked-in local artifacts

Use this when you already have files under `artifacts/<arxiv_id>/<version>/`:

```bash
uv run python scripts/run_pipeline.py 2603.15726v1 --mode local-artifacts
bash scripts/run_demo.sh
bash scripts/run_demo.sh 2603.15726v1 --mode local-artifacts
```

`local-artifacts` mode requires a versioned id such as `2603.15726v1`.

### Run extractor only

```bash
uv run python scripts/run_extractor.py 2603.15726v1
uv run python scripts/run_extractor.py 1 --skip-if-ready
uv run python scripts/run_extractor.py 2603.15726v1 --json
```

### Use the API directly

```bash
curl -X POST http://127.0.0.1:8000/crawl/arxiv \
  -H 'Content-Type: application/json' \
  -d '{"arxiv_ids":["2603.15726"]}'

curl -X POST http://127.0.0.1:8000/parse/1
curl -X POST http://127.0.0.1:8000/extract/1

curl http://127.0.0.1:8000/papers/2603.15726v1
curl http://127.0.0.1:8000/papers/2603.15726v1/references
curl "http://127.0.0.1:8000/citations/search?intent=comparison"
```

Write endpoints use `paper_id`. Read endpoints accept either `2603.15726` or `2603.15726v1`.

## Database model

The main tables are:

- `papers`: one row per arXiv paper version
- `artifacts`: raw and derived files such as `pdf`, `source`, `pdf_text`
- `paper_references`: references extracted from one paper
- `citation_blocks`: text chunks that may contain citations
- `citation_mentions`: mention-level extraction results
- `ingestion_jobs`: crawl, parse, and extract job history

This schema is intentionally paper-centric. References are scoped to the source paper rather than stored in a global citation graph.

## Inspecting pipeline state

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

## Summary Eval MVP

The repository includes a small evaluation harness for one narrow question:
did the extracted summary capture the cited work's useful contribution without overclaiming?

Export candidates for labeling:

```bash
uv run python scripts/export_summary_eval_candidates.py --limit 100
```

Run the evaluator:

```bash
uv run python scripts/run_summary_eval.py \
  --samples data/summary_eval/mvp_examples.jsonl \
  --predictions data/summary_eval/mvp_predictions_example.jsonl \
  --judge-mode heuristic
```

If your gold file already includes `mention_id`, you can omit `--predictions` and score the current DB summaries directly.

The current report centers on:

- `Insight Correctness`
- `Insight Lift`
- `Overreach Rate`

## Configuration reference

Environment variables:

- `DATABASE_URL`: SQLAlchemy database URL, default `sqlite:///./briefgpt.db`
- `ARTIFACT_ROOT`: artifact root, default `./artifacts`
- `OPEN_ROUTER_API_KEY`: API key for `openai_compatible`
- `OPEN_ROUTER_MODEL`: default model for `openai_compatible`
- `OPENROUTER_BASE_URL`: OpenRouter-compatible base URL
- `OPENROUTER_SITE_URL`: optional OpenRouter attribution URL
- `OPENROUTER_SITE_NAME`: optional OpenRouter attribution name
- `OPENROUTER_REASONING_ENABLED`: default `true`
- `OPENROUTER_TIMEOUT_SECONDS`: timeout used by parser repair requests
- `GEMINI_API_KEY`: API key for `gemini`
- `GEMINI_MODEL`: default model for `gemini`
- `SUMMARY_DEBUG_LOG_PATH`: debug JSONL for extractor prompt/response traces

YAML config:

- `llm.parser.provider`
- `llm.parser.model_name`
- `llm.parser.reasoning_enabled`
- `llm.extractor.provider`
- `llm.extractor.model_name`
- `llm.extractor.reasoning_enabled`

## Current boundaries

In scope:

- arXiv ingestion
- artifact persistence
- citation-aware parsing
- mention-level citation extraction
- inspectable local operations
- summary-quality evaluation

Out of scope for now:

- full research-agent orchestration
- polished end-user UX
- a global canonical paper graph
- compatibility promises for pre-release schemas

For architecture details, see `ARCHITECTURE.md`.
