# Development

## Development goals

Development in this repository should keep the codebase easy to trust, easy to inspect, and easy to rerun locally.

The bar is not "feature complete."
The bar is:

- clear pipeline behavior
- understandable failures
- minimal hidden state
- documentation that matches the code

## Local setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Useful local commands:

```bash
uv run pytest
uv run uvicorn briefgpt_arxiv.main:app --reload
uv run python scripts/run_pipeline.py 2603.15726 --json
uv run python scripts/inspect_db.py overview
```

If you want extraction enabled, provide either `OPEN_ROUTER_API_KEY` or `GEMINI_API_KEY`.

## Quality bar

Do not leave behind:

- hardcoded secrets
- stale integrations after provider or schema changes
- duplicate scripts for nearly identical workflows
- commented-out dead code
- temporary hacks without a clear reason
- documentation that no longer matches runtime behavior

Prefer changes that:

- add or update tests when behavior changes
- make failures visible and actionable
- preserve inspectability at stage boundaries
- keep reruns safe

## Verification order

Default verification order:

1. prove the narrowest changed behavior first
2. run the relevant unit or module tests
3. verify the CLI or API path that exercises the change
4. verify database inspectability if pipeline state changes
5. only then broaden scope or optimize

For pipeline changes, try to verify:

- crawl still persists the expected artifacts
- parse outputs are inspectable
- extract outputs are inspectable
- reruns do not require destructive resets

## Testing expectations

Target the smallest test surface that proves the change.

Typical commands:

```bash
uv run pytest
uv run pytest tests/test_parser.py
uv run pytest tests/test_extractor.py
uv run pytest tests/test_api.py
uv run pytest tests/test_summary_eval.py
```

When changing data flow, inspect the resulting state as well:

```bash
uv run python scripts/inspect_db.py overview
uv run python scripts/inspect_db.py paper 2603.15726v1
uv run python scripts/inspect_db.py jobs --limit 20
```

## Documentation expectations

Update docs whenever you change:

- setup steps
- runtime commands
- environment variables
- data model shape
- pipeline stage behavior
- architectural boundaries

At minimum, check whether the change requires updates to:

- `README.md`
- `ARCHITECTURE.md`
- `docs/design.md`
- `docs/development.md`

## Change tracking

For multi-step, high-risk, or multi-session work, create a change folder:

```text
dev/changes/{yymmdd}_{topic}/
```

Recommended files:

- `spec.md`: goals, scope, non-goals, constraints, acceptance criteria
- `progress.md`: execution log, decisions, blockers, next steps
- `verification.md`: commands run, observed results, residual risks
- `learning.md`: reusable lessons and follow-up ideas

Use this workflow for:

- schema redesigns
- parser rewrites
- extractor changes
- reliability fixes
- other changes that will span sessions or require careful verification

Do not create this structure for every small cleanup.

## Working style

When touching the pipeline:

- keep stage boundaries explicit
- prefer deleting stale code over preserving dead compatibility layers
- keep scripts and API paths aligned to the same service layer
- avoid introducing hidden state that makes local debugging harder

When uncertain, choose the option that makes the next debugging session easier.
