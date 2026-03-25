# Development

## Quality Bar

Keep the repository clean and production-minded even before release.

Do not leave behind:

- hardcoded secrets
- stale provider integrations after migrations
- duplicate scripts for nearly identical workflows
- commented-out dead code
- temporary hacks without a clear follow-up path

Whenever practical:

- add or update tests with code changes
- make failures visible and understandable
- keep local inspection and verification easy

## Verification Order

Default verification flow:

1. Verify type-level and unit-level correctness first.
2. Verify module-level behavior in isolation.
3. Verify local end-to-end pipeline flow.
4. Verify rerun and inspection workflows.
5. Only then optimize or broaden scope.

For ingestion and extraction changes, prefer proving:

- the pipeline can ingest a paper end-to-end
- parser outputs are inspectable
- extractor outputs are inspectable
- reruns do not require destructive resets

## Change Tracking

For multi-step, high-risk, or multi-session changes, create and maintain a change folder:

- `dev/changes/{yymmdd}_{topic}/`

Recommended files:

- `spec.md`: goals, scope, non-goals, constraints, acceptance criteria
- `progress.md`: running execution log, decisions, blockers, next steps
- `verification.md`: commands run, results, residual risks
- `learning.md`: reusable lessons, follow-up ideas, documentation updates

Use this for:

- schema redesigns
- parser rewrites
- extraction changes
- reliability fixes
- other changes that span multiple sessions or need deliberate tracking

Do not default to this folder for every small refactor.
