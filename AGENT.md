# AGENT.md

## Project

This repository is building the foundational paper and citation knowledge base for a future scientific deep research agent.

Current priority:

- reliable paper ingestion
- robust parsing across source and PDF inputs
- structured citation extraction
- inspectable and rerunnable pipeline state
- clean database-backed knowledge storage

This repository is not yet the full deep research agent. Do not optimize for planner behavior, agent UX, or high-level orchestration features until the knowledge base layer is solid.

## Working Defaults

- The project is pre-release.
- Do not optimize for backwards compatibility.
- Prefer the cleanest and most robust design over preserving current weaknesses.
- Breaking changes are acceptable when they improve correctness, clarity, maintainability, or data quality.
- Keep the repository clean. Remove obsolete code instead of preserving it for historical reasons.

Practical consequences:

- It is fine to change schema, APIs, scripts, or module boundaries.
- It is fine to remove prototypes that no longer match the intended architecture.
- Do not add compatibility layers unless they solve a real current problem.

## Current Scope

Focus on the knowledge base, not product polish.

In scope:

- arXiv ingestion and artifact tracking
- source parsing, structured parse ingestion, and PDF fallback
- reference extraction and citation block detection
- LLM-backed citation mention and citation semantics extraction
- database design, inspectability, and rerun support
- scripts and APIs that help validate and operate the pipeline

Out of scope for now:

- long-term backwards compatibility
- end-user product polish
- full scientific deep research planning and execution stack
- premature canonical graph modeling beyond what the current pipeline needs

## Read This As Canon

This file should stay focused on high-signal defaults, current scope, and implementation posture.

Detailed guidance now lives here:

- architecture docs: [ARCHITECTURE.md](/Users/sl/typefuture/next_briefgpt/ARCHITECTURE.md)
- design or product docs: [docs/design.md](/Users/sl/typefuture/next_briefgpt/docs/design.md)
- development docs: [docs/development.md](/Users/sl/typefuture/next_briefgpt/docs/development.md)

Use `AGENT.md` for the short version.

## Core Reminders

- Keep the repository clean.
- Prefer the simplest robust design.
- Breaking changes are acceptable when they improve the actual system.
- Do not add compatibility layers without a current need.
- Optimize the knowledge base first, not future product layers.
