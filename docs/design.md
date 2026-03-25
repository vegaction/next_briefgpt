# Design

## Product Boundary

This repository is building the foundational paper and citation knowledge base for a future scientific deep research agent.

It is not yet the full deep research agent.

Do not optimize for:

- planner behavior
- end-user product polish
- agent UX
- high-level orchestration features beyond the needs of the knowledge base

## Current Product Intent

Current priority:

- reliable paper ingestion
- robust parsing across source and PDF inputs
- structured citation extraction
- inspectable and rerunnable pipeline state
- clean database-backed knowledge storage

## Current Scope

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

## Design Bias

Prefer changes that strengthen the knowledge base layer first.

That usually means:

- reliable ingestion before richer UX
- inspectability before convenience abstractions
- simpler workflows before broader feature coverage
- robust internals before presentation concerns
