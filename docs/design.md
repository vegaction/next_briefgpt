# Design

## Product boundary

This repository is building the citation knowledge-base layer for a future scientific research system.
It is not yet the full research product.

That means the project should optimize first for:

- reliable ingestion
- robust parsing across imperfect inputs
- inspectable extraction outputs
- durable pipeline state
- local operability and reruns

It should not yet optimize for:

- end-user polish
- planner or agent UX
- broad orchestration features unrelated to the knowledge-base layer
- compatibility work that locks in weak early abstractions

## Current product intent

The current product promise is simple:
given an arXiv paper, the repository should be able to ingest it, preserve the important artifacts, parse references and citation-bearing text, extract structured citation summaries, and make the resulting state easy to inspect.

Success looks like:

- a paper can be processed end to end
- intermediate outputs can be inspected after each stage
- extraction outputs are attributable to specific blocks and references
- reruns are safe and understandable
- failures leave enough evidence to debug them quickly

## Scope

In scope:

- arXiv metadata and artifact ingestion
- source parsing and PDF fallback
- structured parse ingestion when available
- paper-local reference extraction
- citation-block detection
- LLM-backed citation intent and summary extraction
- database-backed persistence
- local scripts and APIs for operating and validating the pipeline
- summary-eval workflows for extraction quality

Out of scope for now:

- a global canonical bibliography graph
- document ranking, planning, or agent execution
- polished end-user applications
- multi-tenant infrastructure
- long-term backward compatibility guarantees

## Design biases

When making tradeoffs, bias toward the knowledge-base layer first.

In practice that means:

- choose inspectability over hidden convenience
- choose correctness over broad but fragile coverage
- choose explicit reruns over magical state reuse
- choose deterministic preprocessing over LLM improvisation
- choose fewer, clearer abstractions over generalized architecture too early

## Data-model stance

The schema is paper-centric on purpose.

Why:

- pipeline stages operate on one source paper at a time
- references are first discovered in the context of that paper
- extraction needs strong traceability from mention to block to reference to paper
- a paper-local model is easier to debug than an early global graph

This does not rule out a future global graph.
It just means the repository should not pretend to have one before the ingestion layer is trustworthy.

## LLM usage stance

LLMs are used for narrow semantic work, not as a substitute for the whole pipeline.

Preferred pattern:

- deterministic detection finds candidate citations
- deterministic parsing maps those citations to references when possible
- the LLM labels intent and writes a short summary for each candidate

Avoid:

- asking the LLM to invent citation structure
- asking the LLM to guess unknown references
- pushing orchestration logic into prompts

## Rerun philosophy

Reruns should be explicit, understandable, and cheap enough for local development.

That means:

- stage outputs can be cleared and rebuilt independently
- skipping work should be an explicit user choice
- job history should show what happened
- inspecting the database should be easier than reconstructing state from logs

## Pre-release posture

This repository is still pre-release.
Breaking changes are acceptable when they clearly improve:

- correctness
- inspectability
- developer velocity
- conceptual clarity

What is not acceptable is carrying stale compatibility code that obscures how the system really works today.
