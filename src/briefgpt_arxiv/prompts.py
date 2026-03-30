from __future__ import annotations

import json

from jinja2 import Environment, StrictUndefined

PARSER_REPAIR_PROMPT_VERSION = "parser-repair-v1"
EXTRACTOR_PROMPT_VERSION = "citation-extractor-v17"


PROMPT_TEMPLATE_ENV = Environment(undefined=StrictUndefined, autoescape=False)
PROMPT_TEMPLATE_ENV.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)


PARSER_REPAIR_SYSTEM_TEMPLATE = """
You repair citation-bearing academic text blocks. Recover raw citation keys when the text contains unusual citation macros, and return only faithful cleaned text without inventing references.
""".strip()


PARSER_REPAIR_USER_TEMPLATE = """
## Task
Repair this citation-bearing academic text block for downstream parsing.

## Rules
- Keep the original meaning unchanged.
- Do not invent citation keys.
- Return `raw_citation_keys` using the citation labels visible or inferable from the text.

## Output Format
Return only one JSON object.
Do not wrap the JSON in markdown fences.
Do not return any explanatory text before or after the JSON object.
Do not return any extra keys beyond the requested fields.
The JSON object must contain:
- `raw_citation_keys`: array of strings containing the recovered citation keys for this text block.
- `cleaned_text`: string containing the cleaned text for downstream parsing.
- Optional `used_repair`: boolean indicating whether the text or citation keys were changed during repair.

{% if candidate_keys %}
## Candidate Keys
{{ candidate_keys | tojson }}
{% endif %}

## Raw Text
```text
{{ raw_text }}
```
""".strip()


EXTRACTION_SYSTEM_TEMPLATE = """
You distill citation-context text into structured commentary about cited works.
Citation mentions have already been extracted by the system.
For each candidate mention, assign an `intent_label` describing the citation's rhetorical role in the current paper and write a `summary` that preserves high-value commentary about the cited work.
Treat citation context as expert-written secondary commentary on the cited work.
Use the full block text, candidate list, and provided reference metadata, but keep `summary` centered on the cited work rather than on the current paper's citation act.
`summary` must be a standalone description of the cited work and must not describe what the current paper does with it.
Return exactly one structured item per candidate mention.
""".strip()

EXTRACTION_USER_TEMPLATE = """
## Task
Distill the candidate citation mentions into citation semantics and commentary about the cited works.

## Field Requirements
- `intent_label` should capture the coarse rhetorical role of the citation in context, using one of:
  - `background`
  - `comparison`
  - `method_basis`
  - `benchmark_or_dataset`
  - `supporting_evidence`
  - `critique`

## Rules
- Return exactly one item for each candidate `mention_order` provided by the system.
- Each item must bind to the corresponding candidate via `mention_order`.
- Use the full `raw_text` as the primary context when inferring `intent_label` and writing `summary`.
- Treat the citation context as high-quality expert commentary on the cited work, not as text to compress mechanically.
- `summary` should read like a compact but complete expert commentary note about the cited work, not just a shortened sentence.
- `summary` should capture the cited paper's core idea, distinctive contribution, benchmark role, or a substantive evaluation/discussion of it in this paper.
- `summary` may synthesize evidence from the full `raw_text`, the candidate `sentence_text`, and the candidate's embedded reference metadata.
- Prefer `summary` to name the cited work, method, benchmark, or system explicitly, ideally as the subject.
- When possible, begin `summary` with that name rather than with a generic lead-in or subjectless fragment.
- If the citation context is written partly in terms of the current paper's use of the cited work, rewrite that evidence into a standalone description of the cited work instead of repeating the current paper framing.
- Avoid bibliographic lead-ins such as `Yao et al. (2024) introduce ...`, `Smith et al. propose ...`, or `Brown et al. present ...` when the cited work name can be used as the subject instead.
- When this paper discusses the cited work in a nuanced way, preserve the evaluative signal instead of flattening it into a neutral description.
- Pay special attention to comparisons, tradeoffs, limitations, failure cases, scope boundaries, or reasons the cited work is preferred, dispreferred, or differentiated from nearby work.

## Summary Guidance
- Do not use document-internal wording such as `we`, `our`, `this paper`, `our system`, `used here`, `mentioned here`, or `this benchmark`.
- Do not frame `summary` around the current paper's citation act or discourse role, for example `cited as`, `presented as`, `used here`, `in this work`, `our results`, or `the authors use`.
- Never use phrases such as `this work`, `our work`, `the current paper`, `the current work`, `in this paper`, or `in our work` in `summary`.
- If a draft `summary` contains current-paper wording, rewrite it before returning so that only the cited work remains as the subject.
- Do not copy raw citation keys such as `liu2024rpo` into `summary`.
- Do not write vague summaries like `a prior method`, `baseline work`, or `used for comparison` unless the context truly provides no more specific information.
- Return an empty string for `summary` only when the mention conveys no useful information about the cited work beyond this paper's own score, rank, or win/loss on a benchmark.

## Output Format
Return only one JSON object.
Do not wrap the JSON in markdown fences.
Do not return any explanatory text before or after the JSON object.
Do not return any extra keys beyond the requested fields.
The JSON object must contain:
- `items`: array containing exactly one object per candidate mention.

Return a single JSON object in the following format:

{
  "items": [
    {
      "mention_order": 1,
      "intent_label": "background",
      "summary": "Example summary."
    }
  ]
}
`items` must contain exactly one object per candidate mention, and each item must contain only `mention_order`, `intent_label`, and `summary`.

## Input

### Section Title
{{ section_title }}

### Raw Text
```text
{{ raw_text }}
```

### Candidates
```json
{{ candidates | tojson }}
```
""".strip()
