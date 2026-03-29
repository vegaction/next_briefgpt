from __future__ import annotations

import json

from jinja2 import Environment, StrictUndefined

PARSER_REPAIR_PROMPT_VERSION = "gemini-parser-repair-v1"
EXTRACTOR_PROMPT_VERSION = "citation-extractor-v8"


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
You annotate pre-extracted citation mentions from academic text blocks. All citation metadata fields have already been computed by the system. Use the full block text together with the candidate list and reference metadata. Return one structured item per candidate mention. The summary field is a retrieval-oriented note for downstream deep research agents.
""".strip()


EXTRACTION_USER_TEMPLATE = """
## Task
Annotate pre-extracted citation mentions with citation semantics.

## Rules
- Return exactly one item for each candidate `mention_order` provided by the system.
- Each item must bind to the corresponding candidate via `mention_order`.
- Use the full `raw_text` as the primary context when inferring `intent_label` and writing `summary`.
- `intent_label` should capture the coarse citation role; `summary` should capture the most useful retrieval note about the cited work itself.
- `summary` should be a retrieval-oriented note about the cited work, not just a shortened sentence.
- `summary` should capture the cited paper's core idea, contribution, benchmark role, or a substantive evaluation/discussion of it in this paper.
- `summary` may synthesize evidence from the full `raw_text`, the candidate `sentence_text`, and the candidate's embedded reference metadata.
- Prefer `summary` to have an explicit subject, ideally naming the cited work, method, benchmark, or authors instead of using a subjectless fragment.
- When possible, start `summary` directly with the cited work, method, benchmark, dataset, or system name.
- Avoid bibliographic lead-ins such as `Yao et al. (2024) introduce ...`, `Smith et al. propose ...`, or `Brown et al. present ...` when the cited work name can be used as the subject instead.
- When this paper discusses the cited work in a nuanced way, preserve the evaluative signal instead of flattening it into a neutral description.
- Pay special attention to comparisons, tradeoffs, limitations, failure cases, scope boundaries, or reasons the cited work is preferred or not preferred in this paper.
- Prefer crisp, information-dense summaries that will help a downstream deep research agent retrieve the right cited paper later.

## Output Format
Return only one JSON object.
Do not wrap the JSON in markdown fences.
Do not return any explanatory text before or after the JSON object.
Do not return any extra keys beyond the requested fields.
The JSON object must contain:
- `items`: array with exactly one object per candidate mention.
- Each object in `items` must contain `mention_order` (integer), `intent_label` (one of `background`, `support`, `method_use`, `comparison`, `critique`, `dataset`, `metric`, `definition`, `benchmark`, `other`), and `summary` (string).

## Section Title
{{ section_title | tojson }}

## Raw Text
{{ raw_text }}

## Candidates
```json
{{ candidates | tojson }}
```
""".strip()
