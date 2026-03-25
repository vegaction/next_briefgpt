from __future__ import annotations

import json

from jinja2 import Environment, StrictUndefined

PARSER_REPAIR_PROMPT_VERSION = "gemini-parser-repair-v1"
EXTRACTOR_PROMPT_VERSION = "gemini-citation-extractor-v5"


PROMPT_TEMPLATE_ENV = Environment(undefined=StrictUndefined, autoescape=False)
PROMPT_TEMPLATE_ENV.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)


EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mention_order": {"type": "integer"},
                    "intent_label": {
                        "type": "string",
                        "enum": [
                            "background",
                            "support",
                            "method_use",
                            "comparison",
                            "critique",
                            "dataset",
                            "metric",
                            "definition",
                            "other",
                        ],
                    },
                    "summary": {"type": "string"},
                },
                "required": [
                    "mention_order",
                    "intent_label",
                    "summary",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


PARSER_REPAIR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "raw_citation_keys": {"type": "array", "items": {"type": "string"}},
        "cleaned_text": {"type": "string"},
        "used_repair": {"type": "boolean"},
    },
    "required": ["raw_citation_keys", "cleaned_text", "used_repair"],
    "additionalProperties": False,
}


PARSER_REPAIR_SYSTEM_TEMPLATE = """
You repair citation-bearing academic text blocks. Recover raw citation keys when the text contains unusual citation macros, and return only faithful cleaned text without inventing references.
""".strip()


PARSER_REPAIR_USER_TEMPLATE = """
{
  "task": "Repair a citation-bearing text block for downstream parsing.",
  "rules": [
    "Keep the original meaning unchanged.",
    "Do not invent citation keys.",
    "Return raw_citation_keys using the citation labels visible or inferable from the text.",
    "If no repair is needed, return the original text and used_repair=false."
  ],
  "candidate_keys": {{ candidate_keys | tojson }},
  "raw_text": {{ raw_text | tojson }}
}
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
- When this paper discusses the cited work in a nuanced way, preserve the evaluative signal instead of flattening it into a neutral description.
- Pay special attention to comparisons, tradeoffs, limitations, failure cases, scope boundaries, or reasons the cited work is preferred or not preferred in this paper.
- Keep `summary` compact: usually one sentence or two short clauses.
- Do not waste `summary` on generic framing about what the current paper is doing; focus on the cited work and why it matters here.
- Prefer crisp, information-dense summaries that will help a downstream deep research agent retrieve the right cited paper later.

## Section Title
{{ section_title | tojson }}

## Raw Text
{{ raw_text }}

## Candidates
```json
{{ candidates | tojson }}
```
""".strip()
