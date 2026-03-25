from __future__ import annotations

import json
from unittest import TestCase

from briefgpt_arxiv.prompts import (
    EXTRACTION_SYSTEM_TEMPLATE,
    EXTRACTION_USER_TEMPLATE,
    PARSER_REPAIR_SYSTEM_TEMPLATE,
    PARSER_REPAIR_USER_TEMPLATE,
    PROMPT_TEMPLATE_ENV,
)


class PromptTemplateTests(TestCase):
    def test_parser_repair_template_renders_json_payload(self) -> None:
        system_instruction = PROMPT_TEMPLATE_ENV.from_string(PARSER_REPAIR_SYSTEM_TEMPLATE).render()
        user_text = PROMPT_TEMPLATE_ENV.from_string(PARSER_REPAIR_USER_TEMPLATE).render(
            raw_text="Text with \\mycite{BIBREF2}",
            candidate_keys=["BIBREF0", "BIBREF1"],
        )

        payload = json.loads(user_text)
        self.assertIn("repair citation-bearing academic text blocks", system_instruction.lower())
        self.assertEqual(["BIBREF0", "BIBREF1"], payload["candidate_keys"])
        self.assertEqual("Text with \\mycite{BIBREF2}", payload["raw_text"])

    def test_extraction_template_renders_json_payload(self) -> None:
        system_instruction = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_SYSTEM_TEMPLATE).render()
        user_text = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_USER_TEMPLATE).render(
            raw_text="ReAct BIBREF0 improves planning in a tool-using setup.",
            section_title="Introduction",
            candidates=[
                {
                    "mention_order": 0,
                    "raw_citation_key": "BIBREF0",
                    "citation_mention": "ReAct",
                    "sentence_text": "ReAct BIBREF0 improves planning.",
                    "reference": {
                        "title": "ReAct",
                        "year": 2023,
                        "raw_text": "ReAct raw text",
                    },
                }
            ],
        )

        self.assertIn("pre-extracted citation mentions", system_instruction.lower())
        self.assertIn("retrieval-oriented note", system_instruction.lower())
        self.assertIn("## Task", user_text)
        self.assertIn("## Raw Text", user_text)
        self.assertIn("## Candidates", user_text)
        self.assertIn('"Introduction"', user_text)
        self.assertIn('"raw_citation_key": "BIBREF0"', user_text)
        self.assertIn('"reference": {"title": "ReAct", "year": 2023, "raw_text": "ReAct raw text"}', user_text)
        self.assertIn('"title": "ReAct"', user_text)
        self.assertIn("Use the full `raw_text` as the primary context", user_text)
        self.assertIn("`intent_label` should capture the coarse citation role", user_text)
        self.assertIn("preserve the evaluative signal", user_text)
        self.assertIn("comparisons, tradeoffs, limitations, failure cases", user_text)
        self.assertIn("Keep `summary` compact", user_text)
        self.assertIn("focus on the cited work and why it matters here", user_text)
