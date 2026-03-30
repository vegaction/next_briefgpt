from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from briefgpt_arxiv.config import LLMEndpointSettings, settings
from briefgpt_arxiv.llm_client import BaseLLMClient, create_llm_client
from briefgpt_arxiv.models import CitationMention
from briefgpt_arxiv.utils import ensure_parent, normalize_whitespace


SUMMARY_EVAL_JUDGE_PROMPT_VERSION = "summary-eval-mvp-v1"
EXPECTED_MODES = ("insight", "conservative")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "with",
}
STRONG_CLAIM_PATTERN = re.compile(
    r"\b(always|never|proves|provably|guarantees|guaranteed|definitive|definitively|best|state-of-the-art|sota)\b",
    re.IGNORECASE,
)


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class SummaryEvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    mention_id: int | None = Field(default=None, ge=1)
    raw_text: str = Field(min_length=1)
    sentence_text: str = Field(min_length=1)
    reference_title: str = Field(min_length=1)
    summary_gold: str = ""
    best_insight_gold: str = ""
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    expected_mode: Literal["insight", "conservative"]


class SummaryEvalPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    summary_pred: str = ""
    model_name: str | None = None
    prompt_version: str | None = None


class SummaryEvalJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insight_correctness: int = Field(ge=1, le=5)
    insight_lift: int = Field(ge=1, le=5)
    overreach: bool
    rationale: str = ""


class SummaryEvalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    expected_mode: Literal["insight", "conservative"]
    summary_pred: str
    model_name: str | None = None
    prompt_version: str | None = None
    judge_mode: Literal["llm", "heuristic"]
    judge_prompt_version: str
    insight_correctness: int
    insight_lift: int
    overreach: bool
    overreach_penalty: float
    overall: float
    rationale: str = ""
    reference_title: str
    best_insight_gold: str


@dataclass(slots=True)
class SummaryEvalReport:
    records: list[SummaryEvalRecord]
    judge_mode: str
    judge_prompt_version: str

    def as_dict(self) -> dict:
        sample_count = len(self.records)
        if sample_count == 0:
            return {
                "sample_count": 0,
                "judge_mode": self.judge_mode,
                "judge_prompt_version": self.judge_prompt_version,
                "metrics": {
                    "insight_correctness_avg": 0.0,
                    "insight_lift_avg": 0.0,
                    "overreach_rate": 0.0,
                    "overall_avg": 0.0,
                },
                "breakdown_by_expected_mode": {},
            }

        def aggregate(records: list[SummaryEvalRecord]) -> dict[str, float]:
            total = len(records)
            return {
                "count": total,
                "insight_correctness_avg": round(
                    sum(item.insight_correctness for item in records) / total,
                    4,
                ),
                "insight_lift_avg": round(sum(item.insight_lift for item in records) / total, 4),
                "overreach_rate": round(sum(1 for item in records if item.overreach) / total, 4),
                "overall_avg": round(sum(item.overall for item in records) / total, 4),
            }

        by_mode: dict[str, list[SummaryEvalRecord]] = {mode: [] for mode in EXPECTED_MODES}
        for record in self.records:
            by_mode[record.expected_mode].append(record)
        return {
            "sample_count": sample_count,
            "judge_mode": self.judge_mode,
            "judge_prompt_version": self.judge_prompt_version,
            "metrics": aggregate(self.records),
            "breakdown_by_expected_mode": {
                mode: aggregate(records)
                for mode, records in by_mode.items()
                if records
            },
        }


def build_overall_score(result: SummaryEvalJudgeResult) -> float:
    overreach_penalty = 5.0 if result.overreach else 0.0
    return round(
        0.45 * result.insight_correctness + 0.45 * result.insight_lift - 0.10 * overreach_penalty,
        4,
    )


def load_jsonl_models(path: Path, model_type):
    items = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                items.append(model_type.model_validate(payload))
            except ValidationError as exc:
                raise RuntimeError(f"Invalid record in {path} line {line_number}: {exc}") from exc
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_samples(path: Path) -> list[SummaryEvalSample]:
    return load_jsonl_models(path, SummaryEvalSample)


def load_predictions(path: Path) -> list[SummaryEvalPrediction]:
    return load_jsonl_models(path, SummaryEvalPrediction)


def load_predictions_from_db(session: Session, samples: list[SummaryEvalSample]) -> list[SummaryEvalPrediction]:
    mention_ids = [sample.mention_id for sample in samples if sample.mention_id is not None]
    if len(mention_ids) != len(samples):
        missing = [sample.sample_id for sample in samples if sample.mention_id is None]
        raise RuntimeError(
            "DB prediction loading requires `mention_id` on every sample. "
            f"Missing mention_id for sample_ids={missing!r}"
        )

    mentions = {
        mention.id: mention
        for mention in session.scalars(select(CitationMention).where(CitationMention.id.in_(mention_ids)))
    }
    missing_ids = sorted(set(mention_ids) - set(mentions.keys()))
    if missing_ids:
        raise RuntimeError(f"Missing citation_mentions in DB for ids={missing_ids!r}")

    return [
        SummaryEvalPrediction(
            sample_id=sample.sample_id,
            summary_pred=mentions[sample.mention_id].summary or "",
            model_name=mentions[sample.mention_id].model,
            prompt_version=mentions[sample.mention_id].prompt_version,
        )
        for sample in samples
    ]


def export_annotation_candidates(
    session: Session,
    *,
    limit: int,
) -> list[dict]:
    mentions = list(
        session.scalars(
            select(CitationMention)
            .where(CitationMention.summary.is_not(None))
            .order_by(CitationMention.id)
            .limit(limit)
        )
    )
    rows: list[dict] = []
    for mention in mentions:
        rows.append(
            {
                "sample_id": f"mention-{mention.id}",
                "mention_id": mention.id,
                "raw_text": mention.citation_block.raw_text,
                "sentence_text": mention.sentence_text,
                "reference_title": mention.citation_mention,
                "summary_gold": mention.summary or "",
                "best_insight_gold": "",
                "evidence_spans": [],
                "expected_mode": "insight",
            }
        )
    return rows


def normalize_prediction_text(text: str) -> str:
    return normalize_whitespace(text)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text.lower())
        if token not in STOPWORDS
    }


def _jaccard(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evidence_text(sample: SummaryEvalSample) -> str:
    if sample.evidence_spans:
        return " ".join(span.text for span in sample.evidence_spans)
    return sample.sentence_text


def heuristic_overreach(sample: SummaryEvalSample, summary_pred: str) -> bool:
    normalized = normalize_prediction_text(summary_pred)
    if not normalized:
        return False
    source_text = " ".join(
        filter(
            None,
            [
                sample.raw_text,
                sample.sentence_text,
                sample.reference_title,
                sample.best_insight_gold,
                _evidence_text(sample),
            ],
        )
    )
    pred_tokens = _tokenize(normalized)
    source_tokens = _tokenize(source_text)
    unsupported = pred_tokens - source_tokens
    if sample.expected_mode == "conservative" and len(pred_tokens) >= 12:
        return True
    if STRONG_CLAIM_PATTERN.search(normalized) and len(unsupported) >= 3:
        return True
    return False


class BaseSummaryEvalJudge:
    judge_mode: Literal["llm", "heuristic"] = "heuristic"
    judge_prompt_version: str = SUMMARY_EVAL_JUDGE_PROMPT_VERSION

    def judge(self, sample: SummaryEvalSample, prediction: SummaryEvalPrediction) -> SummaryEvalJudgeResult:
        raise NotImplementedError


class HeuristicSummaryEvalJudge(BaseSummaryEvalJudge):
    judge_mode: Literal["heuristic", "llm"] = "heuristic"

    def judge(self, sample: SummaryEvalSample, prediction: SummaryEvalPrediction) -> SummaryEvalJudgeResult:
        summary_pred = normalize_prediction_text(prediction.summary_pred)
        overreach = heuristic_overreach(sample, summary_pred)
        if sample.expected_mode == "conservative":
            if not summary_pred:
                return SummaryEvalJudgeResult(
                    insight_correctness=5,
                    insight_lift=4,
                    overreach=False,
                    rationale="Conservative sample stayed restrained and did not force an insight.",
                )
            correctness = 2 if overreach else 3
            lift = 1 if overreach else 2
            return SummaryEvalJudgeResult(
                insight_correctness=correctness,
                insight_lift=lift,
                overreach=overreach,
                rationale="Conservative sample should avoid over-distillation when evidence is weak.",
            )

        if not summary_pred:
            return SummaryEvalJudgeResult(
                insight_correctness=1,
                insight_lift=1,
                overreach=False,
                rationale="Insight sample returned an empty summary.",
            )

        gold_overlap = _jaccard(summary_pred, sample.best_insight_gold)
        evidence_overlap = _jaccard(summary_pred, _evidence_text(sample))
        raw_overlap = _jaccard(summary_pred, sample.raw_text)
        correctness_signal = max(gold_overlap, evidence_overlap)
        if correctness_signal >= 0.7:
            correctness = 5
        elif correctness_signal >= 0.5:
            correctness = 4
        elif correctness_signal >= 0.3:
            correctness = 3
        elif correctness_signal >= 0.15:
            correctness = 2
        else:
            correctness = 1

        lift_signal = max(gold_overlap - raw_overlap * 0.3, 0.0)
        if lift_signal >= 0.6:
            lift = 5
        elif lift_signal >= 0.4:
            lift = 4
        elif lift_signal >= 0.2:
            lift = 3
        elif lift_signal >= 0.1:
            lift = 2
        else:
            lift = 1

        if overreach:
            correctness = max(1, correctness - 1)
            lift = max(1, lift - 1)

        return SummaryEvalJudgeResult(
            insight_correctness=correctness,
            insight_lift=lift,
            overreach=overreach,
            rationale="Heuristic score estimated from overlap with the gold insight and source evidence.",
        )


class LLMSummaryEvalJudge(BaseSummaryEvalJudge):
    judge_mode: Literal["heuristic", "llm"] = "llm"

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def judge(self, sample: SummaryEvalSample, prediction: SummaryEvalPrediction) -> SummaryEvalJudgeResult:
        payload = self.llm_client.generate_json(
            system_instruction=SUMMARY_EVAL_JUDGE_SYSTEM_TEMPLATE,
            user_text=build_summary_eval_judge_user_text(sample=sample, prediction=prediction),
        )
        return parse_summary_eval_judge_payload(payload)


SUMMARY_EVAL_JUDGE_SYSTEM_TEMPLATE = """
You evaluate whether a citation summary distilled the best available insight from the input without overreaching.
Score only what is supported by the provided citation context and evidence.
""".strip()


def build_summary_eval_judge_user_text(*, sample: SummaryEvalSample, prediction: SummaryEvalPrediction) -> str:
    evidence_payload = [
        {"text": span.text, "start": span.start, "end": span.end}
        for span in sample.evidence_spans
    ]
    payload = {
        "sample_id": sample.sample_id,
        "expected_mode": sample.expected_mode,
        "reference_title": sample.reference_title,
        "raw_text": sample.raw_text,
        "sentence_text": sample.sentence_text,
        "summary_gold": sample.summary_gold,
        "best_insight_gold": sample.best_insight_gold,
        "evidence_spans": evidence_payload,
        "summary_pred": normalize_prediction_text(prediction.summary_pred),
    }
    return f"""
## Task
Judge whether `summary_pred` captured the best available insight from the citation input.

## Scoring Rubric
- `insight_correctness`: 1-5. Does the core claim in `summary_pred` match the best available insight and stay grounded in the evidence?
- `insight_lift`: 1-5. Does `summary_pred` distill the input into a higher-density insight rather than paraphrasing it?
- `overreach`: true if `summary_pred` makes claims that go beyond the supported evidence.

## Special Rule
- If `expected_mode` is `conservative`, reward restraint. A weak or ambiguous input should not be turned into a strong insight.

## Output Format
Return only one JSON object with exactly these keys:
- `insight_correctness`
- `insight_lift`
- `overreach`
- `rationale`

## Input
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```
""".strip()


def parse_summary_eval_judge_payload(payload: dict) -> SummaryEvalJudgeResult:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Summary eval judge returned a non-object JSON payload: {payload!r}")
    expected_keys = {"insight_correctness", "insight_lift", "overreach", "rationale"}
    if set(payload.keys()) != expected_keys:
        raise RuntimeError(
            "Summary eval judge returned unexpected keys: "
            f"{sorted(payload.keys())!r}; expected {sorted(expected_keys)!r}"
        )
    return SummaryEvalJudgeResult.model_validate(payload)


def build_summary_eval_judge(
    *,
    judge_mode: Literal["llm", "heuristic"],
    provider: str | None = None,
    model_name: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> BaseSummaryEvalJudge:
    if judge_mode == "heuristic":
        return HeuristicSummaryEvalJudge()
    if llm_client is None:
        endpoint = LLMEndpointSettings(
            provider=provider or settings.extractor_llm.provider,
            model_name=model_name or settings.extractor_llm.model_name,
            reasoning_enabled=settings.extractor_llm.reasoning_enabled,
        )
        if not settings.has_llm_api_key(endpoint.provider):
            raise RuntimeError(
                f"Summary eval judge requires credentials for provider {endpoint.provider!r}. "
                "Use --judge-mode heuristic for an offline smoke test."
            )
        llm_client = create_llm_client(endpoint)
    return LLMSummaryEvalJudge(llm_client=llm_client)


def evaluate_predictions(
    *,
    samples: list[SummaryEvalSample],
    predictions: list[SummaryEvalPrediction],
    judge: BaseSummaryEvalJudge,
) -> SummaryEvalReport:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    prediction_by_id = {prediction.sample_id: prediction for prediction in predictions}

    missing_predictions = sorted(set(sample_by_id.keys()) - set(prediction_by_id.keys()))
    if missing_predictions:
        raise RuntimeError(f"Missing predictions for sample_ids={missing_predictions!r}")

    unexpected_predictions = sorted(set(prediction_by_id.keys()) - set(sample_by_id.keys()))
    if unexpected_predictions:
        raise RuntimeError(f"Unexpected predictions for sample_ids={unexpected_predictions!r}")

    records: list[SummaryEvalRecord] = []
    for sample in samples:
        prediction = prediction_by_id[sample.sample_id]
        result = judge.judge(sample, prediction)
        overreach_penalty = 5.0 if result.overreach else 0.0
        records.append(
            SummaryEvalRecord(
                sample_id=sample.sample_id,
                expected_mode=sample.expected_mode,
                summary_pred=normalize_prediction_text(prediction.summary_pred),
                model_name=prediction.model_name,
                prompt_version=prediction.prompt_version,
                judge_mode=judge.judge_mode,
                judge_prompt_version=judge.judge_prompt_version,
                insight_correctness=result.insight_correctness,
                insight_lift=result.insight_lift,
                overreach=result.overreach,
                overreach_penalty=overreach_penalty,
                overall=build_overall_score(result),
                rationale=result.rationale,
                reference_title=sample.reference_title,
                best_insight_gold=sample.best_insight_gold,
            )
        )
    return SummaryEvalReport(
        records=records,
        judge_mode=judge.judge_mode,
        judge_prompt_version=judge.judge_prompt_version,
    )
