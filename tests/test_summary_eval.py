from __future__ import annotations

from unittest.mock import Mock

from evaluation.summary_eval import (
    HeuristicSummaryEvalJudge,
    SummaryEvalPrediction,
    SummaryEvalSample,
    build_overall_score,
    build_summary_eval_judge,
    evaluate_predictions,
    parse_summary_eval_judge_payload,
)


def test_heuristic_judge_rewards_restrained_conservative_prediction() -> None:
    sample = SummaryEvalSample.model_validate(
        {
            "sample_id": "s1",
            "raw_text": "We compare against ReAct BIBREF0 in our experiments.",
            "sentence_text": "We compare against ReAct BIBREF0 in our experiments.",
            "reference_title": "ReAct",
            "summary_gold": "",
            "best_insight_gold": "",
            "evidence_spans": [{"text": "compare against ReAct"}],
            "expected_mode": "conservative",
        }
    )
    prediction = SummaryEvalPrediction(sample_id="s1", summary_pred="")

    result = HeuristicSummaryEvalJudge().judge(sample, prediction)

    assert result.insight_correctness == 5
    assert result.insight_lift == 4
    assert result.overreach is False


def test_heuristic_judge_flags_conservative_overreach() -> None:
    sample = SummaryEvalSample.model_validate(
        {
            "sample_id": "s2",
            "raw_text": "We compare against ReAct BIBREF0 in our experiments.",
            "sentence_text": "We compare against ReAct BIBREF0 in our experiments.",
            "reference_title": "ReAct",
            "summary_gold": "",
            "best_insight_gold": "",
            "evidence_spans": [{"text": "compare against ReAct"}],
            "expected_mode": "conservative",
        }
    )
    prediction = SummaryEvalPrediction(
        sample_id="s2",
        summary_pred="ReAct definitively outperforms every prior agent across planning tasks.",
    )

    result = HeuristicSummaryEvalJudge().judge(sample, prediction)

    assert result.overreach is True
    assert result.insight_correctness <= 2
    assert result.insight_lift == 1


def test_evaluate_predictions_aggregates_scores() -> None:
    samples = [
        SummaryEvalSample.model_validate(
            {
                "sample_id": "s1",
                "raw_text": "TravelPlanner outperforms earlier planning agents but struggles with budget constraints.",
                "sentence_text": "TravelPlanner outperforms earlier planning agents but struggles with budget constraints.",
                "reference_title": "TravelPlanner",
                "summary_gold": "",
                "best_insight_gold": "TravelPlanner is strong on itinerary planning but weak on budget constraints.",
                "evidence_spans": [{"text": "struggles with budget constraints"}],
                "expected_mode": "insight",
            }
        ),
        SummaryEvalSample.model_validate(
            {
                "sample_id": "s2",
                "raw_text": "We compare against ReAct BIBREF0 in our experiments.",
                "sentence_text": "We compare against ReAct BIBREF0 in our experiments.",
                "reference_title": "ReAct",
                "summary_gold": "",
                "best_insight_gold": "",
                "evidence_spans": [{"text": "compare against ReAct"}],
                "expected_mode": "conservative",
            }
        ),
    ]
    predictions = [
        SummaryEvalPrediction(
            sample_id="s1",
            summary_pred="TravelPlanner is effective for itinerary planning but weak at budget constraints.",
        ),
        SummaryEvalPrediction(sample_id="s2", summary_pred=""),
    ]

    report = evaluate_predictions(
        samples=samples,
        predictions=predictions,
        judge=HeuristicSummaryEvalJudge(),
    )

    assert len(report.records) == 2
    assert report.as_dict()["sample_count"] == 2
    assert "metrics" in report.as_dict()


def test_parse_summary_eval_judge_payload_requires_expected_keys() -> None:
    result = parse_summary_eval_judge_payload(
        {
            "insight_correctness": 4,
            "insight_lift": 5,
            "overreach": False,
            "rationale": "Good distillation.",
        }
    )

    assert result.insight_correctness == 4
    assert result.insight_lift == 5
    assert result.overreach is False


def test_build_overall_score_applies_overreach_penalty() -> None:
    score = build_overall_score(
        parse_summary_eval_judge_payload(
            {
                "insight_correctness": 5,
                "insight_lift": 5,
                "overreach": True,
                "rationale": "Overreached.",
            }
        )
    )

    assert score == 4.0


def test_build_summary_eval_judge_uses_supplied_llm_client() -> None:
    llm_client = Mock()
    llm_client.generate_json.return_value = {
        "insight_correctness": 4,
        "insight_lift": 4,
        "overreach": False,
        "rationale": "Grounded.",
    }
    judge = build_summary_eval_judge(judge_mode="llm", llm_client=llm_client)
    sample = SummaryEvalSample.model_validate(
        {
            "sample_id": "s1",
            "raw_text": "TravelPlanner outperforms earlier planning agents but struggles with budget constraints.",
            "sentence_text": "TravelPlanner outperforms earlier planning agents but struggles with budget constraints.",
            "reference_title": "TravelPlanner",
            "summary_gold": "",
            "best_insight_gold": "TravelPlanner is strong on itinerary planning but weak on budget constraints.",
            "evidence_spans": [{"text": "struggles with budget constraints"}],
            "expected_mode": "insight",
        }
    )
    prediction = SummaryEvalPrediction(sample_id="s1", summary_pred="TravelPlanner has a budget limitation.")

    result = judge.judge(sample, prediction)

    assert result.insight_correctness == 4
    llm_client.generate_json.assert_called_once()
