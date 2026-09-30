"""Focused E5 tests for raw observations and separate human annotations."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evaluation.schemas import (
    CitationValidationObservation,
    DerivedMetrics,
    HumanAnnotation,
    RawQuestionResult,
)


def _raw_data() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": "run-fixture",
        "experiment_id": "exp_0123456789ab",
        "question_id": "Q002",
        "category": "direct_factual",
        "answerability": "answerable",
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "overall_status": "success",
        "retrieval": {
            "status": "success",
            "latency_ms": 10.5,
            "requested_top_k": 1,
            "candidate_count": 2,
            "result_count": 1,
            "embedding_model": "fixture-embedding-model",
            "embedding_dimension": 8,
            "results": [
                {
                    "rank": 1,
                    "document_id": 1,
                    "chunk_id": 2,
                    "chunk_order": 0,
                    "distance": 0.1,
                    "chunk_text": "retrieved fixture text",
                }
            ],
            "error": None,
        },
        "generation": {
            "status": "success",
            "latency_ms": 20.5,
            "generation_model": "fixture-generation-model",
            "answer": "fixture answer",
            "context_available": True,
            "sources": [
                {
                    "citation_id": 1,
                    "document_id": 1,
                    "chunk_id": 2,
                    "chunk_order": 0,
                    "distance": 0.1,
                }
            ],
            "citation_validation": None,
            "error": None,
        },
        "retrieval_observation_mismatch": None,
    }


def _parse_raw(data: dict[str, object]) -> RawQuestionResult:
    return RawQuestionResult.model_validate_json(json.dumps(data))


def test_valid_raw_question_result_parses() -> None:
    result = _parse_raw(_raw_data())
    assert result.overall_status.value == "success"
    assert result.retrieval is not None
    assert result.retrieval.results[0].chunk_text == "retrieved fixture text"


def test_operational_failure_needs_no_correctness_score() -> None:
    data = _raw_data()
    data.update(
        {
            "overall_status": "retrieval_failure",
            "retrieval": None,
            "generation": None,
            "completed_at": "2026-01-01T00:00:00.5Z",
        }
    )
    result = _parse_raw(data)
    assert result.overall_status.value == "retrieval_failure"
    assert "correctness" not in RawQuestionResult.model_fields


@pytest.mark.parametrize("value", [None, True, False])
def test_retrieval_observation_mismatch_states(value: bool | None) -> None:
    data = _raw_data()
    data["retrieval_observation_mismatch"] = value
    assert _parse_raw(data).retrieval_observation_mismatch is value


def test_unreviewed_human_annotation_accepts_nulls() -> None:
    annotation = HumanAnnotation.model_validate_json(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "run-fixture",
                "experiment_id": "exp_0123456789ab",
                "question_id": "Q002",
                "reviewer_id": None,
                "annotated_at": None,
                "correctness": None,
                "grounding": None,
                "refusal": None,
                "retrieval_relevance": [],
                "failure_tags": [],
                "notes": None,
            }
        )
    )
    assert annotation.correctness is None


def test_invalid_human_label_rejected() -> None:
    data = {
        "schema_version": "1.0.0",
        "run_id": "run-fixture",
        "experiment_id": "exp_0123456789ab",
        "question_id": "Q002",
        "correctness": "almost",
    }
    with pytest.raises(ValidationError):
        HumanAnnotation.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    ("payload", "expected_cited", "expected_valid", "expected_invalid"),
    [
        (
            {"cited_ids": [1, 2], "valid": True, "invalid_citations": []},
            (1, 2),
            True,
            (),
        ),
        (
            {"cited_ids": [1, 4], "valid": False, "invalid_citations": [4]},
            (1, 4),
            False,
            (4,),
        ),
        (
            {"cited_ids": [], "valid": True, "invalid_citations": []},
            (),
            True,
            (),
        ),
    ],
)
def test_citation_validation_observation_parses_production_shapes(
    payload: dict[str, object],
    expected_cited: tuple[int, ...],
    expected_valid: bool,
    expected_invalid: tuple[int, ...],
) -> None:
    observation = CitationValidationObservation.model_validate_json(json.dumps(payload))
    assert observation.cited_ids == expected_cited
    assert observation.valid is expected_valid
    assert observation.invalid_citations == expected_invalid


@pytest.mark.parametrize(
    "payload",
    [
        {"cited_ids": ["1"], "valid": True, "invalid_citations": []},
        {"cited_ids": [1], "valid": True, "invalid_citations": ["1"]},
        {"cited_ids": [1], "valid": "true", "invalid_citations": []},
        {
            "cited_ids": [1],
            "valid": True,
            "invalid_citations": [],
            "unexpected": "rejected",
        },
    ],
)
def test_citation_validation_observation_rejects_invalid_types_and_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CitationValidationObservation.model_validate_json(json.dumps(payload))


def test_citation_validation_observation_round_trip_uses_json_arrays() -> None:
    payload = {"cited_ids": [1, 4], "valid": False, "invalid_citations": [4]}
    observation = CitationValidationObservation.model_validate_json(json.dumps(payload))
    serialized = observation.model_dump(mode="json")
    assert serialized == payload
    assert isinstance(serialized["cited_ids"], list)
    assert isinstance(serialized["invalid_citations"], list)


def _derived_metrics_data() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": "run-fixture",
        "experiment_id": "exp_0123456789ab",
        "question_id": "Q002",
        "hit_at_k": None,
        "recall_at_k": None,
        "complete_evidence_at_k": None,
        "expected_point_coverage": None,
        "expected_point_matched_count": None,
        "expected_point_eligible_count": None,
        "automated_refusal_signal": None,
    }


@pytest.mark.parametrize(
    ("matched", "eligible", "coverage"),
    [
        (3, 3, 1.0),
        (1, 3, 1 / 3),
        (0, 3, 0.0),
    ],
)
def test_expected_point_coverage_accepts_eligible_values(
    matched: int,
    eligible: int,
    coverage: float,
) -> None:
    data = _derived_metrics_data()
    data.update(
        {
            "expected_point_coverage": coverage,
            "expected_point_matched_count": matched,
            "expected_point_eligible_count": eligible,
        }
    )
    metrics = DerivedMetrics.model_validate(data)
    assert metrics.expected_point_coverage == coverage
    assert metrics.expected_point_matched_count == matched
    assert metrics.expected_point_eligible_count == eligible


def test_expected_point_coverage_accepts_all_null_as_ineligible() -> None:
    metrics = DerivedMetrics.model_validate(_derived_metrics_data())
    assert metrics.expected_point_coverage is None
    assert metrics.expected_point_matched_count is None
    assert metrics.expected_point_eligible_count is None


@pytest.mark.parametrize(
    "updates",
    [
        {"expected_point_coverage": 0.5},
        {
            "expected_point_matched_count": 1,
            "expected_point_eligible_count": 2,
        },
        {
            "expected_point_coverage": 0.5,
            "expected_point_matched_count": -1,
            "expected_point_eligible_count": 2,
        },
        {
            "expected_point_coverage": 0.0,
            "expected_point_matched_count": 0,
            "expected_point_eligible_count": 0,
        },
        {
            "expected_point_coverage": 0.0,
            "expected_point_matched_count": 0,
            "expected_point_eligible_count": -1,
        },
        {
            "expected_point_coverage": 1.0,
            "expected_point_matched_count": 3,
            "expected_point_eligible_count": 2,
        },
        {
            "expected_point_coverage": -0.1,
            "expected_point_matched_count": 0,
            "expected_point_eligible_count": 2,
        },
        {
            "expected_point_coverage": 1.1,
            "expected_point_matched_count": 2,
            "expected_point_eligible_count": 2,
        },
        {
            "expected_point_coverage": 0.5,
            "expected_point_matched_count": 1,
            "expected_point_eligible_count": 3,
        },
    ],
)
def test_expected_point_coverage_rejects_inconsistent_values(
    updates: dict[str, object],
) -> None:
    data = _derived_metrics_data()
    data.update(updates)
    with pytest.raises(ValidationError):
        DerivedMetrics.model_validate(data)


@pytest.mark.parametrize(
    ("matched", "eligible", "coverage"),
    [
        (None, None, None),
        (1, 3, 1 / 3),
    ],
)
def test_expected_point_coverage_serializes_explicit_fields(
    matched: int | None,
    eligible: int | None,
    coverage: float | None,
) -> None:
    data = _derived_metrics_data()
    data.update(
        {
            "expected_point_coverage": coverage,
            "expected_point_matched_count": matched,
            "expected_point_eligible_count": eligible,
        }
    )
    serialized = DerivedMetrics.model_validate(data).model_dump(mode="json")
    assert serialized["expected_point_coverage"] == coverage
    assert serialized["expected_point_matched_count"] == matched
    assert serialized["expected_point_eligible_count"] == eligible
