"""Synthetic tests for reusable expected-point matching and coverage."""

import json
from datetime import UTC, datetime

import pytest

from evaluation.metrics import (
    MetricContractError,
    PointMatcher,
    aggregate_question_metric,
    calculate_question_metrics,
    match_expected_point,
    normalize_answer_clauses,
    same_clause_matches,
    validate_matcher_registry,
    whole_token_matcher,
)
from evaluation.schemas import BenchmarkRecord, GenerationObservation, OperationalStatus, RawQuestionResult


@pytest.fixture
def record():
    return BenchmarkRecord.model_validate_json(json.dumps({
        "id": "synthetic-forms",
        "benchmark_id": "fictional-lab",
        "benchmark_version": "1.0.0",
        "review_status": "reviewed",
        "category": "direct_factual",
        "question": "Which fictional items are needed?",
        "answerability": "answerable",
        "expected_behavior": "answer",
        "reference_answer": "Alpha form and beta certificate.",
        "expected_points": [
            {"id": "alpha", "text": "Alpha form", "automated_match": True},
            {"id": "beta", "text": "Beta certificate", "automated_match": True},
            {"id": "gamma", "text": "Manual interpretation", "automated_match": False},
        ],
        "evidence": [{
            "evidence_id": "synthetic-evidence", "pdf_page": 1,
            "printed_page_label": None, "section": None,
            "anchor": "Fictional item requirements.", "notes": None,
        }],
        "notes": None,
    }))


@pytest.fixture
def matchers(record):
    return {
        (record.id, "alpha"): PointMatcher(
            "Alpha form", "contiguous synthetic phrase",
            lambda answer: same_clause_matches(answer, (("alpha", "form"),)),
        ),
        (record.id, "beta"): PointMatcher(
            "Beta certificate", "contiguous synthetic phrase",
            lambda answer: same_clause_matches(answer, (("beta", "certificate"),)),
        ),
    }


def _generation(answer, status=OperationalStatus.SUCCESS):
    return GenerationObservation(
        status=status, latency_ms=1.0 if status is OperationalStatus.SUCCESS else None,
        generation_model="synthetic", answer=answer,
        context_available=True if status is OperationalStatus.SUCCESS else None,
        sources=(), citation_validation=None, error=None,
    )


def _result(record, generation):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RawQuestionResult(
        run_id="synthetic-run", experiment_id="synthetic-experiment",
        question_id=record.id, category=record.category, answerability=record.answerability,
        started_at=now, completed_at=now,
        overall_status=generation.status if generation else OperationalStatus.GENERATION_FAILURE,
        retrieval=None, generation=generation, retrieval_observation_mismatch=None,
    )


@pytest.mark.parametrize(("answer", "matched"), [
    ("The alpha form and beta certificate are required.", 2),
    ("Only the alpha form is present.", 1),
    ("Only a beta certificate is present.", 1),
    ("No fictional items are present.", 0),
    ("Alphabet forms and betamax certificates.", 0),
    ("Alpha. Form.", 0),
    ("ＡＬＰＨＡ—ＦＯＲＭ", 1),
    ("Alpha form. Alpha form.", 1),
])
def test_eligible_counts_and_coverage(record, matchers, answer, matched):
    metrics = calculate_question_metrics(record, _result(record, _generation(answer)), matchers=matchers)
    assert metrics.expected_point_matched_count == matched
    assert metrics.expected_point_eligible_count == 2
    assert metrics.expected_point_coverage == matched / 2


def test_normalization_preserves_decimal_and_normalizes_unicode():
    assert normalize_answer_clauses("  ＡＬＰＨＡ—form  7.25! BETA; café  ") == (
        ("alpha", "form", "7.25"), ("beta",), ("café",),
    )


def test_whole_token_does_not_match_substrings():
    matcher = whole_token_matcher("alpha")
    assert matcher("ＡＬＰＨＡ, beta.")
    assert not matcher("alphabet")


@pytest.mark.parametrize(("answer", "expected"), [
    ("alpha form approved", True),
    ("a form approved", True),
    ("alpha form; approved", False),
    ("alpha form pending", False),
    ("alpha format approved", False),
])
def test_clause_components_and_alternatives(answer, expected):
    assert same_clause_matches(
        answer, (("alpha", "form"), ("a", "form")), (("approved",),),
    ) is expected


def test_empty_phrase_does_not_match():
    assert not same_clause_matches("alpha form", ((),))


def test_registry_keys_and_text_are_validated(record, matchers):
    validate_matcher_registry([record], matchers)
    with pytest.raises(MetricContractError, match="registry drift"):
        validate_matcher_registry([record], {})
    extra = {**matchers, (record.id, "unused"): PointMatcher("Unused", "unused", lambda _: False)}
    with pytest.raises(MetricContractError, match="registry drift"):
        validate_matcher_registry([record], extra)
    changed = dict(matchers)
    changed[(record.id, "alpha")] = PointMatcher("Changed text", "changed", lambda _: False)
    with pytest.raises(MetricContractError, match="changed_text"):
        validate_matcher_registry([record], changed)


def test_automatable_points_require_explicit_matchers(record):
    with pytest.raises(MetricContractError, match="No automatable expected-point matcher"):
        calculate_question_metrics(record, _result(record, _generation("Alpha form")))
    with pytest.raises(MetricContractError, match="No automatable expected-point matcher"):
        match_expected_point("unregistered", "point", "alpha", matchers={})


def test_per_record_text_drift_fails_without_disclosing_content(record, matchers):
    changed = record.model_copy(update={
        "expected_points": (
            record.expected_points[0].model_copy(update={"text": "Changed synthetic content"}),
            *record.expected_points[1:],
        ),
    })
    with pytest.raises(MetricContractError, match="text drift") as exc:
        calculate_question_metrics(changed, _result(changed, _generation("alpha")), matchers=matchers)
    assert "Changed synthetic content" not in str(exc.value)


@pytest.mark.parametrize("generation", [
    None, _generation(None), _generation("alpha", OperationalStatus.GENERATION_FAILURE),
])
def test_unavailable_generation_is_null(record, matchers, generation):
    metrics = calculate_question_metrics(record, _result(record, generation), matchers=matchers)
    assert metrics.expected_point_coverage is None
    assert metrics.expected_point_matched_count is None
    assert metrics.expected_point_eligible_count is None


def test_missing_registry_fails_even_for_unavailable_generation(record):
    with pytest.raises(MetricContractError, match="No automatable expected-point matcher"):
        calculate_question_metrics(record, _result(record, None))


def test_no_eligible_points_needs_no_registry(record):
    record = record.model_copy(update={"expected_points": tuple(
        point.model_copy(update={"automated_match": False}) for point in record.expected_points
    )})
    metrics = calculate_question_metrics(record, _result(record, _generation("Alpha form")))
    assert metrics.expected_point_coverage is None
    assert metrics.expected_point_matched_count is None
    assert metrics.expected_point_eligible_count is None


def test_macro_aggregation_keeps_zero_and_excludes_null(record, matchers):
    results = [
        calculate_question_metrics(record, _result(record, _generation(answer)), matchers=matchers)
        for answer in ("Alpha form and beta certificate", "No matching item", None)
    ]
    aggregate = aggregate_question_metric(results, "expected_point_coverage")
    assert aggregate.n == 2
    assert aggregate.mean == 0.5


def test_result_identity_must_match(record, matchers):
    result = _result(record, _generation("alpha")).model_copy(update={"question_id": "different"})
    with pytest.raises(MetricContractError, match="question IDs differ"):
        calculate_question_metrics(record, result, matchers=matchers)
