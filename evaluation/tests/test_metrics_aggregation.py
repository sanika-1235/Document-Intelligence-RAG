"""Offline aggregation tests for E7 automated metrics and latency."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from evaluation.metrics import (
    aggregate_generation_latencies,
    aggregate_question_metric,
    aggregate_refusal_signals,
    aggregate_retrieval_latencies,
)
from evaluation.schemas import (
    Answerability,
    BenchmarkCategory,
    DerivedMetrics,
    GenerationObservation,
    OperationalStatus,
    RawQuestionResult,
    RetrievalObservation,
)


def _metrics(
    *,
    question_id: str,
    hit: bool | None = None,
    recall: float | None = None,
    complete: bool | None = None,
    coverage: float | None = None,
    matched: int | None = None,
    eligible: int | None = None,
    refusal: bool | None = None,
) -> DerivedMetrics:
    return DerivedMetrics(
        run_id="run-synthetic",
        experiment_id="exp_synthetic",
        question_id=question_id,
        hit_at_k=hit,
        recall_at_k=recall,
        complete_evidence_at_k=complete,
        expected_point_coverage=coverage,
        expected_point_matched_count=matched,
        expected_point_eligible_count=eligible,
        automated_refusal_signal=refusal,
    )


def test_macro_aggregation_excludes_nulls_and_reports_n() -> None:
    metrics = [
        _metrics(question_id="Q1", hit=True, recall=1.0),
        _metrics(question_id="Q2", hit=False, recall=0.5),
        _metrics(question_id="Q3", hit=None, recall=None),
    ]
    hit = aggregate_question_metric(metrics, "hit_at_k")
    recall = aggregate_question_metric(metrics, "recall_at_k")
    assert (hit.n, hit.mean) == (2, 0.5)
    assert (recall.n, recall.mean) == (2, 0.75)


def test_expected_point_coverage_uses_question_level_macro_mean() -> None:
    metrics = [
        _metrics(question_id="Q1", coverage=1.0, matched=1, eligible=1),
        _metrics(question_id="Q2", coverage=0.5, matched=2, eligible=4),
        _metrics(question_id="Q3"),
    ]
    aggregate = aggregate_question_metric(metrics, "expected_point_coverage")
    assert aggregate.n == 2
    assert aggregate.mean == 0.75


def test_no_eligible_metric_values_returns_zero_n_and_null_mean() -> None:
    aggregate = aggregate_question_metric([_metrics(question_id="Q1")], "recall_at_k")
    assert aggregate.n == 0
    assert aggregate.mean is None


def test_refusal_aggregation_reports_counts_and_signal_rate() -> None:
    aggregate = aggregate_refusal_signals(
        [
            _metrics(question_id="Q46", refusal=True),
            _metrics(question_id="Q47", refusal=False),
            _metrics(question_id="Q48", refusal=True),
            _metrics(question_id="Q42", refusal=None),
        ]
    )
    assert aggregate.eligible_n == 3
    assert aggregate.true_count == 2
    assert aggregate.false_count == 1
    assert aggregate.signal_rate == pytest.approx(2 / 3)


def test_refusal_aggregation_with_no_eligible_values() -> None:
    aggregate = aggregate_refusal_signals([_metrics(question_id="Q42")])
    assert aggregate.eligible_n == 0
    assert aggregate.true_count == 0
    assert aggregate.false_count == 0
    assert aggregate.signal_rate is None


def _retrieval(status: OperationalStatus, latency: float | None) -> RetrievalObservation:
    return RetrievalObservation(
        status=status,
        latency_ms=latency,
        requested_top_k=5,
        candidate_count=0,
        result_count=0,
        embedding_model="synthetic",
        embedding_dimension=8,
        results=(),
        error=None,
    )


def _generation(status: OperationalStatus, latency: float | None) -> GenerationObservation:
    return GenerationObservation(
        status=status,
        latency_ms=latency,
        generation_model="synthetic",
        answer="synthetic" if status is OperationalStatus.SUCCESS else None,
        context_available=True if status is OperationalStatus.SUCCESS else None,
        sources=(),
        citation_validation=None,
        error=None,
    )


def _raw(
    question_id: str,
    retrieval: RetrievalObservation | None,
    generation: GenerationObservation | None,
) -> RawQuestionResult:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RawQuestionResult(
        run_id="run-synthetic",
        experiment_id="exp_synthetic",
        question_id=question_id,
        category=BenchmarkCategory.DIRECT_FACTUAL,
        answerability=Answerability.ANSWERABLE,
        started_at=now,
        completed_at=now,
        overall_status=OperationalStatus.SUCCESS,
        retrieval=retrieval,
        generation=generation,
        retrieval_observation_mismatch=None,
    )


def test_one_latency_uses_the_single_value_for_every_statistic() -> None:
    aggregate = aggregate_retrieval_latencies(
        [_raw("Q1", _retrieval(OperationalStatus.SUCCESS, 12.5), None)]
    )
    assert aggregate.n == 1
    assert aggregate.mean == aggregate.median == 12.5
    assert aggregate.min == aggregate.max == aggregate.p95 == 12.5


def test_multiple_latencies_use_inclusive_p95() -> None:
    results = [
        _raw(f"Q{index}", _retrieval(OperationalStatus.SUCCESS, float(value)), None)
        for index, value in enumerate((1, 2, 3, 4), 1)
    ]
    aggregate = aggregate_retrieval_latencies(results)
    assert aggregate.n == 4
    assert aggregate.mean == 2.5
    assert aggregate.median == 2.5
    assert aggregate.min == 1.0
    assert aggregate.max == 4.0
    assert aggregate.p95 == pytest.approx(3.85)


def test_no_successful_latency_returns_null_statistics() -> None:
    result = _raw(
        "Q1",
        _retrieval(OperationalStatus.RETRIEVAL_FAILURE, 99.0),
        _generation(OperationalStatus.GENERATION_FAILURE, 199.0),
    )
    retrieval = aggregate_retrieval_latencies([result])
    generation = aggregate_generation_latencies([result])
    assert retrieval.n == generation.n == 0
    assert retrieval.mean is retrieval.median is retrieval.min is retrieval.max is retrieval.p95 is None
    assert generation.mean is generation.median is generation.min is generation.max is generation.p95 is None


def test_retrieval_and_generation_latencies_are_never_mixed() -> None:
    results = [
        _raw(
            "Q1",
            _retrieval(OperationalStatus.SUCCESS, 10.0),
            _generation(OperationalStatus.SUCCESS, 100.0),
        ),
        _raw(
            "Q2",
            _retrieval(OperationalStatus.SUCCESS, 20.0),
            _generation(OperationalStatus.GENERATION_FAILURE, 900.0),
        ),
        _raw(
            "Q3",
            _retrieval(OperationalStatus.RETRIEVAL_FAILURE, 800.0),
            _generation(OperationalStatus.SUCCESS, 300.0),
        ),
    ]
    retrieval = aggregate_retrieval_latencies(results)
    generation = aggregate_generation_latencies(results)
    assert (retrieval.n, retrieval.mean) == (2, 15.0)
    assert (generation.n, generation.mean) == (2, 200.0)
