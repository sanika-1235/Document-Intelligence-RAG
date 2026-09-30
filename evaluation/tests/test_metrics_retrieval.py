"""Synthetic tests for exact-anchor E7 retrieval metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from evaluation.metrics import calculate_question_metrics
from evaluation.schemas import (
    BenchmarkRecord,
    OperationalStatus,
    RawQuestionResult,
    RetrievalObservation,
    RetrievedResult,
)


def _record(*anchors: str, answerability: str = "answerable") -> BenchmarkRecord:
    evidence = [
        {
            "evidence_id": f"Q900-E{index}",
            "pdf_page": index,
            "printed_page_label": str(index),
            "section": "Synthetic",
            "anchor": anchor,
            "notes": None,
        }
        for index, anchor in enumerate(anchors, 1)
    ]
    behavior = (
        "refuse_or_state_insufficient_information"
        if answerability == "unanswerable"
        else "clarify_or_qualify"
        if answerability == "ambiguous"
        else "answer"
    )
    return BenchmarkRecord.model_validate_json(
        json.dumps(
            {
                "id": "Q900",
                "benchmark_id": "synthetic",
                "benchmark_version": "1.0.0",
                "review_status": "reviewed",
                "category": answerability if answerability != "answerable" else "direct_factual",
                "question": "Synthetic question?",
                "answerability": answerability,
                "expected_behavior": behavior,
                "reference_answer": None if answerability == "unanswerable" else "Synthetic answer.",
                "expected_points": [
                    {"id": "P1", "text": "Human-only point", "automated_match": False}
                ],
                "evidence": evidence,
                "notes": None,
            }
        )
    )


def _retrieval(
    ranked_text: list[tuple[int, str]],
    *,
    status: OperationalStatus = OperationalStatus.SUCCESS,
    requested_top_k: int = 5,
) -> RetrievalObservation:
    results = tuple(
        RetrievedResult(
            rank=rank,
            document_id=1,
            chunk_id=rank,
            chunk_order=rank - 1,
            distance=rank / 100,
            chunk_text=text,
        )
        for rank, text in ranked_text
    )
    return RetrievalObservation(
        status=status,
        latency_ms=1.0 if status is OperationalStatus.SUCCESS else None,
        requested_top_k=requested_top_k,
        candidate_count=len(results),
        result_count=len(results),
        embedding_model="synthetic",
        embedding_dimension=8,
        results=results,
        error=None,
    )


def _result(
    record: BenchmarkRecord,
    retrieval: RetrievalObservation | None,
    *,
    mismatch: bool | None = None,
) -> RawQuestionResult:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RawQuestionResult(
        run_id="run-synthetic",
        experiment_id="exp_synthetic",
        question_id=record.id,
        category=record.category,
        answerability=record.answerability,
        started_at=now,
        completed_at=now,
        overall_status=(
            retrieval.status if retrieval is not None else OperationalStatus.RETRIEVAL_FAILURE
        ),
        retrieval=retrieval,
        generation=None,
        retrieval_observation_mismatch=mismatch,
    )


def test_exact_anchor_present_and_absent() -> None:
    record = _record("The exact required anchor")
    present = calculate_question_metrics(
        record, _result(record, _retrieval([(1, "Before the exact required anchor after")]))
    )
    absent = calculate_question_metrics(
        record, _result(record, _retrieval([(1, "Unrelated text")]))
    )
    assert (present.hit_at_k, present.recall_at_k, present.complete_evidence_at_k) == (
        True,
        1.0,
        True,
    )
    assert (absent.hit_at_k, absent.recall_at_k, absent.complete_evidence_at_k) == (
        False,
        0.0,
        False,
    )


def test_anchor_normalization_uses_nfkc_casefold_and_whitespace() -> None:
    record = _record("ＣＡＦÉ   HOURS")
    result = _result(record, _retrieval([(1, "The cafe\u0301\n\tHours are listed.")]))
    assert calculate_question_metrics(record, result).hit_at_k is True


def test_duplicate_chunk_matches_count_evidence_once_and_partial_recall() -> None:
    record = _record("first anchor", "second anchor")
    result = _result(
        record,
        _retrieval([(1, "first anchor"), (2, "first anchor repeated")]),
    )
    metrics = calculate_question_metrics(record, result)
    assert metrics.hit_at_k is True
    assert metrics.recall_at_k == 0.5
    assert metrics.complete_evidence_at_k is False


def test_top_k_uses_explicit_rank_not_insertion_order() -> None:
    record = _record("gold anchor")
    retrieval = _retrieval([(2, "gold anchor"), (1, "not gold")], requested_top_k=2)
    metrics = calculate_question_metrics(record, _result(record, retrieval), k=1)
    assert metrics.hit_at_k is False


def test_fewer_than_k_results_are_scored() -> None:
    record = _record("gold anchor")
    retrieval = _retrieval([(1, "gold anchor")], requested_top_k=5)
    assert calculate_question_metrics(record, _result(record, retrieval)).recall_at_k == 1.0


def test_successful_empty_retrieval_is_zero_not_null() -> None:
    record = _record("gold anchor")
    metrics = calculate_question_metrics(record, _result(record, _retrieval([])))
    assert metrics.hit_at_k is False
    assert metrics.recall_at_k == 0.0
    assert metrics.complete_evidence_at_k is False


def test_unanswerable_retrieval_metrics_are_null() -> None:
    record = _record(answerability="unanswerable")
    metrics = calculate_question_metrics(record, _result(record, _retrieval([(1, "anything")])))
    assert metrics.hit_at_k is None
    assert metrics.recall_at_k is None
    assert metrics.complete_evidence_at_k is None


def test_retrieval_failure_metrics_are_null() -> None:
    record = _record("gold anchor")
    failed = _retrieval([], status=OperationalStatus.RETRIEVAL_FAILURE)
    metrics = calculate_question_metrics(record, _result(record, failed))
    assert metrics.hit_at_k is None
    assert metrics.recall_at_k is None
    assert metrics.complete_evidence_at_k is None


def test_mismatch_true_does_not_change_retrieval_metrics() -> None:
    record = _record("gold anchor")
    retrieval = _retrieval([(1, "gold anchor")])
    matched = calculate_question_metrics(record, _result(record, retrieval, mismatch=False))
    mismatched = calculate_question_metrics(record, _result(record, retrieval, mismatch=True))
    assert mismatched.hit_at_k == matched.hit_at_k is True
    assert mismatched.recall_at_k == matched.recall_at_k == 1.0
    assert mismatched.complete_evidence_at_k == matched.complete_evidence_at_k is True


def test_anchor_split_across_chunks_does_not_match() -> None:
    record = _record("complete anchor text")
    retrieval = _retrieval([(1, "complete anchor"), (2, "text continues")])
    assert calculate_question_metrics(record, _result(record, retrieval)).hit_at_k is False


def test_ambiguous_question_retrieval_is_scoreable() -> None:
    record = _record("qualified source", answerability="ambiguous")
    metrics = calculate_question_metrics(
        record, _result(record, _retrieval([(1, "qualified source")]))
    )
    assert metrics.hit_at_k is True


def test_nonpositive_k_is_rejected() -> None:
    record = _record("gold anchor")
    with pytest.raises(ValueError, match="K must be greater than zero"):
        calculate_question_metrics(record, _result(record, _retrieval([])), k=0)
