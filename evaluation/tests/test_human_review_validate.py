"""Lifecycle and completeness tests for blinded review responses."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evaluation.human_review import validate_review_batch
from evaluation.schemas import (
    CorrectnessLabel,
    GroundingLabel,
    RefusalLabel,
    RelevanceLabel,
    RetrievalRelevanceAnnotation,
)
from evaluation.tests.review_test_support import (
    completed_response,
    create_batch,
    read_packets,
    read_responses,
    write_responses,
)


def test_fully_unreviewed_batch(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    report = validate_review_batch(batch)
    assert report.batch_state == "unreviewed"
    assert report.unreviewed_count == 3
    assert report.partial_count == report.complete_count == report.invalid_count == 0
    assert (batch / "validation.json").is_file()


def test_partially_reviewed_item(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    responses = read_responses(batch)
    responses[0] = responses[0].model_copy(
        update={"correctness": CorrectnessLabel.CORRECT}
    )
    write_responses(batch, responses)
    report = validate_review_batch(batch)
    assert report.batch_state == "partial"
    assert report.partial_count == 1
    assert "missing_grounding" in report.items[0].issues


def test_completely_reviewed_item_and_all_complete_batch(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    packets = read_packets(batch)
    responses = [
        completed_response(packet, response)
        for packet, response in zip(packets, read_responses(batch))
    ]
    write_responses(batch, responses)
    report = validate_review_batch(batch)
    assert report.batch_state == "complete"
    assert report.complete_count == 3
    assert report.unreviewed_count == report.partial_count == report.invalid_count == 0


def test_mixed_batch_is_partial(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    packets = read_packets(batch)
    responses = read_responses(batch)
    responses[0] = completed_response(packets[0], responses[0])
    write_responses(batch, responses)
    report = validate_review_batch(batch)
    assert report.batch_state == "partial"
    assert report.complete_count == 1
    assert report.unreviewed_count == 2


def test_missing_response_is_reported(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    responses = read_responses(batch)
    missing_id = responses[-1].review_item_id
    write_responses(batch, responses[:-1])
    report = validate_review_batch(batch)
    assert report.batch_state == "partial"
    assert report.missing_response_ids == (missing_id,)
    assert report.invalid_count >= 1


def test_extra_response_is_reported(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    responses = read_responses(batch)
    responses.append(responses[0].model_copy(update={"review_item_id": "unknown_item"}))
    write_responses(batch, responses)
    report = validate_review_batch(batch)
    assert report.extra_response_ids == ("unknown_item",)
    assert report.batch_state == "partial"


def test_duplicate_response_id_is_reported(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    responses = read_responses(batch)
    responses.append(responses[0])
    write_responses(batch, responses)
    report = validate_review_batch(batch)
    assert report.duplicate_response_ids == (responses[0].review_item_id,)
    assert report.batch_state == "partial"


def test_missing_annotated_at_is_partial(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    complete = completed_response(packet, read_responses(batch)[0])
    write_responses(batch, [complete.model_copy(update={"annotated_at": None})])
    report = validate_review_batch(batch)
    assert report.items[0].state == "partial"
    assert "missing_annotated_at" in report.items[0].issues


def test_missing_retrieval_rank_is_invalid_and_incomplete(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    complete = completed_response(packet, read_responses(batch)[0])
    write_responses(batch, [complete.model_copy(update={"retrieval_relevance": ()})])
    report = validate_review_batch(batch)
    assert report.items[0].state == "partial"
    assert report.items[0].invalid is True
    assert report.incomplete_retrieval_ranks[packet.review_item_id] == (1,)


def test_extra_retrieval_rank_is_invalid(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    complete = completed_response(packet, read_responses(batch)[0])
    relevance = complete.retrieval_relevance + (
        RetrievalRelevanceAnnotation(rank=99, relevance=RelevanceLabel.RELEVANT),
    )
    write_responses(
        batch,
        [complete.model_copy(update={"retrieval_relevance": relevance})],
    )
    report = validate_review_batch(batch)
    assert report.items[0].invalid is True
    assert any(issue.startswith("extra_retrieval_ranks") for issue in report.items[0].issues)


def test_not_reviewed_retrieval_rank_keeps_item_partial(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    response = completed_response(packet, read_responses(batch)[0])
    pending = response.retrieval_relevance[0].model_copy(
        update={"relevance": RelevanceLabel.NOT_REVIEWED}
    )
    write_responses(
        batch,
        [response.model_copy(update={"retrieval_relevance": (pending,)})],
    )
    report = validate_review_batch(batch)
    assert report.items[0].state == "partial"
    assert "retrieval_relevance_not_reviewed" in report.items[0].issues


def test_explicit_not_applicable_counts_as_reviewed(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    response = completed_response(packet, read_responses(batch)[0]).model_copy(
        update={
            "correctness": CorrectnessLabel.NOT_APPLICABLE,
            "grounding": GroundingLabel.NOT_APPLICABLE,
            "refusal": RefusalLabel.NOT_APPLICABLE,
        }
    )
    write_responses(batch, [response])
    assert validate_review_batch(batch).batch_state == "complete"


def test_unassessable_grounding_requires_not_applicable_and_note(tmp_path: Path) -> None:
    batch, _, _ = create_batch(
        tmp_path,
        question_ids=("Q020",),
        missing_source_questions=frozenset({"Q020"}),
    )
    packet = read_packets(batch)[0]
    response = completed_response(packet, read_responses(batch)[0])

    without_note = response.model_copy(update={"notes": None})
    write_responses(batch, [without_note])
    report = validate_review_batch(batch)
    assert report.items[0].state == "partial"
    assert "grounding_unassessable_note_required" in report.items[0].issues

    non_na = response.model_copy(
        update={"grounding": GroundingLabel.GROUNDED, "notes": "Still unavailable."}
    )
    write_responses(batch, [non_na])
    report = validate_review_batch(batch)
    assert report.items[0].invalid is True
    assert "grounding_must_be_not_applicable" in report.items[0].issues

    write_responses(batch, [response])
    assert validate_review_batch(batch).batch_state == "complete"


def test_expected_count_is_optional_and_enforceable(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    assert validate_review_batch(batch, expected_count=3).invalid_count == 0
    report = validate_review_batch(batch, expected_count=40)
    assert report.batch_state == "partial"
    assert report.invalid_count == 1


def test_activity_without_semantic_labels_is_partial(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    response = read_responses(batch)[0].model_copy(
        update={
            "reviewer_id": "synthetic-reviewer",
            "annotated_at": datetime(2026, 1, 3, tzinfo=UTC),
        }
    )
    write_responses(batch, [response])
    assert validate_review_batch(batch).batch_state == "partial"
