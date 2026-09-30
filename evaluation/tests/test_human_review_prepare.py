"""Preparation, ordering, source-resolution, and blinding tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.human_review import HumanReviewError, prepare_review_batch
from evaluation.tests.review_test_support import (
    create_batch,
    create_run,
    read_packets,
    read_responses,
    write_benchmark,
)


def _packet_text(batch: Path) -> str:
    return (batch / "packet.jsonl").read_text(encoding="utf-8")


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_prepare_one_synthetic_run_and_preserve_answer_and_scope_note(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    packets = read_packets(batch)
    assert len(packets) == 3
    q020 = next(item for item in packets if item.question_id == "Q020")
    assert q020.generated_answer == "Exact observed answer variant 1 for Q020.\n[1]"
    assert q020.benchmark_notes == (
        "Synthetic scoring-scope note; excluded clauses must not be restored."
    )
    assert q020.expected_points[0].model_dump() == {
        "point_id": "P1",
        "text": "Alpha condition",
    }
    assert q020.gold_evidence[0].anchor == "The policy requires the alpha condition."


def test_prepare_multiple_runs_preserves_every_unique_item_once(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, run_count=2)
    packets = read_packets(batch)
    assert len(packets) == 6
    assert len({item.review_item_id for item in packets}) == 6
    assert sorted(item.review_item_id for item in packets) == [
        f"review_item_{index:06d}" for index in range(1, 7)
    ]
    answers = [item.generated_answer for item in packets]
    assert len(answers) == len(set(answers))


def test_same_seed_and_inputs_produce_identical_packet(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    runs = [create_run(tmp_path, benchmark, f"run_synthetic_{index}") for index in (1, 2)]
    first = tmp_path / "batch_first"
    second = tmp_path / "batch_second"
    for output in (first, second):
        prepare_review_batch(
            run_dirs=runs,
            benchmark_path=benchmark,
            seed="repeatable-seed",
            output_dir=output,
        )
    assert _packet_text(first) == _packet_text(second)


def test_different_seed_changes_order_when_possible(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    runs = [create_run(tmp_path, benchmark, f"run_synthetic_{index}") for index in (1, 2, 3)]
    first = tmp_path / "batch_first"
    second = tmp_path / "batch_second"
    prepare_review_batch(
        run_dirs=runs, benchmark_path=benchmark, seed="seed-a", output_dir=first
    )
    prepare_review_batch(
        run_dirs=runs, benchmark_path=benchmark, seed="seed-b", output_dir=second
    )
    first_answers = [item.generated_answer for item in read_packets(first)]
    second_answers = [item.generated_answer for item in read_packets(second)]
    assert first_answers != second_answers


def test_duplicate_raw_identity_is_rejected(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    run = create_run(tmp_path, benchmark, "run_synthetic_1")
    raw_path = run / "raw_results.jsonl"
    first_line = raw_path.read_text(encoding="utf-8").splitlines()[0]
    with raw_path.open("a", encoding="utf-8") as stream:
        stream.write(first_line + "\n")
    with pytest.raises(HumanReviewError, match="Duplicate raw identities"):
        prepare_review_batch(
            run_dirs=[run],
            benchmark_path=benchmark,
            seed="seed",
            output_dir=tmp_path / "batch",
        )


def test_same_question_items_are_separated_when_possible(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, run_count=3)
    questions = [item.question_id for item in read_packets(batch)]
    assert all(left != right for left, right in zip(questions, questions[1:]))


def test_impossible_same_question_separation_is_deterministic(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    runs = [
        create_run(
            tmp_path,
            benchmark,
            f"run_synthetic_{index}",
            question_ids=("Q020",),
        )
        for index in (1, 2, 3)
    ]
    first = tmp_path / "batch_first"
    second = tmp_path / "batch_second"
    for output in (first, second):
        prepare_review_batch(
            run_dirs=runs,
            benchmark_path=benchmark,
            seed="same-question-seed",
            output_dir=output,
        )
    assert _packet_text(first) == _packet_text(second)
    assert [item.question_id for item in read_packets(first)] == ["Q020"] * 3


def test_exact_generation_source_identity_exposes_exact_text(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path, question_ids=("Q020",))
    packet = read_packets(batch)[0]
    assert packet.generation_sources[0].source_text_available is True
    assert packet.generation_sources[0].source_text == "Exact retrieved text for Q020."
    assert packet.grounding_assessable is True


def test_missing_generation_source_is_not_substituted(tmp_path: Path) -> None:
    batch, _, _ = create_batch(
        tmp_path,
        question_ids=("Q020",),
        missing_source_questions=frozenset({"Q020"}),
    )
    packet = read_packets(batch)[0]
    assert packet.generation_sources[0].source_text_available is False
    assert packet.generation_sources[0].source_text is None
    assert packet.grounding_assessable is False
    serialized = json.loads(_packet_text(batch))
    assert "retrieval_observation_mismatch" not in _all_keys(serialized)


def test_absent_generation_sources_make_grounding_unassessable(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    run_dir = create_run(
        tmp_path,
        benchmark,
        "run_no_sources",
        question_ids=("Q020",),
        no_source_questions=frozenset({"Q020"}),
    )
    batch = tmp_path / "batch"
    prepare_review_batch(
        run_dirs=(run_dir,),
        benchmark_path=benchmark,
        seed="no-source-seed",
        output_dir=batch,
    )

    packet = read_packets(batch)[0]
    assert packet.generation_sources == ()
    assert packet.grounding_assessable is False


def test_response_template_contains_no_human_prelabels_even_for_failure(tmp_path: Path) -> None:
    batch, _, _ = create_batch(
        tmp_path,
        question_ids=("Q020",),
        failed_questions=frozenset({"Q020"}),
    )
    response = read_responses(batch)[0]
    assert response.reviewer_id is None
    assert response.annotated_at is None
    assert response.correctness is None
    assert response.grounding is None
    assert response.refusal is None
    assert all(item.relevance.value == "not_reviewed" for item in response.retrieval_relevance)
    assert response.failure_tags == ()
    assert response.notes is None


def test_packet_has_no_hidden_fields_or_secret_sentinels(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path, question_ids=("Q020",))
    text = _packet_text(batch)
    payload = json.loads(text)
    keys = _all_keys(payload)
    forbidden_keys = {
        "run_id",
        "experiment_id",
        "config_hash",
        "generation_model",
        "embedding_model",
        "latency_ms",
        "distance",
        "document_id",
        "chunk_id",
        "automated_match",
        "hit_at_k",
        "recall_at_k",
        "complete_evidence_at_k",
        "expected_point_coverage",
        "automated_refusal_signal",
        "citation_validation",
        "valid",
        "invalid_citations",
        "retrieval_observation_mismatch",
        "error",
    }
    assert not (keys & forbidden_keys)
    sentinels = [
        "SECRET_MODEL_SENTINEL",
        "SECRET_EMBEDDING_SENTINEL",
        "SECRET_PASSWORD_SENTINEL",
        "eyJSECRET.JWT.SENTINEL",
        "secret@example.invalid",
        "Authorization",
        runs[0].name,
    ]
    assert all(sentinel not in text for sentinel in sentinels)


def test_raw_error_messages_and_secret_like_values_never_reach_packet(tmp_path: Path) -> None:
    batch, _, _ = create_batch(
        tmp_path,
        question_ids=("Q020",),
        failed_questions=frozenset({"Q020"}),
    )
    text = _packet_text(batch)
    assert "RAW_ERROR_PASSWORD" not in text
    assert "eyJRAW.ERROR.TOKEN" not in text
    assert "raw-error@example.invalid" not in text
    assert "RAW_ERROR_SECRET" not in text
    packet = read_packets(batch)[0]
    assert packet.operational_status.failure_category == "generation_failure"
