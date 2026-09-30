"""Integrity, identity restoration, and safe-write tests for unblinding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.human_review import (
    HumanReviewError,
    ReviewMapping,
    canonical_sha256,
    sha256_file,
    unblind_review_batch,
)
from evaluation.schemas import HumanAnnotation
from evaluation.tests.review_test_support import (
    completed_response,
    create_batch,
    read_packets,
    read_responses,
    write_responses,
)


def _complete_batch(batch: Path) -> None:
    packets = read_packets(batch)
    responses = [
        completed_response(packet, response)
        for packet, response in zip(packets, read_responses(batch))
    ]
    write_responses(batch, responses)


def test_unblind_restores_exact_identity_and_strict_annotations(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    _complete_batch(batch)
    annotations = unblind_review_batch(batch)
    mapping = ReviewMapping.model_validate_json(
        (batch / "mapping.json").read_text(encoding="utf-8")
    )
    expected = {
        (item.run_id, item.experiment_id, item.question_id) for item in mapping.items
    }
    actual = {
        (item.run_id, item.experiment_id, item.question_id) for item in annotations
    }
    assert actual == expected
    assert all(isinstance(item, HumanAnnotation) for item in annotations)
    assert (batch / "unblinded_annotations.jsonl").is_file()


def test_default_unblind_writes_preview_only(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path)
    _complete_batch(batch)
    unblind_review_batch(batch)
    assert (batch / "unblinded_annotations.jsonl").is_file()
    assert all(not (run / "human_annotations.jsonl").exists() for run in runs)


def test_explicit_run_write_groups_annotations_and_preserves_raw_results(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path, run_count=2)
    _complete_batch(batch)
    raw_hashes = {run: sha256_file(run / "raw_results.jsonl") for run in runs}
    annotations = unblind_review_batch(batch, write_run_annotations=True)
    assert len(annotations) == 6
    for run in runs:
        target = run / "human_annotations.jsonl"
        parsed = [
            HumanAnnotation.model_validate_json(line)
            for line in target.read_text(encoding="utf-8").splitlines()
        ]
        assert len(parsed) == 3
        assert {item.run_id for item in parsed} == {run.name}
        assert sha256_file(run / "raw_results.jsonl") == raw_hashes[run]


def test_existing_run_annotation_target_is_not_overwritten(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path)
    _complete_batch(batch)
    target = runs[0] / "human_annotations.jsonl"
    target.write_text("existing\n", encoding="utf-8")
    with pytest.raises(HumanReviewError, match="already exists"):
        unblind_review_batch(batch, write_run_annotations=True)
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_packet_tampering_is_rejected(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    _complete_batch(batch)
    packet_path = batch / "packet.jsonl"
    lines = packet_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["generated_answer"] = "tampered"
    lines[0] = json.dumps(payload)
    packet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(HumanReviewError, match="packet integrity"):
        unblind_review_batch(batch)


def test_mapping_tampering_is_rejected(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    _complete_batch(batch)
    path = batch / "mapping.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["run_id"] = "tampered-run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HumanReviewError, match="mapping integrity"):
        unblind_review_batch(batch)


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    _complete_batch(batch)
    path = batch / "review_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = "tampered-seed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HumanReviewError, match="manifest integrity"):
        unblind_review_batch(batch)


def test_unknown_review_item_is_rejected(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    packets = read_packets(batch)
    responses = [
        completed_response(packet, response)
        for packet, response in zip(packets, read_responses(batch))
    ]
    responses.append(responses[0].model_copy(update={"review_item_id": "unknown_item"}))
    write_responses(batch, responses)
    with pytest.raises(HumanReviewError, match="complete valid review batch"):
        unblind_review_batch(batch)


def test_incomplete_review_is_rejected_without_writes(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path)
    with pytest.raises(HumanReviewError, match="complete valid review batch"):
        unblind_review_batch(batch)
    assert not (batch / "unblinded_annotations.jsonl").exists()
    assert all(not (run / "human_annotations.jsonl").exists() for run in runs)


def test_preview_cannot_be_redirected_into_a_run(tmp_path: Path) -> None:
    batch, _, runs = create_batch(tmp_path)
    _complete_batch(batch)
    with pytest.raises(HumanReviewError, match="remain inside"):
        unblind_review_batch(
            batch,
            preview_path=runs[0] / "human_annotations.jsonl",
        )
    assert not (runs[0] / "human_annotations.jsonl").exists()


def test_mapping_digest_is_canonical_and_stable(tmp_path: Path) -> None:
    batch, _, _ = create_batch(tmp_path)
    mapping = ReviewMapping.model_validate_json(
        (batch / "mapping.json").read_text(encoding="utf-8")
    )
    reordered_text = json.dumps(
        mapping.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=False,
        indent=4,
    )
    reparsed = ReviewMapping.model_validate_json(reordered_text)
    assert canonical_sha256(mapping) == canonical_sha256(reparsed)
