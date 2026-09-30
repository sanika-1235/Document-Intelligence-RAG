"""Focused command-contract tests for offline human-review tooling."""

from __future__ import annotations

from pathlib import Path

from evaluation.human_review import main
from evaluation.tests.review_test_support import (
    completed_response,
    create_run,
    read_packets,
    read_responses,
    write_benchmark,
    write_responses,
)


def test_prepare_and_validate_cli_expectations(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    run = create_run(tmp_path, benchmark, "run_synthetic_1")
    batch = tmp_path / "cli_batch"
    assert main(
        [
            "prepare",
            "--run",
            str(run),
            "--benchmark",
            str(benchmark),
            "--seed",
            "cli-seed",
            "--output",
            str(batch),
        ]
    ) == 0
    assert main(
        [
            "validate",
            "--review-batch",
            str(batch),
            "--expect-state",
            "unreviewed",
            "--expected-count",
            "3",
        ]
    ) == 0
    assert main(
        [
            "validate",
            "--review-batch",
            str(batch),
            "--expect-state",
            "complete",
        ]
    ) == 1


def test_unblind_cli_requires_complete_review(tmp_path: Path) -> None:
    benchmark = write_benchmark(tmp_path)
    run = create_run(tmp_path, benchmark, "run_synthetic_1")
    batch = tmp_path / "cli_batch"
    assert main(
        [
            "prepare",
            "--run",
            str(run),
            "--benchmark",
            str(benchmark),
            "--seed",
            "cli-seed",
            "--output",
            str(batch),
        ]
    ) == 0
    assert main(["unblind", "--review-batch", str(batch)]) == 1

    packets = read_packets(batch)
    responses = [
        completed_response(packet, response)
        for packet, response in zip(packets, read_responses(batch))
    ]
    write_responses(batch, responses)
    assert main(["unblind", "--review-batch", str(batch)]) == 0
    assert (batch / "unblinded_annotations.jsonl").is_file()
    assert not (run / "human_annotations.jsonl").exists()
