"""Focused E5 tests for immutable experiment configuration identity."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from evaluation.schemas import (
    ExperimentConfig,
    RunManifest,
    RunStatus,
    WarmupStatus,
    canonical_config_json,
    configuration_hash,
    experiment_id,
)


def _config_data() -> dict[str, object]:
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    return {
        "schema_version": "1.0.0",
        "corpus": {"corpus_id": "fixture_corpus", "corpus_sha256": digest_a},
        "benchmark": {
            "benchmark_id": "fixture_benchmark",
            "benchmark_version": "test-only",
            "benchmark_sha256": digest_b,
        },
        "generation": {"generation_model": "fixture-generation-model"},
        "embedding": {
            "embedding_model": "fixture-embedding-model",
            "embedding_dimension": 8,
        },
        "prompt": {"prompt_id": "fixture-prompt", "prompt_sha256": digest_c},
        "chunking": {
            "chunk_size_chars": 100,
            "chunk_overlap_chars": 10,
            "chunking_version": "fixture-v1",
        },
        "retrieval": {"top_k": 3, "retrieval_strategy": "fixture-vector-distance"},
        "sampling": {
            "mode": "runtime_default",
            "temperature": None,
            "top_p": None,
            "seed": None,
        },
        "timeouts": {
            "request_timeout_seconds": None,
            "retrieval_timeout_seconds": 5.0,
            "generation_timeout_seconds": 10.0,
        },
        "warmup": {"required": True},
    }


def _config(data: dict[str, object] | None = None) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(json.dumps(data or _config_data()))


def test_configuration_hash_is_stable() -> None:
    config = _config()
    assert configuration_hash(config) == configuration_hash(config)
    assert canonical_config_json(config) == canonical_config_json(config)
    assert experiment_id(config) == f"exp_{configuration_hash(config)[:12]}"


def test_key_order_does_not_change_hash() -> None:
    original = _config_data()
    reordered = dict(reversed(list(original.items())))
    assert configuration_hash(_config(original)) == configuration_hash(_config(reordered))


def test_meaningful_configuration_change_changes_hash() -> None:
    original = _config_data()
    changed = json.loads(json.dumps(original))
    changed["retrieval"]["top_k"] = 4
    assert configuration_hash(_config(original)) != configuration_hash(_config(changed))


def test_runtime_metadata_does_not_affect_configuration_hash() -> None:
    config = _config()
    digest = configuration_hash(config)
    common = {
        "schema_version": "1.0.0",
        "run_id": "run-fixture",
        "experiment_id": experiment_id(config),
        "config_hash": digest,
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "completed_at": None,
        "status": RunStatus.RUNNING,
        "corpus": config.corpus,
        "benchmark": config.benchmark,
        "warmup_status": WarmupStatus.PENDING,
        "question_total": 40,
        "success_total": 0,
        "failure_total": 0,
    }
    first = RunManifest(**common, git_commit_sha="1" * 40, git_dirty=False, platform="one")
    second = RunManifest(**common, git_commit_sha="2" * 40, git_dirty=True, platform="two")
    assert first.config_hash == second.config_hash == configuration_hash(config)
