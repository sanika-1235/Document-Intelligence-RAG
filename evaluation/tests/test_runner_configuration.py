"""E6A tests for safe configuration and identity resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.prompting as prompting
from app.prompting import (
    ANSWER_COMPLETENESS_PROMPT_ID,
    PRODUCTION_PROMPT_ID,
    PromptDefinition,
)
from evaluation.runner import (
    RETRIEVAL_STRATEGY,
    RunnerConfigurationError,
    generate_run_id,
    resolve_experiment_config,
    resolve_prompt_identity,
    sha256_file,
)
from evaluation.schemas import SamplingMode, configuration_hash, experiment_id

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "evaluation/benchmarks/csu_handbook_v1.jsonl"
MANIFEST = ROOT / "evaluation/corpus/corpus_manifest.json"

pytestmark = pytest.mark.skipif(
    not BENCHMARK.is_file(),
    reason="requires authorized local CSU benchmark artifact",
)


def _settings(prompt_id: str = PRODUCTION_PROMPT_ID) -> SimpleNamespace:
    return SimpleNamespace(
        generation_model="resolved-generation",
        embedding_model="resolved-embedding",
        embedding_dimension=321,
        chunk_size_chars=777,
        chunk_overlap_chars=77,
        chunking_version="resolved-chunking",
        retrieval_default_top_k=9,
        generation_prompt_id=prompt_id,
        jwt_secret="must-not-be-serialized",
    )


def test_configuration_uses_resolved_values_and_exact_benchmark_hash() -> None:
    config = resolve_experiment_config(
        app_settings=_settings(),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
    )
    assert config.benchmark.benchmark_sha256 == sha256_file(BENCHMARK)
    assert config.generation.generation_model == "resolved-generation"
    assert config.embedding.embedding_model == "resolved-embedding"
    assert config.embedding.embedding_dimension == 321
    assert config.chunking.chunk_size_chars == 777
    assert config.chunking.chunk_overlap_chars == 77
    assert config.chunking.chunking_version == "resolved-chunking"
    assert config.retrieval.top_k == 9
    assert config.retrieval.retrieval_strategy == RETRIEVAL_STRATEGY
    assert config.prompt.prompt_id == PRODUCTION_PROMPT_ID
    assert config.sampling.mode is SamplingMode.RUNTIME_DEFAULT
    assert config.sampling.temperature is None
    assert config.sampling.top_p is None
    assert config.sampling.seed is None
    assert config.timeouts.request_timeout_seconds == 42.0


def test_top_k_override_and_experiment_identity_are_stable() -> None:
    first = resolve_experiment_config(
        app_settings=_settings(),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
        top_k_override=4,
        warmup_required=True,
    )
    second = resolve_experiment_config(
        app_settings=_settings(),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
        top_k_override=4,
        warmup_required=True,
    )
    assert first.retrieval.top_k == 4
    assert configuration_hash(first) == configuration_hash(second)
    assert experiment_id(first) == experiment_id(second)


def test_configuration_contains_no_credentials() -> None:
    config = resolve_experiment_config(
        app_settings=_settings(),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
    )
    serialized = config.model_dump_json()
    assert "must-not-be-serialized" not in serialized
    assert "password" not in serialized.casefold()
    assert "access_token" not in serialized.casefold()
    assert "authorization" not in serialized.casefold()


def test_prompt_hash_excludes_runtime_question_and_context() -> None:
    config = resolve_experiment_config(
        app_settings=_settings(),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
    )
    serialized = json.loads(config.model_dump_json())
    assert "runtime question" not in json.dumps(serialized)
    assert len(config.prompt.prompt_sha256) == 64


def test_run_id_is_sortable_safe_and_independent_from_experiment() -> None:
    run_id = generate_run_id(random_hex="deadbeef")
    assert re.fullmatch(r"run_\d{8}T\d{12}Z_deadbeef", run_id)
    assert not run_id.startswith("exp_")


def test_runner_uses_backend_prompt_identity_resolver() -> None:
    backend_identity = prompting.resolve_prompt_identity(PRODUCTION_PROMPT_ID)
    runner_identity = resolve_prompt_identity(PRODUCTION_PROMPT_ID)

    assert runner_identity.prompt_id == backend_identity.prompt_id
    assert runner_identity.prompt_sha256 == backend_identity.prompt_sha256


def test_runner_can_snapshot_test_only_registered_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_id = "test-only-prompt"
    monkeypatch.setitem(
        prompting._PROMPT_REGISTRY,
        prompt_id,
        PromptDefinition(prompt_id=prompt_id, system_prompt="Test-only system prompt."),
    )

    config = resolve_experiment_config(
        app_settings=_settings(prompt_id),
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=42.0,
    )
    backend_identity = prompting.resolve_prompt_identity(prompt_id)

    assert config.prompt.prompt_id == prompt_id
    assert config.prompt.prompt_sha256 == backend_identity.prompt_sha256


def test_runner_rejects_unknown_prompt_before_run_creation() -> None:
    with pytest.raises(RunnerConfigurationError, match="Unknown internal generation prompt ID"):
        resolve_experiment_config(
            app_settings=_settings("unknown-prompt"),
            benchmark_path=BENCHMARK,
            manifest_path=MANIFEST,
            request_timeout_seconds=42.0,
        )


def test_canonical_production_configuration_identity_is_unchanged() -> None:
    settings = _settings()
    settings.generation_model = "qwen2.5:1.5b"
    settings.embedding_model = "qwen3-embedding:0.6b"
    settings.embedding_dimension = 1024
    settings.chunk_size_chars = 1000
    settings.chunk_overlap_chars = 200
    settings.chunking_version = "ddrag-chunking-v1"
    settings.retrieval_default_top_k = 5

    config = resolve_experiment_config(
        app_settings=settings,
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        request_timeout_seconds=90.0,
        warmup_required=True,
    )

    assert config.prompt.prompt_id == PRODUCTION_PROMPT_ID
    assert config.prompt.prompt_sha256 == "425edd4569bc82ba8aa292c842846ae65c186abf2c622b5cdbc3af98a24d0462"
    assert experiment_id(config) == "exp_eb14097ede63"
    assert configuration_hash(config) == "eb14097ede63989684c2c4ff06268985318ab580784658c0c2188b6acc979e8d"


def test_canonical_candidate_configuration_differs_only_by_prompt_identity() -> None:
    control_settings = _settings(PRODUCTION_PROMPT_ID)
    candidate_settings = _settings(ANSWER_COMPLETENESS_PROMPT_ID)
    for settings in (control_settings, candidate_settings):
        settings.generation_model = "qwen2.5:1.5b"
        settings.embedding_model = "qwen3-embedding:0.6b"
        settings.embedding_dimension = 1024
        settings.chunk_size_chars = 1000
        settings.chunk_overlap_chars = 200
        settings.chunking_version = "ddrag-chunking-v1"
        settings.retrieval_default_top_k = 5

    common = {
        "benchmark_path": BENCHMARK,
        "manifest_path": MANIFEST,
        "request_timeout_seconds": 90.0,
        "warmup_required": True,
    }
    control = resolve_experiment_config(app_settings=control_settings, **common)
    candidate = resolve_experiment_config(app_settings=candidate_settings, **common)
    control_data = control.model_dump(mode="json")
    candidate_data = candidate.model_dump(mode="json")

    assert [key for key in control_data if control_data[key] != candidate_data[key]] == ["prompt"]
    assert {
        key
        for key in control_data["prompt"]
        if control_data["prompt"][key] != candidate_data["prompt"][key]
    } == {"prompt_id", "prompt_sha256"}
    assert candidate.prompt.prompt_id == ANSWER_COMPLETENESS_PROMPT_ID
    assert candidate.prompt.prompt_sha256 == "17657d3e30a781fe44267c07e1867c2fac55791b34b50098918379fe98681771"
    assert experiment_id(candidate) == "exp_e8b36f081c18"
    assert configuration_hash(candidate) == "e8b36f081c181d625924a6edd3d17916d6bdc160060ed7f10555d6d1c3b8b434"
