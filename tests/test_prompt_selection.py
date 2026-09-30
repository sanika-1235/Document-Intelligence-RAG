"""Focused tests for private allowlisted generation-prompt selection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.prompting as prompting
from app.answering import answer_question
from app.api.generation import GenerationRequest, GenerationResponse
from app.config import Settings
from app.prompting import (
    ANSWER_COMPLETENESS_PROMPT,
    ANSWER_COMPLETENESS_PROMPT_ID,
    PRODUCTION_PROMPT_ID,
    PromptDefinition,
    PromptResolutionError,
    SYSTEM_PROMPT,
    build_system_prompt,
    resolve_prompt_identity,
)
from app.retrieval import RetrievalResult

PRODUCTION_PROMPT_SHA256 = "425edd4569bc82ba8aa292c842846ae65c186abf2c622b5cdbc3af98a24d0462"
PRODUCTION_SYSTEM_PROMPT_SHA256 = "ef33d5370f8ae4bc2a5f9b86879e316adde3e71c8fd8a7a84a597d6bd0857b91"
CANDIDATE_PROMPT_SHA256 = "17657d3e30a781fe44267c07e1867c2fac55791b34b50098918379fe98681771"
CANDIDATE_RAW_TEXT_SHA256 = "16353d5d53834c0d4b419874a30f7018bf85f5e7170eec439cd4c8ae5267d274"
CANDIDATE_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "evaluation/prompts/ddrag-answer-completeness-v1.txt"
)


def test_missing_prompt_setting_uses_production_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENERATION_PROMPT_ID", raising=False)

    settings = Settings(_env_file=None, jwt_secret="test-only-secret")

    assert settings.generation_prompt_id == PRODUCTION_PROMPT_ID


def test_explicit_production_prompt_setting_resolves_identically() -> None:
    settings = Settings(
        _env_file=None,
        jwt_secret="test-only-secret",
        generation_prompt_id=PRODUCTION_PROMPT_ID,
    )

    assert build_system_prompt(settings.generation_prompt_id) == SYSTEM_PROMPT
    assert resolve_prompt_identity(settings.generation_prompt_id).prompt_sha256 == PRODUCTION_PROMPT_SHA256


def test_registry_contains_exactly_two_frozen_runtime_prompts() -> None:
    assert list(prompting._PROMPT_REGISTRY) == [
        PRODUCTION_PROMPT_ID,
        ANSWER_COMPLETENESS_PROMPT_ID,
    ]
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == PRODUCTION_SYSTEM_PROMPT_SHA256
    assert resolve_prompt_identity(PRODUCTION_PROMPT_ID).prompt_sha256 == PRODUCTION_PROMPT_SHA256


def test_candidate_file_matches_registry_text_and_frozen_hashes() -> None:
    candidate_bytes = CANDIDATE_PROMPT_FILE.read_bytes()

    assert candidate_bytes.decode("utf-8") == ANSWER_COMPLETENESS_PROMPT
    assert hashlib.sha256(candidate_bytes).hexdigest() == CANDIDATE_RAW_TEXT_SHA256
    assert (
        resolve_prompt_identity(ANSWER_COMPLETENESS_PROMPT_ID).prompt_sha256
        == CANDIDATE_PROMPT_SHA256
    )


def test_candidate_setting_resolves_without_changing_default() -> None:
    default_settings = Settings(_env_file=None, jwt_secret="test-only-secret")
    candidate_settings = Settings(
        _env_file=None,
        jwt_secret="test-only-secret",
        generation_prompt_id=ANSWER_COMPLETENESS_PROMPT_ID,
    )

    assert default_settings.generation_prompt_id == PRODUCTION_PROMPT_ID
    assert candidate_settings.generation_prompt_id == ANSWER_COMPLETENESS_PROMPT_ID
    assert build_system_prompt(candidate_settings.generation_prompt_id) == ANSWER_COMPLETENESS_PROMPT


def test_unknown_configured_prompt_fails_without_fallback() -> None:
    with pytest.raises(ValidationError, match="Unknown internal generation prompt ID"):
        Settings(
            _env_file=None,
            jwt_secret="test-only-secret",
            generation_prompt_id="unknown-prompt",
        )

    with pytest.raises(PromptResolutionError, match="unknown-prompt"):
        build_system_prompt("unknown-prompt")


def test_answering_uses_configured_resolved_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    retrieval_results = [
        RetrievalResult(
            document_id=1,
            chunk_id=2,
            chunk_order=0,
            chunk_text="Supported context.",
            distance=0.1,
        )
    ]
    monkeypatch.setattr(
        "app.answering.retrieve_chunks",
        lambda *args, **kwargs: SimpleNamespace(results=retrieval_results),
    )
    monkeypatch.setattr(
        "app.answering.get_settings",
        lambda: SimpleNamespace(generation_prompt_id=PRODUCTION_PROMPT_ID),
    )

    def fake_generate(system_prompt: str, user_prompt: str) -> str:
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return "Supported answer [1]"

    monkeypatch.setattr("app.answering.generate_answer", fake_generate)

    answer_question(object(), object(), "What is supported?")

    assert captured["system_prompt"] == SYSTEM_PROMPT
    assert "Supported context." in captured["user_prompt"]


def test_unknown_prompt_does_not_invoke_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    retrieval_results = [
        RetrievalResult(
            document_id=1,
            chunk_id=2,
            chunk_order=0,
            chunk_text="Supported context.",
            distance=0.1,
        )
    ]
    monkeypatch.setattr(
        "app.answering.retrieve_chunks",
        lambda *args, **kwargs: SimpleNamespace(results=retrieval_results),
    )
    monkeypatch.setattr(
        "app.answering.get_settings",
        lambda: SimpleNamespace(generation_prompt_id="unknown-prompt"),
    )

    def fake_generate(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return "must not run"

    monkeypatch.setattr("app.answering.generate_answer", fake_generate)

    with pytest.raises(PromptResolutionError, match="unknown-prompt"):
        answer_question(object(), object(), "What is supported?")

    assert called is False


def test_same_code_path_switches_to_test_only_allowlisted_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id = "test-only-prompt"
    test_prompt = "Test-only alternate system prompt."
    monkeypatch.setitem(
        prompting._PROMPT_REGISTRY,
        prompt_id,
        PromptDefinition(prompt_id=prompt_id, system_prompt=test_prompt),
    )
    retrieval_results = [
        RetrievalResult(
            document_id=1,
            chunk_id=2,
            chunk_order=0,
            chunk_text="Supported context.",
            distance=0.1,
        )
    ]
    selected = {"prompt_id": PRODUCTION_PROMPT_ID}
    captured_system_prompts: list[str] = []
    monkeypatch.setattr(
        "app.answering.retrieve_chunks",
        lambda *args, **kwargs: SimpleNamespace(results=retrieval_results),
    )
    monkeypatch.setattr(
        "app.answering.get_settings",
        lambda: SimpleNamespace(generation_prompt_id=selected["prompt_id"]),
    )

    def fake_generate(system_prompt: str, user_prompt: str) -> str:
        captured_system_prompts.append(system_prompt)
        return "Supported answer [1]"

    monkeypatch.setattr("app.answering.generate_answer", fake_generate)

    production_identity = resolve_prompt_identity(PRODUCTION_PROMPT_ID)
    test_identity = resolve_prompt_identity(prompt_id)
    answer_question(object(), object(), "What is supported?")
    selected["prompt_id"] = prompt_id
    answer_question(object(), object(), "What is supported?")

    assert build_system_prompt(PRODUCTION_PROMPT_ID) == SYSTEM_PROMPT
    assert build_system_prompt(prompt_id) == test_prompt
    assert captured_system_prompts == [SYSTEM_PROMPT, test_prompt]
    assert test_identity.prompt_id == prompt_id
    assert test_identity.prompt_sha256 != production_identity.prompt_sha256


def test_public_generation_schema_has_no_prompt_selector() -> None:
    assert set(GenerationRequest.model_fields) == {"question", "top_k"}
    assert set(GenerationResponse.model_fields) == {
        "answer",
        "context_available",
        "sources",
        "citation_validation",
    }
