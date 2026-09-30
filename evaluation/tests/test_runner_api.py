"""E6A tests for current production API mapping through mock transports."""

from __future__ import annotations

import json

import httpx
import pytest

from evaluation.runner import (
    AuthenticationError,
    DDRAGApiClient,
    overall_status,
    retrieval_observation_mismatch,
)
from evaluation.schemas import (
    ExperimentConfig,
    GenerationObservation,
    GenerationSource,
    OperationalStatus,
    RetrievalObservation,
    RetrievedResult,
)


def _config() -> ExperimentConfig:
    payload = {
        "schema_version": "1.0.0",
        "corpus": {"corpus_id": "fixture", "corpus_sha256": "a" * 64},
        "benchmark": {
            "benchmark_id": "fixture",
            "benchmark_version": "1.0.0",
            "benchmark_sha256": "b" * 64,
        },
        "generation": {"generation_model": "generation-fixture"},
        "embedding": {"embedding_model": "embedding-fixture", "embedding_dimension": 8},
        "prompt": {"prompt_id": "prompt-fixture", "prompt_sha256": "c" * 64},
        "chunking": {
            "chunk_size_chars": 100,
            "chunk_overlap_chars": 10,
            "chunking_version": "chunk-fixture",
        },
        "retrieval": {"top_k": 2, "retrieval_strategy": "fixture"},
        "sampling": {
            "mode": "runtime_default",
            "temperature": None,
            "top_p": None,
            "seed": None,
        },
        "timeouts": {
            "request_timeout_seconds": 5.0,
            "retrieval_timeout_seconds": 30.0,
            "generation_timeout_seconds": 60.0,
        },
        "warmup": {"required": False},
    }
    return ExperimentConfig.model_validate_json(json.dumps(payload))


def _retrieval_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "document_id": 10,
                "chunk_id": 20,
                "chunk_order": 3,
                "chunk_text": "full chunk text",
                "distance": 0.123,
            },
            {
                "document_id": 10,
                "chunk_id": 21,
                "chunk_order": 4,
                "chunk_text": "second chunk",
                "distance": 0.456,
            },
        ],
        "diagnostics": {
            "top_k_requested": 2,
            "candidates_considered": 7,
            "results_returned": 2,
            "embedding_model": "embedding-fixture",
            "embedding_dimension": 8,
            "duration_ms": 12.34,
        },
    }


def _generation_payload() -> dict[str, object]:
    return {
        "answer": "grounded answer [1]",
        "context_available": True,
        "sources": [
            {
                "citation_id": 1,
                "document_id": 10,
                "chunk_id": 20,
                "chunk_order": 3,
                "distance": 0.999,
            },
            {
                "citation_id": 2,
                "document_id": 10,
                "chunk_id": 21,
                "chunk_order": 4,
                "distance": 0.888,
            },
        ],
        "citation_validation": {
            "cited_ids": [1],
            "valid": True,
            "invalid_citations": [],
        },
    }


def test_authentication_and_successful_api_mapping() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/auth/login":
            assert json.loads(request.content) == {
                "email": "eval@example.invalid",
                "password": "sentinel-password",
            }
            return httpx.Response(200, json={"access_token": "sentinel-token", "token_type": "bearer"})
        assert request.headers["Authorization"] == "Bearer sentinel-token"
        if request.url.path == "/retrieval":
            assert json.loads(request.content)["top_k"] == 2
            return httpx.Response(200, json=_retrieval_payload())
        if request.url.path == "/generation":
            assert json.loads(request.content)["top_k"] == 2
            return httpx.Response(200, json=_generation_payload())
        raise AssertionError(request.url.path)

    times = iter([10.0, 10.25])
    with httpx.Client(
        base_url="http://ddrag.invalid", transport=httpx.MockTransport(handler)
    ) as http_client:
        client = DDRAGApiClient(http_client, _config(), clock=lambda: next(times))
        client.authenticate("eval@example.invalid", "sentinel-password")
        retrieval = client.retrieve("question")
        generation = client.generate("question")

    assert [item.rank for item in retrieval.results] == [1, 2]
    assert retrieval.results[0].chunk_text == "full chunk text"
    assert retrieval.latency_ms == 12.34
    assert generation.latency_ms == 250.0
    assert generation.sources[0].chunk_id == 20
    assert generation.citation_validation is not None
    assert generation.citation_validation.cited_ids == (1,)
    assert retrieval_observation_mismatch(retrieval, generation) is False
    assert len(requests) == 3


def _retrieval(ids: list[tuple[int, int, int]], *, distance: float = 0.1) -> RetrievalObservation:
    return RetrievalObservation(
        status=OperationalStatus.SUCCESS,
        latency_ms=1.0,
        requested_top_k=5,
        candidate_count=len(ids),
        result_count=len(ids),
        embedding_model="fixture",
        embedding_dimension=8,
        results=tuple(
            RetrievedResult(
                rank=index,
                document_id=document,
                chunk_id=chunk,
                chunk_order=order,
                distance=distance,
                chunk_text="text",
            )
            for index, (document, chunk, order) in enumerate(ids, 1)
        ),
    )


def _generation(
    ids: list[tuple[int, int, int]],
    *,
    status: OperationalStatus = OperationalStatus.SUCCESS,
    distance: float = 9.9,
) -> GenerationObservation:
    return GenerationObservation(
        status=status,
        latency_ms=2.0,
        generation_model="fixture",
        answer="answer",
        context_available=True,
        sources=tuple(
            GenerationSource(
                citation_id=index,
                document_id=document,
                chunk_id=chunk,
                chunk_order=order,
                distance=distance,
            )
            for index, (document, chunk, order) in enumerate(ids, 1)
        ),
    )


def test_mismatch_ignores_distance_but_preserves_order() -> None:
    ids = [(1, 2, 3), (1, 4, 5)]
    assert retrieval_observation_mismatch(_retrieval(ids, distance=0.1), _generation(ids, distance=8.0)) is False
    assert retrieval_observation_mismatch(_retrieval(ids), _generation([(1, 2, 3), (2, 4, 5)])) is True
    assert retrieval_observation_mismatch(_retrieval(ids), _generation(list(reversed(ids)))) is True


def test_mismatch_is_null_on_either_failure() -> None:
    success_r = _retrieval([(1, 2, 3)])
    success_g = _generation([(1, 2, 3)])
    failed_r = success_r.model_copy(update={"status": OperationalStatus.RETRIEVAL_FAILURE})
    failed_g = success_g.model_copy(update={"status": OperationalStatus.GENERATION_FAILURE})
    assert retrieval_observation_mismatch(failed_r, success_g) is None
    assert retrieval_observation_mismatch(success_r, failed_g) is None


def test_http_and_malformed_response_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "token", "token_type": "bearer"})
        if request.url.path == "/retrieval":
            return httpx.Response(503, json={"detail": "internal detail is not persisted"})
        return httpx.Response(200, json={"unexpected": "shape"})

    with httpx.Client(base_url="http://ddrag.invalid", transport=httpx.MockTransport(handler)) as http_client:
        client = DDRAGApiClient(http_client, _config(), clock=iter([1.0, 1.1]).__next__)
        client.authenticate("a@b.invalid", "password")
        retrieval = client.retrieve("question")
        generation = client.generate("question")
    assert retrieval.status is OperationalStatus.RETRIEVAL_FAILURE
    assert retrieval.error is not None and retrieval.error.details == {"http_status": 503}
    assert generation.status is OperationalStatus.RESPONSE_PARSE_FAILURE
    assert generation.error is not None


def test_timeout_mapping_and_status_precedence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "token", "token_type": "bearer"})
        raise httpx.ReadTimeout("timeout", request=request)

    with httpx.Client(base_url="http://ddrag.invalid", transport=httpx.MockTransport(handler)) as http_client:
        client = DDRAGApiClient(http_client, _config(), clock=iter([1.0, 1.2]).__next__)
        client.authenticate("a@b.invalid", "password")
        retrieval = client.retrieve("question")
        generation = client.generate("question")
    assert retrieval.status is OperationalStatus.TIMEOUT
    assert generation.status is OperationalStatus.TIMEOUT
    assert overall_status(retrieval, generation) is OperationalStatus.TIMEOUT


def test_authentication_failure_is_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "do not persist credentials"})

    with httpx.Client(base_url="http://ddrag.invalid", transport=httpx.MockTransport(handler)) as http_client:
        client = DDRAGApiClient(http_client, _config())
        with pytest.raises(AuthenticationError, match="HTTP 401") as error:
            client.authenticate("a@b.invalid", "sentinel-password")
    assert "sentinel-password" not in str(error.value)
