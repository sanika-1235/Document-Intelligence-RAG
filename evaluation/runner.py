"""Resumable DDRAG evaluation runner using only public production HTTP APIs.

Importing this module performs no network, database, Ollama, or filesystem work.
Live execution requires the explicit CLI ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from evaluation.schemas import (
    Answerability,
    BenchmarkIdentity,
    BenchmarkRecord,
    ChunkingConfig,
    CitationValidationObservation,
    CorpusIdentity,
    EmbeddingConfig,
    ErrorDetail,
    ExperimentConfig,
    GenerationConfig,
    GenerationObservation,
    GenerationSource,
    OperationalStatus,
    PromptConfig,
    RawQuestionResult,
    RetrievalConfig,
    RetrievalObservation,
    RetrievedResult,
    RunManifest,
    RunStatus,
    SamplingConfig,
    SamplingMode,
    TimeoutConfig,
    WarmupConfig,
    WarmupStatus,
    configuration_hash,
    experiment_id,
)
from evaluation.validation import load_benchmark_jsonl, load_corpus_manifest, validate_artifacts

RETRIEVAL_STRATEGY = "pgvector_cosine_distance_owner_scoped_v1"
PRODUCTION_EMBEDDING_TIMEOUT_SECONDS = 30.0
PRODUCTION_GENERATION_TIMEOUT_SECONDS = 60.0
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "run_manifest.json"
RAW_RESULTS_FILENAME = "raw_results.jsonl"


class RunnerError(RuntimeError):
    """Safe runner error whose message contains no credentials or tokens."""


class RunnerConfigurationError(RunnerError):
    """Raised before execution when configuration or persisted state is incompatible."""


class AuthenticationError(RunnerError):
    """Raised when normal DDRAG authentication cannot establish a bearer token."""


class DuplicateQuestionResultError(RunnerError):
    """Raised instead of writing a second result for the same run/question pair."""


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _TokenResponse(_ApiModel):
    access_token: str
    token_type: str


class _RetrievalApiResult(_ApiModel):
    document_id: int
    chunk_id: int
    chunk_order: int
    chunk_text: str
    distance: float


class _RetrievalApiDiagnostics(_ApiModel):
    top_k_requested: int
    candidates_considered: int
    results_returned: int
    embedding_model: str
    embedding_dimension: int
    duration_ms: float


class _RetrievalApiResponse(_ApiModel):
    results: list[_RetrievalApiResult]
    diagnostics: _RetrievalApiDiagnostics


class _GenerationApiSource(_ApiModel):
    citation_id: int
    document_id: int
    chunk_id: int
    chunk_order: int
    distance: float


class _GenerationApiResponse(_ApiModel):
    answer: str
    context_available: bool
    sources: list[_GenerationApiSource]
    citation_validation: CitationValidationObservation


class RunnerApi(Protocol):
    def authenticate(self, email: str, password: str) -> None: ...

    def retrieve(self, question: str) -> RetrievalObservation: ...

    def generate(self, question: str) -> GenerationObservation: ...

    def warmup(self, question: str) -> GenerationObservation: ...


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without parsing or reformatting."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_prompt_identity(prompt_id: str) -> PromptConfig:
    """Resolve canonical prompt identity through the backend's registry."""

    from app.prompting import PromptResolutionError
    from app.prompting import resolve_prompt_identity as resolve_backend_prompt_identity

    try:
        identity = resolve_backend_prompt_identity(prompt_id)
    except PromptResolutionError as exc:
        raise RunnerConfigurationError(str(exc)) from exc
    return PromptConfig(
        prompt_id=identity.prompt_id,
        prompt_sha256=identity.prompt_sha256,
    )


def resolve_experiment_config(
    *,
    app_settings: Any,
    benchmark_path: Path,
    manifest_path: Path,
    request_timeout_seconds: float,
    top_k_override: int | None = None,
    warmup_required: bool = False,
) -> ExperimentConfig:
    """Snapshot safe production/evaluation values without credentials."""

    manifest = load_corpus_manifest(manifest_path)
    records = load_benchmark_jsonl(benchmark_path)
    if not records:
        raise RunnerConfigurationError("Benchmark must contain at least one record")
    benchmark_ids = {record.benchmark_id for record in records}
    benchmark_versions = {record.benchmark_version for record in records}
    if len(benchmark_ids) != 1 or len(benchmark_versions) != 1:
        raise RunnerConfigurationError("Benchmark identity is inconsistent")
    document = manifest.documents[0]
    resolved_top_k = (
        top_k_override if top_k_override is not None else app_settings.retrieval_default_top_k
    )
    return ExperimentConfig(
        corpus=CorpusIdentity(corpus_id=manifest.corpus_id, corpus_sha256=document.sha256),
        benchmark=BenchmarkIdentity(
            benchmark_id=next(iter(benchmark_ids)),
            benchmark_version=next(iter(benchmark_versions)),
            benchmark_sha256=sha256_file(benchmark_path),
        ),
        generation=GenerationConfig(generation_model=app_settings.generation_model),
        embedding=EmbeddingConfig(
            embedding_model=app_settings.embedding_model,
            embedding_dimension=app_settings.embedding_dimension,
        ),
        prompt=resolve_prompt_identity(app_settings.generation_prompt_id),
        chunking=ChunkingConfig(
            chunk_size_chars=app_settings.chunk_size_chars,
            chunk_overlap_chars=app_settings.chunk_overlap_chars,
            chunking_version=app_settings.chunking_version,
        ),
        retrieval=RetrievalConfig(
            top_k=resolved_top_k,
            retrieval_strategy=RETRIEVAL_STRATEGY,
        ),
        sampling=SamplingConfig(
            mode=SamplingMode.RUNTIME_DEFAULT,
            temperature=None,
            top_p=None,
            seed=None,
        ),
        timeouts=TimeoutConfig(
            request_timeout_seconds=request_timeout_seconds,
            retrieval_timeout_seconds=PRODUCTION_EMBEDDING_TIMEOUT_SECONDS,
            generation_timeout_seconds=PRODUCTION_GENERATION_TIMEOUT_SECONDS,
        ),
        warmup=WarmupConfig(required=warmup_required),
    )


def generate_run_id(
    *,
    now: datetime | None = None,
    random_hex: str | None = None,
) -> str:
    """Generate a sortable, filesystem-safe, collision-resistant run ID."""

    current = (now or datetime.now(UTC)).astimezone(UTC)
    suffix = random_hex or secrets.token_hex(4)
    return f"run_{current.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}"


def _safe_error(code: str, message: str, *, http_status: int | None = None) -> ErrorDetail:
    details = {"http_status": http_status} if http_status is not None else None
    return ErrorDetail(code=code, message=message, details=details)


class DDRAGApiClient:
    """Maps current authenticated production API responses to E5 observations."""

    def __init__(
        self,
        http_client: httpx.Client,
        config: ExperimentConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._http = http_client
        self._config = config
        self._clock = clock
        self._access_token: str | None = None

    def authenticate(self, email: str, password: str) -> None:
        try:
            response = self._http.post("/auth/login", json={"email": email, "password": password})
        except httpx.TimeoutException as exc:
            raise AuthenticationError("Authentication request timed out") from exc
        except httpx.RequestError as exc:
            raise AuthenticationError("Authentication request failed") from exc
        if response.status_code != 200:
            raise AuthenticationError(f"Authentication failed with HTTP {response.status_code}")
        try:
            parsed = _TokenResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise AuthenticationError("Authentication response was malformed") from exc
        if parsed.token_type.casefold() != "bearer" or not parsed.access_token.strip():
            raise AuthenticationError("Authentication response did not contain a bearer token")
        self._access_token = parsed.access_token

    def _headers(self) -> dict[str, str]:
        if self._access_token is None:
            raise AuthenticationError("Runner is not authenticated")
        return {"Authorization": f"Bearer {self._access_token}"}

    def _retrieval_failure(
        self,
        status: OperationalStatus,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
    ) -> RetrievalObservation:
        return RetrievalObservation(
            status=status,
            requested_top_k=self._config.retrieval.top_k,
            candidate_count=None,
            result_count=None,
            embedding_model=self._config.embedding.embedding_model,
            embedding_dimension=self._config.embedding.embedding_dimension,
            results=(),
            error=_safe_error(code, message, http_status=http_status),
        )

    def retrieve(self, question: str) -> RetrievalObservation:
        try:
            response = self._http.post(
                "/retrieval",
                json={"query": question, "top_k": self._config.retrieval.top_k},
                headers=self._headers(),
            )
        except httpx.TimeoutException:
            return self._retrieval_failure(
                OperationalStatus.TIMEOUT, "request_timeout", "Retrieval request timed out"
            )
        except (httpx.RequestError, AuthenticationError):
            return self._retrieval_failure(
                OperationalStatus.RETRIEVAL_FAILURE,
                "request_error",
                "Retrieval request failed",
            )
        if response.status_code in (401, 403):
            return self._retrieval_failure(
                OperationalStatus.AUTHENTICATION_FAILURE,
                "authentication_failure",
                "Retrieval authentication failed",
                http_status=response.status_code,
            )
        if not response.is_success:
            return self._retrieval_failure(
                OperationalStatus.RETRIEVAL_FAILURE,
                "http_error",
                "Retrieval endpoint returned an error",
                http_status=response.status_code,
            )
        try:
            parsed = _RetrievalApiResponse.model_validate(response.json())
            results = tuple(
                RetrievedResult(
                    rank=rank,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    chunk_order=item.chunk_order,
                    distance=item.distance,
                    chunk_text=item.chunk_text,
                )
                for rank, item in enumerate(parsed.results, start=1)
            )
            return RetrievalObservation(
                status=OperationalStatus.SUCCESS,
                latency_ms=parsed.diagnostics.duration_ms,
                requested_top_k=parsed.diagnostics.top_k_requested,
                candidate_count=parsed.diagnostics.candidates_considered,
                result_count=parsed.diagnostics.results_returned,
                embedding_model=parsed.diagnostics.embedding_model,
                embedding_dimension=parsed.diagnostics.embedding_dimension,
                results=results,
                error=None,
            )
        except (ValueError, ValidationError):
            return self._retrieval_failure(
                OperationalStatus.RESPONSE_PARSE_FAILURE,
                "response_parse_failure",
                "Retrieval response was malformed",
            )

    def _generation_failure(
        self,
        status: OperationalStatus,
        code: str,
        message: str,
        latency_ms: float,
        *,
        http_status: int | None = None,
    ) -> GenerationObservation:
        return GenerationObservation(
            status=status,
            latency_ms=latency_ms,
            generation_model=self._config.generation.generation_model,
            answer=None,
            context_available=None,
            sources=(),
            citation_validation=None,
            error=_safe_error(code, message, http_status=http_status),
        )

    def generate(self, question: str) -> GenerationObservation:
        started = self._clock()
        try:
            response = self._http.post(
                "/generation",
                json={"question": question, "top_k": self._config.retrieval.top_k},
                headers=self._headers(),
            )
        except httpx.TimeoutException:
            latency = max(0.0, (self._clock() - started) * 1000)
            return self._generation_failure(
                OperationalStatus.TIMEOUT,
                "request_timeout",
                "Generation request timed out",
                latency,
            )
        except (httpx.RequestError, AuthenticationError):
            latency = max(0.0, (self._clock() - started) * 1000)
            return self._generation_failure(
                OperationalStatus.GENERATION_FAILURE,
                "request_error",
                "Generation request failed",
                latency,
            )
        latency = max(0.0, (self._clock() - started) * 1000)
        if response.status_code in (401, 403):
            return self._generation_failure(
                OperationalStatus.AUTHENTICATION_FAILURE,
                "authentication_failure",
                "Generation authentication failed",
                latency,
                http_status=response.status_code,
            )
        if not response.is_success:
            return self._generation_failure(
                OperationalStatus.GENERATION_FAILURE,
                "http_error",
                "Generation endpoint returned an error",
                latency,
                http_status=response.status_code,
            )
        try:
            parsed = _GenerationApiResponse.model_validate(response.json())
            status = (
                OperationalStatus.SUCCESS
                if parsed.citation_validation.valid
                else OperationalStatus.CITATION_VALIDATION_FAILURE
            )
            return GenerationObservation(
                status=status,
                latency_ms=latency,
                generation_model=self._config.generation.generation_model,
                answer=parsed.answer,
                context_available=parsed.context_available,
                sources=tuple(
                    GenerationSource(
                        citation_id=source.citation_id,
                        document_id=source.document_id,
                        chunk_id=source.chunk_id,
                        chunk_order=source.chunk_order,
                        distance=source.distance,
                    )
                    for source in parsed.sources
                ),
                citation_validation=parsed.citation_validation,
                error=None,
            )
        except (ValueError, ValidationError):
            return self._generation_failure(
                OperationalStatus.RESPONSE_PARSE_FAILURE,
                "response_parse_failure",
                "Generation response was malformed",
                latency,
            )

    def warmup(self, question: str) -> GenerationObservation:
        return self.generate(question)


def retrieval_observation_mismatch(
    retrieval: RetrievalObservation | None,
    generation: GenerationObservation | None,
) -> bool | None:
    """Compare ordered ``(document_id, chunk_id, chunk_order)`` sequences only."""

    if (
        retrieval is None
        or generation is None
        or retrieval.status is not OperationalStatus.SUCCESS
        or generation.status is not OperationalStatus.SUCCESS
    ):
        return None
    retrieved = tuple(
        (item.document_id, item.chunk_id, item.chunk_order) for item in retrieval.results
    )
    generated = tuple(
        (item.document_id, item.chunk_id, item.chunk_order) for item in generation.sources
    )
    return retrieved != generated


_STATUS_PRECEDENCE = (
    OperationalStatus.AUTHENTICATION_FAILURE,
    OperationalStatus.CONFIGURATION_FAILURE,
    OperationalStatus.RUNNER_FAILURE,
    OperationalStatus.TIMEOUT,
    OperationalStatus.RESPONSE_PARSE_FAILURE,
    OperationalStatus.CITATION_VALIDATION_FAILURE,
    OperationalStatus.GENERATION_FAILURE,
    OperationalStatus.RETRIEVAL_FAILURE,
)


def overall_status(
    retrieval: RetrievalObservation | None,
    generation: GenerationObservation | None,
) -> OperationalStatus:
    """Resolve mixed outcomes by documented operational precedence.

    Sub-observations always retain both detailed outcomes. Authentication and
    configuration failures outrank timeouts, parsing/citation failures, then
    endpoint-specific generation and retrieval failures.
    """

    statuses = {
        observation.status
        for observation in (retrieval, generation)
        if observation is not None
    }
    if statuses == {OperationalStatus.SUCCESS}:
        return OperationalStatus.SUCCESS
    for candidate in _STATUS_PRECEDENCE:
        if candidate in statuses:
            return candidate
    return OperationalStatus.RUNNER_FAILURE


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_model(path: Path, model: Any) -> None:
    _atomic_write(path, model.model_dump_json(indent=2) + "\n")


class ResultJournal:
    """Fsynced JSONL writer enforcing one result per run/question."""

    def __init__(self, path: Path, *, run_id: str, experiment_id_value: str) -> None:
        self.path = path
        self.run_id = run_id
        self.experiment_id = experiment_id_value
        self._question_ids: set[str] = set()
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    raise RunnerConfigurationError(
                        f"Raw results contain a blank line at line {line_number}"
                    )
                try:
                    result = RawQuestionResult.model_validate_json(line)
                except ValidationError as exc:
                    raise RunnerConfigurationError(
                        f"Raw results are invalid at line {line_number}: {exc}"
                    ) from exc
                if result.run_id != run_id or result.experiment_id != experiment_id_value:
                    raise RunnerConfigurationError("Raw results identity does not match the run")
                if result.question_id in self._question_ids:
                    raise RunnerConfigurationError(
                        f"Raw results contain duplicate question ID {result.question_id}"
                    )
                self._question_ids.add(result.question_id)

    @property
    def question_ids(self) -> frozenset[str]:
        return frozenset(self._question_ids)

    def append(self, result: RawQuestionResult) -> None:
        if result.run_id != self.run_id or result.experiment_id != self.experiment_id:
            raise RunnerConfigurationError("Result identity does not match the journal")
        if result.question_id in self._question_ids:
            raise DuplicateQuestionResultError(
                f"Question {result.question_id} already has a result in run {self.run_id}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._question_ids.add(result.question_id)


class RunStore:
    """Persisted immutable config, atomic manifest, and append-only results."""

    def __init__(
        self,
        run_dir: Path,
        config: ExperimentConfig,
        manifest: RunManifest,
        journal: ResultJournal,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.manifest = manifest
        self.journal = journal

    @classmethod
    def create(
        cls,
        *,
        runs_root: Path,
        config: ExperimentConfig,
        question_total: int,
        now: datetime | None = None,
        run_id: str | None = None,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> "RunStore":
        started = (now or datetime.now(UTC)).astimezone(UTC)
        resolved_run_id = run_id or generate_run_id(now=started)
        run_dir = runs_root / resolved_run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunnerConfigurationError(f"Run directory already exists: {run_dir}") from exc
        digest = configuration_hash(config)
        exp_id = experiment_id(config)
        metadata = runtime_metadata or {}
        manifest = RunManifest(
            run_id=resolved_run_id,
            experiment_id=exp_id,
            config_hash=digest,
            started_at=started,
            completed_at=None,
            status=RunStatus.PENDING,
            corpus=config.corpus,
            benchmark=config.benchmark,
            git_commit_sha=metadata.get("git_commit_sha"),
            git_dirty=metadata.get("git_dirty"),
            python_version=metadata.get("python_version"),
            platform=metadata.get("platform"),
            project_version=metadata.get("project_version"),
            ollama_version=metadata.get("ollama_version"),
            hardware=metadata.get("hardware"),
            warmup_status=(
                WarmupStatus.PENDING if config.warmup.required else WarmupStatus.NOT_REQUIRED
            ),
            question_total=question_total,
            success_total=0,
            failure_total=0,
        )
        _atomic_write_model(run_dir / CONFIG_FILENAME, config)
        _atomic_write_model(run_dir / MANIFEST_FILENAME, manifest)
        raw_path = run_dir / RAW_RESULTS_FILENAME
        raw_path.touch(exist_ok=False)
        journal = ResultJournal(raw_path, run_id=resolved_run_id, experiment_id_value=exp_id)
        return cls(run_dir, config, manifest, journal)

    @classmethod
    def resume(
        cls,
        *,
        run_dir: Path,
        expected_config: ExperimentConfig,
        benchmark_path: Path,
    ) -> "RunStore":
        try:
            stored_config = ExperimentConfig.model_validate_json(
                (run_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
            )
            manifest = RunManifest.model_validate_json(
                (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise RunnerConfigurationError(f"Unable to load resumable run state: {exc}") from exc
        stored_hash = configuration_hash(stored_config)
        if manifest.config_hash != stored_hash:
            raise RunnerConfigurationError("Persisted config hash does not match run manifest")
        if manifest.experiment_id != experiment_id(stored_config):
            raise RunnerConfigurationError("Persisted experiment ID does not match configuration")
        if configuration_hash(expected_config) != stored_hash:
            raise RunnerConfigurationError("Current experiment configuration differs from resumed run")
        if experiment_id(expected_config) != manifest.experiment_id:
            raise RunnerConfigurationError("Current experiment ID differs from resumed run")
        if expected_config.benchmark.benchmark_sha256 != sha256_file(benchmark_path):
            raise RunnerConfigurationError("Current benchmark bytes differ from resumed run")
        if run_dir.name != manifest.run_id:
            raise RunnerConfigurationError("Run directory name does not match run manifest")
        journal = ResultJournal(
            run_dir / RAW_RESULTS_FILENAME,
            run_id=manifest.run_id,
            experiment_id_value=manifest.experiment_id,
        )
        return cls(run_dir, stored_config, manifest, journal)

    def update_manifest(self, **updates: Any) -> RunManifest:
        candidate = self.manifest.model_dump()
        candidate.update(updates)
        self.manifest = RunManifest.model_validate(candidate)
        _atomic_write_model(self.run_dir / MANIFEST_FILENAME, self.manifest)
        return self.manifest


class EvaluationRunner:
    """Execute independent retrieval/generation observations and persist each result."""

    def __init__(
        self,
        *,
        store: RunStore,
        records: Sequence[BenchmarkRecord],
        api: RunnerApi,
        email: str,
        password: str,
        warmup_question: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.records = tuple(records)
        self.api = api
        self.email = email
        self.password = password
        self.warmup_question = warmup_question
        self._now = now or (lambda: datetime.now(UTC))

    def _counts(self) -> tuple[int, int]:
        successes = 0
        failures = 0
        if self.store.journal.path.exists():
            for line in self.store.journal.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                result = RawQuestionResult.model_validate_json(line)
                if result.overall_status is OperationalStatus.SUCCESS:
                    successes += 1
                else:
                    failures += 1
        return successes, failures

    def run(self) -> RunManifest:
        pending = [
            record for record in self.records if record.id not in self.store.journal.question_ids
        ]
        if not pending:
            successes, failures = self._counts()
            return self.store.update_manifest(
                status=RunStatus.COMPLETED,
                completed_at=self._now(),
                success_total=successes,
                failure_total=failures,
            )
        if self.store.config.warmup.required and not (self.warmup_question or "").strip():
            self.store.update_manifest(status=RunStatus.FAILED, completed_at=self._now())
            raise RunnerConfigurationError("Warm-up is required but no warm-up question was supplied")
        try:
            self.api.authenticate(self.email, self.password)
        except AuthenticationError:
            self.store.update_manifest(status=RunStatus.FAILED, completed_at=self._now())
            raise

        if (
            self.store.config.warmup.required
            and self.store.manifest.warmup_status is not WarmupStatus.COMPLETED
        ):
            warmup = self.api.warmup(self.warmup_question or "")
            if warmup.status is not OperationalStatus.SUCCESS:
                self.store.update_manifest(
                    status=RunStatus.FAILED,
                    completed_at=self._now(),
                    warmup_status=WarmupStatus.FAILED,
                )
                raise RunnerError("Warm-up failed; official question execution was not started")
            self.store.update_manifest(warmup_status=WarmupStatus.COMPLETED)

        self.store.update_manifest(status=RunStatus.RUNNING, completed_at=None)
        try:
            for record in pending:
                started = self._now()
                retrieval = self.api.retrieve(record.question)
                generation = (
                    None
                    if retrieval.status is OperationalStatus.AUTHENTICATION_FAILURE
                    else self.api.generate(record.question)
                )
                result = RawQuestionResult(
                    run_id=self.store.manifest.run_id,
                    experiment_id=self.store.manifest.experiment_id,
                    question_id=record.id,
                    category=record.category,
                    answerability=record.answerability,
                    started_at=started,
                    completed_at=self._now(),
                    overall_status=overall_status(retrieval, generation),
                    retrieval=retrieval,
                    generation=generation,
                    retrieval_observation_mismatch=retrieval_observation_mismatch(
                        retrieval, generation
                    ),
                )
                self.store.journal.append(result)
                successes, failures = self._counts()
                self.store.update_manifest(
                    success_total=successes,
                    failure_total=failures,
                )
        except Exception:
            successes, failures = self._counts()
            self.store.update_manifest(
                status=RunStatus.PARTIAL if successes + failures else RunStatus.FAILED,
                completed_at=self._now(),
                success_total=successes,
                failure_total=failures,
            )
            raise
        successes, failures = self._counts()
        return self.store.update_manifest(
            status=RunStatus.COMPLETED,
            completed_at=self._now(),
            success_total=successes,
            failure_total=failures,
        )


def collect_runtime_metadata() -> dict[str, Any]:
    """Collect nullable local metadata without calling services or exposing secrets."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    try:
        project_version = version("ddrag")
    except PackageNotFoundError:
        project_version = None
    return {
        "git_commit_sha": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_version": project_version,
        "ollama_version": None,
        "hardware": {"machine": platform.machine(), "processor": platform.processor() or None},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DDRAG evaluation through authenticated production HTTP APIs"
    )
    parser.add_argument("--execute", action="store_true", help="required to send live API requests")
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmarks/csu_handbook_v1.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/corpus/corpus_manifest.json"))
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/corpus/CSU-Student-Handbook.pdf"))
    parser.add_argument("--runs-root", type=Path, default=Path("evaluation/runs"))
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--base-url", default=os.getenv("DDRAG_EVAL_BASE_URL"))
    parser.add_argument("--email", default=os.getenv("DDRAG_EVAL_EMAIL"))
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("DDRAG_EVAL_REQUEST_TIMEOUT", "90")),
    )
    parser.add_argument("--require-warmup", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.execute:
        parser.error("live execution requires the explicit --execute flag")
    password = os.getenv("DDRAG_EVAL_PASSWORD")
    warmup_question = os.getenv("DDRAG_EVAL_WARMUP_QUESTION")
    if not args.base_url or not args.email or not password:
        parser.error(
            "--base-url/DDRAG_EVAL_BASE_URL, --email/DDRAG_EVAL_EMAIL, and "
            "DDRAG_EVAL_PASSWORD are required"
        )
    if args.require_warmup and not (warmup_question or "").strip():
        parser.error("DDRAG_EVAL_WARMUP_QUESTION is required with --require-warmup")

    try:
        _, records = validate_artifacts(
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            corpus_path=args.corpus,
            strict_csu_v1=True,
        )
        from app.config import get_settings

        config = resolve_experiment_config(
            app_settings=get_settings(),
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            request_timeout_seconds=args.request_timeout,
            top_k_override=args.top_k,
            warmup_required=args.require_warmup,
        )
        store = (
            RunStore.resume(
                run_dir=args.resume_run,
                expected_config=config,
                benchmark_path=args.benchmark,
            )
            if args.resume_run
            else RunStore.create(
                runs_root=args.runs_root,
                config=config,
                question_total=len(records),
                runtime_metadata=collect_runtime_metadata(),
            )
        )
        with httpx.Client(base_url=args.base_url, timeout=args.request_timeout) as http_client:
            manifest = EvaluationRunner(
                store=store,
                records=records,
                api=DDRAGApiClient(http_client, config),
                email=args.email,
                password=password,
                warmup_question=warmup_question,
            ).run()
    except RunnerError as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        return 1
    print(f"Run {manifest.run_id} finished with status {manifest.status.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
