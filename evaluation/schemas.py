"""Strict, immutable serialization contracts for DDRAG evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
CONFIG_HASH_PREFIX_LENGTH = 12
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    """Base model that rejects unknown/coerced fields and is immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class CorpusDocument(StrictModel):
    document_id: str
    expected_filename: str
    expected_relative_path: str
    sha256: str
    size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)

    @field_validator("document_id", "expected_filename", "expected_relative_path")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class CorpusManifest(StrictModel):
    """Compatibility model for the existing E3 corpus manifest."""

    manifest_version: str
    corpus_id: str
    title: str
    edition: str
    document_count: int = Field(gt=0)
    documents: tuple[CorpusDocument, ...]
    distribution: str

    @field_validator("documents", mode="before")
    @classmethod
    def parse_documents(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("manifest_version", "corpus_id", "title", "edition", "distribution")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_document_count(self) -> "CorpusManifest":
        if self.document_count != len(self.documents):
            raise ValueError("document_count must equal the number of document entries")
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        return self


class BenchmarkCategory(str, Enum):
    DIRECT_FACTUAL = "direct_factual"
    PROCEDURAL = "procedural"
    MULTI_CONDITION = "multi_condition"
    MULTI_CHUNK = "multi_chunk"
    POLICY_ELIGIBILITY = "policy_eligibility"
    SUMMARIZATION = "summarization"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class ReviewStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"


class ExpectedBehavior(str, Enum):
    ANSWER = "answer"
    CLARIFY_OR_QUALIFY = "clarify_or_qualify"
    REFUSE_OR_STATE_INSUFFICIENT_INFORMATION = "refuse_or_state_insufficient_information"


class ExpectedPoint(StrictModel):
    id: str
    text: str
    automated_match: bool

    @field_validator("id", "text")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class GoldEvidence(StrictModel):
    evidence_id: str
    pdf_page: int = Field(gt=0)
    printed_page_label: str | None
    section: str | None
    anchor: str
    notes: str | None = None

    @field_validator("evidence_id", "anchor")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("section", "printed_page_label", "notes")
    @classmethod
    def validate_optional_non_empty(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            return _non_empty(value, info.field_name)
        return value


class BenchmarkRecord(StrictModel):
    id: str
    benchmark_id: str
    benchmark_version: str
    review_status: ReviewStatus
    category: BenchmarkCategory
    question: str
    answerability: Answerability
    expected_behavior: ExpectedBehavior
    reference_answer: str | None
    expected_points: tuple[ExpectedPoint, ...]
    evidence: tuple[GoldEvidence, ...]
    notes: str | None

    @field_validator("expected_points", "evidence", mode="before")
    @classmethod
    def parse_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("id", "benchmark_id", "benchmark_version", "question")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("reference_answer", "notes")
    @classmethod
    def validate_optional_non_empty(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            return _non_empty(value, info.field_name)
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "BenchmarkRecord":
        point_ids = [point.id for point in self.expected_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("expected-point IDs must be unique within a question")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a question")
        if not self.expected_points:
            raise ValueError("every benchmark record must have at least one expected point")

        if self.answerability is Answerability.ANSWERABLE:
            if self.expected_behavior is not ExpectedBehavior.ANSWER:
                raise ValueError("answerable records must use expected_behavior='answer'")
            if self.reference_answer is None:
                raise ValueError("answerable records require a reference answer")
            if not self.evidence:
                raise ValueError("answerable records require gold evidence")
        elif self.answerability is Answerability.AMBIGUOUS:
            if self.expected_behavior is not ExpectedBehavior.CLARIFY_OR_QUALIFY:
                raise ValueError("ambiguous records must use expected_behavior='clarify_or_qualify'")
            if self.reference_answer is None:
                raise ValueError("ambiguous records require response guidance")
            if not self.evidence:
                raise ValueError("source-grounded ambiguous records require evidence")
        else:
            if self.expected_behavior is not ExpectedBehavior.REFUSE_OR_STATE_INSUFFICIENT_INFORMATION:
                raise ValueError("unanswerable records must use the refusal expected behavior")
            if self.reference_answer is not None:
                raise ValueError("unanswerable records require reference_answer=null")
            if self.evidence:
                raise ValueError("unanswerable records require an empty evidence list")
        return self


class CorpusIdentity(StrictModel):
    corpus_id: str
    corpus_sha256: str

    @field_validator("corpus_id")
    @classmethod
    def validate_corpus_id(cls, value: str) -> str:
        return _non_empty(value, "corpus_id")

    @field_validator("corpus_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
        return value


class BenchmarkIdentity(StrictModel):
    benchmark_id: str
    benchmark_version: str
    benchmark_sha256: str

    @field_validator("benchmark_id", "benchmark_version")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("benchmark_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("benchmark_sha256 must be a lowercase SHA-256 digest")
        return value


class GenerationConfig(StrictModel):
    generation_model: str

    @field_validator("generation_model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "generation_model")


class EmbeddingConfig(StrictModel):
    embedding_model: str
    embedding_dimension: int = Field(gt=0)

    @field_validator("embedding_model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "embedding_model")


class PromptConfig(StrictModel):
    prompt_id: str
    prompt_sha256: str

    @field_validator("prompt_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "prompt_id")

    @field_validator("prompt_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")
        return value


class ChunkingConfig(StrictModel):
    chunk_size_chars: int = Field(gt=0)
    chunk_overlap_chars: int = Field(ge=0)
    chunking_version: str

    @field_validator("chunking_version")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "chunking_version")

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        return self


class RetrievalConfig(StrictModel):
    top_k: int = Field(gt=0)
    retrieval_strategy: str

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "retrieval_strategy")


class SamplingMode(str, Enum):
    EXPLICIT = "explicit"
    RUNTIME_DEFAULT = "runtime_default"
    UNSPECIFIED = "unspecified"


class SamplingConfig(StrictModel):
    mode: SamplingMode
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    seed: int | None = None

    @field_validator("temperature", "top_p")
    @classmethod
    def validate_finite(cls, value: float | None, info: Any) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "SamplingConfig":
        values = (self.temperature, self.top_p, self.seed)
        if self.mode is SamplingMode.EXPLICIT and all(value is None for value in values):
            raise ValueError("explicit sampling requires at least one explicit value")
        if self.mode is not SamplingMode.EXPLICIT and any(value is not None for value in values):
            raise ValueError("runtime-default/unspecified sampling values must be null")
        return self


class TimeoutConfig(StrictModel):
    request_timeout_seconds: float | None = Field(default=None, gt=0)
    retrieval_timeout_seconds: float | None = Field(default=None, gt=0)
    generation_timeout_seconds: float | None = Field(default=None, gt=0)


class WarmupConfig(StrictModel):
    required: bool


class ExperimentConfig(StrictModel):
    """Immutable inputs whose canonical form defines an experiment identity."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    corpus: CorpusIdentity
    benchmark: BenchmarkIdentity
    generation: GenerationConfig
    embedding: EmbeddingConfig
    prompt: PromptConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    sampling: SamplingConfig
    timeouts: TimeoutConfig
    warmup: WarmupConfig


def canonical_config_json(config: ExperimentConfig) -> str:
    """Serialize configuration deterministically, retaining explicit nulls."""

    return json.dumps(
        config.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def configuration_hash(config: ExperimentConfig) -> str:
    """Return the full SHA-256 digest of canonical experiment inputs."""

    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def experiment_id(config: ExperimentConfig) -> str:
    """Return ``exp_`` plus the first 12 hexadecimal config-hash characters."""

    return f"exp_{configuration_hash(config)[:CONFIG_HASH_PREFIX_LENGTH]}"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class WarmupStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunManifest(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    config_hash: str
    started_at: datetime
    completed_at: datetime | None = None
    status: RunStatus
    corpus: CorpusIdentity
    benchmark: BenchmarkIdentity
    git_commit_sha: str | None = None
    git_dirty: bool | None = None
    python_version: str | None = None
    platform: str | None = None
    project_version: str | None = None
    ollama_version: str | None = None
    hardware: dict[str, str | int | float | bool | None] | None = None
    warmup_status: WarmupStatus
    question_total: int = Field(ge=0)
    success_total: int = Field(ge=0)
    failure_total: int = Field(ge=0)

    @field_validator("run_id", "experiment_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("config_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("config_hash must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_totals_and_time(self) -> "RunManifest":
        if self.success_total + self.failure_total > self.question_total:
            raise ValueError("success_total + failure_total cannot exceed question_total")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class OperationalStatus(str, Enum):
    SUCCESS = "success"
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    TIMEOUT = "timeout"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RESPONSE_PARSE_FAILURE = "response_parse_failure"
    CITATION_VALIDATION_FAILURE = "citation_validation_failure"
    CONFIGURATION_FAILURE = "configuration_failure"
    RUNNER_FAILURE = "runner_failure"


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: dict[str, str | int | float | bool | None] | None = None

    @field_validator("code", "message")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class RetrievedResult(StrictModel):
    rank: int = Field(gt=0)
    document_id: int = Field(gt=0)
    chunk_id: int = Field(gt=0)
    chunk_order: int = Field(ge=0)
    distance: float
    chunk_text: str

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("distance must be finite")
        return value


class RetrievalObservation(StrictModel):
    status: OperationalStatus
    latency_ms: float | None = Field(default=None, ge=0)
    requested_top_k: int = Field(gt=0)
    candidate_count: int | None = Field(default=None, ge=0)
    result_count: int | None = Field(default=None, ge=0)
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    results: tuple[RetrievedResult, ...] = ()
    error: ErrorDetail | None = None

    @field_validator("results", mode="before")
    @classmethod
    def parse_results(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("embedding_model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "embedding_model")

    @model_validator(mode="after")
    def validate_counts(self) -> "RetrievalObservation":
        if self.result_count is not None and self.result_count != len(self.results):
            raise ValueError("result_count must equal the number of retrieved results")
        if (
            self.candidate_count is not None
            and self.result_count is not None
            and self.result_count > self.candidate_count
        ):
            raise ValueError("result_count cannot exceed candidate_count")
        ranks = [result.rank for result in self.results]
        if len(ranks) != len(set(ranks)):
            raise ValueError("retrieval result ranks must be unique")
        return self


class GenerationSource(StrictModel):
    citation_id: int = Field(gt=0)
    document_id: int = Field(gt=0)
    chunk_id: int = Field(gt=0)
    chunk_order: int = Field(ge=0)
    distance: float

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("distance must be finite")
        return value


class CitationValidationObservation(StrictModel):
    """Citation-validation values returned by the production generation API."""

    cited_ids: tuple[int, ...]
    valid: bool
    invalid_citations: tuple[int, ...]

    @field_validator("cited_ids", "invalid_citations", mode="before")
    @classmethod
    def parse_citation_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class GenerationObservation(StrictModel):
    status: OperationalStatus
    latency_ms: float | None = Field(default=None, ge=0)
    generation_model: str
    answer: str | None = None
    context_available: bool | None = None
    sources: tuple[GenerationSource, ...] = ()
    citation_validation: CitationValidationObservation | None = None
    error: ErrorDetail | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("generation_model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _non_empty(value, "generation_model")


class RawQuestionResult(StrictModel):
    """Appendable JSONL record containing observations, not quality judgments."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    question_id: str
    category: BenchmarkCategory
    answerability: Answerability
    started_at: datetime
    completed_at: datetime | None = None
    overall_status: OperationalStatus
    retrieval: RetrievalObservation | None = None
    generation: GenerationObservation | None = None
    retrieval_observation_mismatch: bool | None = None

    @field_validator("run_id", "experiment_id", "question_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_time(self) -> "RawQuestionResult":
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class CorrectnessLabel(str, Enum):
    CORRECT = "correct"
    MOSTLY_CORRECT = "mostly_correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NOT_APPLICABLE = "not_applicable"


class GroundingLabel(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class RefusalLabel(str, Enum):
    APPROPRIATE = "appropriate"
    FAILED_REFUSAL = "failed_refusal"
    OVER_REFUSAL = "over_refusal"
    NOT_APPLICABLE = "not_applicable"


class RelevanceLabel(str, Enum):
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"
    NOT_REVIEWED = "not_reviewed"


class FailureTag(str, Enum):
    HALLUCINATION = "hallucination"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    PARTIAL_ANSWER = "partial_answer"
    RETRIEVAL_FAILURE = "retrieval_failure"
    OVER_REFUSAL = "over_refusal"
    FAILED_REFUSAL = "failed_refusal"
    AMBIGUITY_ISSUE = "ambiguity_issue"
    CITATION_ISSUE = "citation_issue"
    REQUEST_ERROR = "request_error"


class RetrievalRelevanceAnnotation(StrictModel):
    rank: int = Field(gt=0)
    relevance: RelevanceLabel = RelevanceLabel.NOT_REVIEWED
    notes: str | None = None


class HumanAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    question_id: str
    reviewer_id: str | None = None
    annotated_at: datetime | None = None
    correctness: CorrectnessLabel | None = None
    grounding: GroundingLabel | None = None
    refusal: RefusalLabel | None = None
    retrieval_relevance: tuple[RetrievalRelevanceAnnotation, ...] = ()
    failure_tags: tuple[FailureTag, ...] = ()
    notes: str | None = None

    @field_validator("retrieval_relevance", "failure_tags", mode="before")
    @classmethod
    def parse_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("run_id", "experiment_id", "question_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_unique_annotations(self) -> "HumanAnnotation":
        ranks = [item.rank for item in self.retrieval_relevance]
        if len(ranks) != len(set(ranks)):
            raise ValueError("retrieval relevance ranks must be unique")
        if len(self.failure_tags) != len(set(self.failure_tags)):
            raise ValueError("failure tags must be unique")
        return self


class DerivedMetrics(StrictModel):
    """Nullable E7 outputs kept separate from immutable raw observations."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str
    experiment_id: str
    question_id: str
    hit_at_k: bool | None = None
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    complete_evidence_at_k: bool | None = None
    expected_point_coverage: float | None = Field(default=None, ge=0, le=1)
    expected_point_matched_count: int | None = Field(default=None, ge=0)
    expected_point_eligible_count: int | None = Field(default=None, gt=0)
    automated_refusal_signal: bool | None = None

    @model_validator(mode="after")
    def validate_expected_point_coverage(self) -> "DerivedMetrics":
        """Require an all-null proxy or counts consistent with its coverage ratio."""

        values = (
            self.expected_point_coverage,
            self.expected_point_matched_count,
            self.expected_point_eligible_count,
        )
        if all(value is None for value in values):
            return self
        if any(value is None for value in values):
            raise ValueError("expected-point coverage and counts must all be null or populated")

        assert self.expected_point_coverage is not None
        assert self.expected_point_matched_count is not None
        assert self.expected_point_eligible_count is not None
        if self.expected_point_matched_count > self.expected_point_eligible_count:
            raise ValueError("expected-point matched count cannot exceed eligible count")

        expected_coverage = (
            self.expected_point_matched_count / self.expected_point_eligible_count
        )
        # Ratios are serialized as binary floats; tolerate only representation noise.
        if not math.isclose(
            self.expected_point_coverage,
            expected_coverage,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("expected-point coverage must equal matched count / eligible count")
        return self
