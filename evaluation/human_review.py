"""Offline preparation, validation, and unblinding for human review."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evaluation.schemas import (
    Answerability,
    BenchmarkCategory,
    CorrectnessLabel,
    ExpectedBehavior,
    ExperimentConfig,
    FailureTag,
    GroundingLabel,
    HumanAnnotation,
    OperationalStatus,
    RawQuestionResult,
    RefusalLabel,
    RelevanceLabel,
    RetrievalRelevanceAnnotation,
    RunManifest,
    configuration_hash,
    experiment_id,
)
from evaluation.validation import load_benchmark_jsonl

REVIEW_SCHEMA_VERSION = "1.0.0"
ORDERING_ALGORITHM_VERSION = "hmac-sha256-greedy-adjacency-v1"
PACKET_FILENAME = "packet.jsonl"
RESPONSES_FILENAME = "responses.jsonl"
MAPPING_FILENAME = "mapping.json"
MANIFEST_FILENAME = "review_manifest.json"
VALIDATION_FILENAME = "validation.json"
PREVIEW_FILENAME = "unblinded_annotations.jsonl"


class HumanReviewError(RuntimeError):
    """Raised for invalid, unsafe, or inconsistent review artifacts."""


class _ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PacketExpectedPoint(_ReviewModel):
    point_id: str
    text: str


class PacketGoldEvidence(_ReviewModel):
    evidence_id: str
    pdf_page: int = Field(gt=0)
    printed_page_label: str | None
    section: str | None
    anchor: str
    notes: str | None = None


class PacketRetrievalItem(_ReviewModel):
    rank: int = Field(gt=0)
    retrieved_item_id: str
    chunk_text: str


class PacketGenerationSource(_ReviewModel):
    citation_id: int = Field(gt=0)
    source_id: str
    source_text: str | None
    source_text_available: bool

    @model_validator(mode="after")
    def validate_source_availability(self) -> "PacketGenerationSource":
        if self.source_text_available != (self.source_text is not None):
            raise ValueError("source_text_available must agree with source_text presence")
        return self


class PacketOperationalStatus(_ReviewModel):
    overall_status: OperationalStatus
    retrieval_status: OperationalStatus | None
    generation_status: OperationalStatus | None
    failure_category: str | None


class ReviewPacketItem(_ReviewModel):
    review_item_id: str
    question_id: str
    category: BenchmarkCategory
    answerability: Answerability
    question: str
    expected_behavior: ExpectedBehavior
    generated_answer: str | None
    reference_answer: str | None
    expected_points: tuple[PacketExpectedPoint, ...]
    benchmark_notes: str | None
    gold_evidence: tuple[PacketGoldEvidence, ...]
    observed_retrieval: tuple[PacketRetrievalItem, ...]
    generation_sources: tuple[PacketGenerationSource, ...]
    operational_status: PacketOperationalStatus
    grounding_assessable: bool

    @field_validator(
        "expected_points", "gold_evidence", "observed_retrieval", "generation_sources", mode="before"
    )
    @classmethod
    def parse_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReviewResponse(_ReviewModel):
    """Strict blinded response; canonical identities are deliberately absent."""

    review_item_id: str
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

    @model_validator(mode="after")
    def validate_unique_values(self) -> "ReviewResponse":
        ranks = [item.rank for item in self.retrieval_relevance]
        if len(ranks) != len(set(ranks)):
            raise ValueError("retrieval relevance ranks must be unique")
        if len(self.failure_tags) != len(set(self.failure_tags)):
            raise ValueError("failure tags must be unique")
        return self


class RetrievalIdentityMapping(_ReviewModel):
    retrieved_item_id: str
    rank: int = Field(gt=0)
    document_id: int = Field(gt=0)
    chunk_id: int = Field(gt=0)
    chunk_order: int = Field(ge=0)


class GenerationIdentityMapping(_ReviewModel):
    source_id: str
    citation_id: int = Field(gt=0)
    document_id: int = Field(gt=0)
    chunk_id: int = Field(gt=0)
    chunk_order: int = Field(ge=0)


class ReviewMappingItem(_ReviewModel):
    review_item_id: str
    run_id: str
    experiment_id: str
    question_id: str
    raw_result_sha256: str
    source_run_path: str
    retrieval_mappings: tuple[RetrievalIdentityMapping, ...]
    generation_source_mappings: tuple[GenerationIdentityMapping, ...]

    @field_validator("retrieval_mappings", "generation_source_mappings", mode="before")
    @classmethod
    def parse_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReviewMapping(_ReviewModel):
    schema_version: Literal[REVIEW_SCHEMA_VERSION] = REVIEW_SCHEMA_VERSION
    items: tuple[ReviewMappingItem, ...]

    @field_validator("items", mode="before")
    @classmethod
    def parse_items(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class SourceRunAudit(_ReviewModel):
    run_id: str
    experiment_id: str
    source_run_path: str
    config_sha256: str
    run_manifest_sha256: str
    raw_results_sha256: str


class ReviewManifest(_ReviewModel):
    schema_version: Literal[REVIEW_SCHEMA_VERSION] = REVIEW_SCHEMA_VERSION
    review_batch_id: str
    created_at: datetime
    ordering_algorithm_version: Literal[ORDERING_ALGORITHM_VERSION] = ORDERING_ALGORITHM_VERSION
    seed: str
    benchmark_path: str
    benchmark_id: str
    benchmark_version: str
    benchmark_sha256: str
    source_runs: tuple[SourceRunAudit, ...]
    packet_sha256: str
    mapping_sha256: str
    item_count: int = Field(ge=0)
    manifest_sha256: str

    @field_validator("source_runs", mode="before")
    @classmethod
    def parse_source_runs(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ItemValidation(_ReviewModel):
    review_item_id: str
    state: Literal["unreviewed", "partial", "complete"]
    issues: tuple[str, ...] = ()
    invalid: bool = False

    @field_validator("issues", mode="before")
    @classmethod
    def parse_issues(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ValidationReport(_ReviewModel):
    schema_version: Literal[REVIEW_SCHEMA_VERSION] = REVIEW_SCHEMA_VERSION
    review_batch_id: str
    batch_state: Literal["unreviewed", "partial", "complete"]
    item_count: int = Field(ge=0)
    unreviewed_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    complete_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    missing_response_ids: tuple[str, ...] = ()
    extra_response_ids: tuple[str, ...] = ()
    duplicate_response_ids: tuple[str, ...] = ()
    incomplete_retrieval_ranks: dict[str, tuple[int, ...]]
    integrity_status: Literal["valid"] = "valid"
    items: tuple[ItemValidation, ...]

    @field_validator(
        "missing_response_ids", "extra_response_ids", "duplicate_response_ids", "items", mode="before"
    )
    @classmethod
    def parse_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


def canonical_json(value: Any) -> str:
    """Canonical UTF-8 JSON contract with stable keys and explicit nulls."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_json(path: Path, model: BaseModel) -> None:
    _atomic_write(path, model.model_dump_json(indent=2) + "\n")


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    payload = "".join(record.model_dump_json() + "\n" for record in records)
    _atomic_write(path, payload)


def _load_jsonl(path: Path, model: type[_ReviewModel] | type[RawQuestionResult]) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HumanReviewError(f"Unable to read {path}") from exc
    records: list[Any] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise HumanReviewError(f"Blank JSONL line in {path} at line {line_number}")
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise HumanReviewError(f"Invalid record in {path} at line {line_number}: {exc}") from exc
    return records


def _manifest_digest(manifest: ReviewManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude_none=False)
    payload.pop("manifest_sha256")
    return canonical_sha256(payload)


def _raw_identity(result: RawQuestionResult) -> tuple[str, str, str]:
    return result.run_id, result.experiment_id, result.question_id


def _raw_identity_json(identity: tuple[str, str, str]) -> str:
    return canonical_json(
        {"run_id": identity[0], "experiment_id": identity[1], "question_id": identity[2]}
    )


def _ordered_results(
    results: Sequence[tuple[RawQuestionResult, Path]], seed: str
) -> list[tuple[RawQuestionResult, Path]]:
    key = seed.encode("utf-8")
    initially_ordered = sorted(results, key=lambda item: _raw_identity(item[0]))
    initially_ordered.sort(
        key=lambda item: (
            hmac.new(key, _raw_identity_json(_raw_identity(item[0])).encode("utf-8"), hashlib.sha256).digest(),
            _raw_identity(item[0]),
        )
    )
    remaining = list(initially_ordered)
    ordered: list[tuple[RawQuestionResult, Path]] = []
    previous_question: str | None = None
    while remaining:
        selected_index = next(
            (
                index
                for index, (result, _) in enumerate(remaining)
                if result.question_id != previous_question
            ),
            0,
        )
        selected = remaining.pop(selected_index)
        ordered.append(selected)
        previous_question = selected[0].question_id
    return ordered


def _load_source_run(
    run_dir: Path,
    *,
    benchmark_sha256: str,
) -> tuple[list[RawQuestionResult], SourceRunAudit]:
    resolved_run = run_dir.resolve()
    config_path = resolved_run / "config.json"
    manifest_path = resolved_run / "run_manifest.json"
    raw_path = resolved_run / "raw_results.jsonl"
    try:
        config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise HumanReviewError(f"Invalid stored run metadata in {resolved_run}") from exc
    if manifest.run_id != resolved_run.name:
        raise HumanReviewError(f"Run directory name does not match manifest: {resolved_run}")
    if manifest.config_hash != configuration_hash(config):
        raise HumanReviewError(f"Run config hash mismatch: {resolved_run}")
    if manifest.experiment_id != experiment_id(config):
        raise HumanReviewError(f"Run experiment identity mismatch: {resolved_run}")
    if config.benchmark.benchmark_sha256 != benchmark_sha256:
        raise HumanReviewError(f"Run benchmark hash differs from review benchmark: {resolved_run}")
    records = _load_jsonl(raw_path, RawQuestionResult)
    for result in records:
        if result.run_id != manifest.run_id or result.experiment_id != manifest.experiment_id:
            raise HumanReviewError(f"Raw result identity differs from run manifest: {resolved_run}")
    audit = SourceRunAudit(
        run_id=manifest.run_id,
        experiment_id=manifest.experiment_id,
        source_run_path=str(resolved_run),
        config_sha256=sha256_file(config_path),
        run_manifest_sha256=sha256_file(manifest_path),
        raw_results_sha256=sha256_file(raw_path),
    )
    return records, audit


def _packet_and_mapping_item(
    review_item_id: str,
    result: RawQuestionResult,
    run_dir: Path,
    benchmark_record: Any,
) -> tuple[ReviewPacketItem, ReviewMappingItem, ReviewResponse]:
    retrieval_results = (
        tuple(sorted(result.retrieval.results, key=lambda item: item.rank))
        if result.retrieval is not None and result.retrieval.status is OperationalStatus.SUCCESS
        else ()
    )
    packet_retrieval: list[PacketRetrievalItem] = []
    retrieval_mapping: list[RetrievalIdentityMapping] = []
    exact_chunks: dict[tuple[int, int, int], list[str]] = {}
    for index, item in enumerate(retrieval_results, 1):
        neutral_id = f"retrieved_{index:02d}"
        packet_retrieval.append(
            PacketRetrievalItem(
                rank=item.rank,
                retrieved_item_id=neutral_id,
                chunk_text=item.chunk_text,
            )
        )
        retrieval_mapping.append(
            RetrievalIdentityMapping(
                retrieved_item_id=neutral_id,
                rank=item.rank,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                chunk_order=item.chunk_order,
            )
        )
        exact_chunks.setdefault(
            (item.document_id, item.chunk_id, item.chunk_order), []
        ).append(item.chunk_text)

    packet_sources: list[PacketGenerationSource] = []
    source_mapping: list[GenerationIdentityMapping] = []
    generation_sources = result.generation.sources if result.generation is not None else ()
    for index, source in enumerate(generation_sources, 1):
        source_id = f"source_{index:02d}"
        identity = (source.document_id, source.chunk_id, source.chunk_order)
        candidates = exact_chunks.get(identity, [])
        source_text = candidates[0] if len(candidates) == 1 else None
        packet_sources.append(
            PacketGenerationSource(
                citation_id=source.citation_id,
                source_id=source_id,
                source_text=source_text,
                source_text_available=source_text is not None,
            )
        )
        source_mapping.append(
            GenerationIdentityMapping(
                source_id=source_id,
                citation_id=source.citation_id,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                chunk_order=source.chunk_order,
            )
        )

    generated_answer = result.generation.answer if result.generation is not None else None
    grounding_assessable = (
        generated_answer is not None
        and bool(packet_sources)
        and all(source.source_text_available for source in packet_sources)
    )
    packet = ReviewPacketItem(
        review_item_id=review_item_id,
        question_id=result.question_id,
        category=result.category,
        answerability=result.answerability,
        question=benchmark_record.question,
        expected_behavior=benchmark_record.expected_behavior,
        generated_answer=generated_answer,
        reference_answer=benchmark_record.reference_answer,
        expected_points=tuple(
            PacketExpectedPoint(point_id=point.id, text=point.text)
            for point in benchmark_record.expected_points
        ),
        benchmark_notes=benchmark_record.notes,
        gold_evidence=tuple(
            PacketGoldEvidence(
                evidence_id=evidence.evidence_id,
                pdf_page=evidence.pdf_page,
                printed_page_label=evidence.printed_page_label,
                section=evidence.section,
                anchor=evidence.anchor,
                notes=evidence.notes,
            )
            for evidence in benchmark_record.evidence
        ),
        observed_retrieval=tuple(packet_retrieval),
        generation_sources=tuple(packet_sources),
        operational_status=PacketOperationalStatus(
            overall_status=result.overall_status,
            retrieval_status=result.retrieval.status if result.retrieval is not None else None,
            generation_status=result.generation.status if result.generation is not None else None,
            failure_category=(
                None
                if result.overall_status is OperationalStatus.SUCCESS
                else result.overall_status.value
            ),
        ),
        grounding_assessable=grounding_assessable,
    )
    mapping = ReviewMappingItem(
        review_item_id=review_item_id,
        run_id=result.run_id,
        experiment_id=result.experiment_id,
        question_id=result.question_id,
        raw_result_sha256=canonical_sha256(result),
        source_run_path=str(run_dir.resolve()),
        retrieval_mappings=tuple(retrieval_mapping),
        generation_source_mappings=tuple(source_mapping),
    )
    response = ReviewResponse(
        review_item_id=review_item_id,
        reviewer_id=None,
        annotated_at=None,
        correctness=None,
        grounding=None,
        refusal=None,
        retrieval_relevance=tuple(
            RetrievalRelevanceAnnotation(
                rank=item.rank,
                relevance=RelevanceLabel.NOT_REVIEWED,
                notes=None,
            )
            for item in packet_retrieval
        ),
        failure_tags=(),
        notes=None,
    )
    return packet, mapping, response


def prepare_review_batch(
    *,
    run_dirs: Sequence[Path],
    benchmark_path: Path,
    seed: str,
    output_dir: Path,
    created_at: datetime | None = None,
) -> ReviewManifest:
    """Create a blinded review batch from immutable stored run artifacts."""

    if not seed:
        raise HumanReviewError("Review seed must not be blank")
    if not run_dirs:
        raise HumanReviewError("At least one source run is required")
    benchmark_path = benchmark_path.resolve()
    benchmark_sha = sha256_file(benchmark_path)
    records = load_benchmark_jsonl(benchmark_path)
    benchmark_by_id = {record.id: record for record in records}
    if len(benchmark_by_id) != len(records):
        raise HumanReviewError("Benchmark contains duplicate question IDs")
    benchmark_ids = {record.benchmark_id for record in records}
    benchmark_versions = {record.benchmark_version for record in records}
    if len(benchmark_ids) != 1 or len(benchmark_versions) != 1:
        raise HumanReviewError("Benchmark identity is inconsistent")

    all_results: list[tuple[RawQuestionResult, Path]] = []
    source_audits: list[SourceRunAudit] = []
    seen_runs: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen_runs:
            raise HumanReviewError(f"Duplicate source run path: {resolved}")
        seen_runs.add(resolved)
        run_results, audit = _load_source_run(resolved, benchmark_sha256=benchmark_sha)
        source_audits.append(audit)
        all_results.extend((result, resolved) for result in run_results)

    identities = [_raw_identity(result) for result, _ in all_results]
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicates:
        raise HumanReviewError(f"Duplicate raw identities: {duplicates}")
    for result, _ in all_results:
        benchmark_record = benchmark_by_id.get(result.question_id)
        if benchmark_record is None:
            raise HumanReviewError(f"Raw result question is absent from benchmark: {result.question_id}")
        if (
            result.category is not benchmark_record.category
            or result.answerability is not benchmark_record.answerability
        ):
            raise HumanReviewError(f"Raw result benchmark metadata mismatch: {result.question_id}")

    ordered = _ordered_results(all_results, seed)
    packets: list[ReviewPacketItem] = []
    mappings: list[ReviewMappingItem] = []
    responses: list[ReviewResponse] = []
    for index, (result, run_dir) in enumerate(ordered, 1):
        review_item_id = f"review_item_{index:06d}"
        packet, mapping, response = _packet_and_mapping_item(
            review_item_id,
            result,
            run_dir,
            benchmark_by_id[result.question_id],
        )
        packets.append(packet)
        mappings.append(mapping)
        responses.append(response)

    mapping = ReviewMapping(items=tuple(mappings))
    packet_sha = canonical_sha256([item.model_dump(mode="json") for item in packets])
    mapping_sha = canonical_sha256(mapping)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise HumanReviewError(f"Review output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = ReviewManifest(
        review_batch_id=output_dir.name,
        created_at=(created_at or datetime.now(UTC)).astimezone(UTC),
        seed=seed,
        benchmark_path=str(benchmark_path),
        benchmark_id=next(iter(benchmark_ids)),
        benchmark_version=next(iter(benchmark_versions)),
        benchmark_sha256=benchmark_sha,
        source_runs=tuple(source_audits),
        packet_sha256=packet_sha,
        mapping_sha256=mapping_sha,
        item_count=len(packets),
        manifest_sha256="pending",
    )
    manifest = manifest.model_copy(update={"manifest_sha256": _manifest_digest(manifest)})
    _write_jsonl(output_dir / PACKET_FILENAME, packets)
    _write_jsonl(output_dir / RESPONSES_FILENAME, responses)
    _write_json(output_dir / MAPPING_FILENAME, mapping)
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def _load_manifest(batch_dir: Path) -> ReviewManifest:
    try:
        return ReviewManifest.model_validate_json(
            (batch_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise HumanReviewError("Invalid review manifest") from exc


def _load_mapping(batch_dir: Path) -> ReviewMapping:
    try:
        return ReviewMapping.model_validate_json(
            (batch_dir / MAPPING_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise HumanReviewError("Invalid review mapping") from exc


def verify_batch_integrity(
    batch_dir: Path,
) -> tuple[ReviewManifest, list[ReviewPacketItem], ReviewMapping]:
    """Verify manifest, packet, mapping, and immutable source-run identities."""

    batch_dir = batch_dir.resolve()
    manifest = _load_manifest(batch_dir)
    packets = _load_jsonl(batch_dir / PACKET_FILENAME, ReviewPacketItem)
    mapping = _load_mapping(batch_dir)
    if manifest.review_batch_id != batch_dir.name:
        raise HumanReviewError("Review batch directory name differs from manifest")
    if manifest.manifest_sha256 != _manifest_digest(manifest):
        raise HumanReviewError("Review manifest integrity check failed")
    if manifest.packet_sha256 != canonical_sha256(
        [item.model_dump(mode="json") for item in packets]
    ):
        raise HumanReviewError("Reviewer packet integrity check failed")
    if manifest.mapping_sha256 != canonical_sha256(mapping):
        raise HumanReviewError("Review mapping integrity check failed")
    if manifest.item_count != len(packets) or manifest.item_count != len(mapping.items):
        raise HumanReviewError("Review item count differs across artifacts")
    packet_ids = [item.review_item_id for item in packets]
    mapping_ids = [item.review_item_id for item in mapping.items]
    if len(packet_ids) != len(set(packet_ids)):
        raise HumanReviewError("Reviewer packet contains duplicate review item IDs")
    if len(mapping_ids) != len(set(mapping_ids)):
        raise HumanReviewError("Review mapping contains duplicate review item IDs")
    if packet_ids != mapping_ids:
        raise HumanReviewError("Reviewer packet and mapping are not one-to-one in order")

    benchmark_path = Path(manifest.benchmark_path)
    if not benchmark_path.is_file() or sha256_file(benchmark_path) != manifest.benchmark_sha256:
        raise HumanReviewError("Review benchmark integrity check failed")
    source_results: dict[tuple[str, str, str], str] = {}
    for source in manifest.source_runs:
        run_dir = Path(source.source_run_path)
        config_path = run_dir / "config.json"
        run_manifest_path = run_dir / "run_manifest.json"
        raw_path = run_dir / "raw_results.jsonl"
        if (
            sha256_file(config_path) != source.config_sha256
            or sha256_file(run_manifest_path) != source.run_manifest_sha256
            or sha256_file(raw_path) != source.raw_results_sha256
        ):
            raise HumanReviewError("Source run integrity check failed")
        for result in _load_jsonl(raw_path, RawQuestionResult):
            identity = _raw_identity(result)
            if identity in source_results:
                raise HumanReviewError("Duplicate source raw identity during integrity check")
            source_results[identity] = canonical_sha256(result)
    for item in mapping.items:
        identity = (item.run_id, item.experiment_id, item.question_id)
        if source_results.get(identity) != item.raw_result_sha256:
            raise HumanReviewError("Mapped raw-result digest does not match immutable source")
    return manifest, packets, mapping


def _is_initial_response(response: ReviewResponse, expected_ranks: set[int]) -> bool:
    return (
        response.reviewer_id is None
        and response.annotated_at is None
        and response.correctness is None
        and response.grounding is None
        and response.refusal is None
        and not response.failure_tags
        and response.notes is None
        and {item.rank for item in response.retrieval_relevance} == expected_ranks
        and all(
            item.relevance is RelevanceLabel.NOT_REVIEWED and item.notes is None
            for item in response.retrieval_relevance
        )
    )


def _validate_item(packet: ReviewPacketItem, response: ReviewResponse) -> ItemValidation:
    expected_ranks = {item.rank for item in packet.observed_retrieval}
    actual_ranks = {item.rank for item in response.retrieval_relevance}
    if _is_initial_response(response, expected_ranks):
        return ItemValidation(review_item_id=packet.review_item_id, state="unreviewed")

    issues: list[str] = []
    invalid = False
    for field_name in ("correctness", "grounding", "refusal"):
        if getattr(response, field_name) is None:
            issues.append(f"missing_{field_name}")
    if response.annotated_at is None:
        issues.append("missing_annotated_at")
    missing_ranks = sorted(expected_ranks - actual_ranks)
    extra_ranks = sorted(actual_ranks - expected_ranks)
    if missing_ranks:
        issues.append(f"missing_retrieval_ranks:{missing_ranks}")
        invalid = True
    if extra_ranks:
        issues.append(f"extra_retrieval_ranks:{extra_ranks}")
        invalid = True
    if any(item.relevance is RelevanceLabel.NOT_REVIEWED for item in response.retrieval_relevance):
        issues.append("retrieval_relevance_not_reviewed")
    if not packet.grounding_assessable:
        if response.grounding is not None and response.grounding is not GroundingLabel.NOT_APPLICABLE:
            issues.append("grounding_must_be_not_applicable")
            invalid = True
        if response.grounding is GroundingLabel.NOT_APPLICABLE and not (response.notes or "").strip():
            issues.append("grounding_unassessable_note_required")
    return ItemValidation(
        review_item_id=packet.review_item_id,
        state="complete" if not issues else "partial",
        issues=tuple(issues),
        invalid=invalid,
    )


def validate_review_batch(
    batch_dir: Path,
    *,
    expected_count: int | None = None,
    write_report: bool = True,
) -> ValidationReport:
    """Validate review progress without mutating reviewer responses."""

    batch_dir = batch_dir.resolve()
    manifest, packets, _ = verify_batch_integrity(batch_dir)
    responses = _load_jsonl(batch_dir / RESPONSES_FILENAME, ReviewResponse)
    response_ids = [item.review_item_id for item in responses]
    duplicate_ids = tuple(sorted(key for key, count in Counter(response_ids).items() if count > 1))
    packet_by_id = {item.review_item_id: item for item in packets}
    response_by_id: dict[str, ReviewResponse] = {}
    for response in responses:
        response_by_id.setdefault(response.review_item_id, response)
    missing_ids = tuple(sorted(set(packet_by_id) - set(response_by_id)))
    extra_ids = tuple(sorted(set(response_by_id) - set(packet_by_id)))

    item_validations: list[ItemValidation] = []
    incomplete_ranks: dict[str, tuple[int, ...]] = {}
    for packet in packets:
        response = response_by_id.get(packet.review_item_id)
        if response is None:
            item_validations.append(
                ItemValidation(
                    review_item_id=packet.review_item_id,
                    state="partial",
                    issues=("missing_response",),
                    invalid=True,
                )
            )
            incomplete_ranks[packet.review_item_id] = tuple(
                sorted(item.rank for item in packet.observed_retrieval)
            )
            continue
        validation = _validate_item(packet, response)
        item_validations.append(validation)
        relevance_by_rank = {item.rank: item.relevance for item in response.retrieval_relevance}
        pending = tuple(
            sorted(
                item.rank
                for item in packet.observed_retrieval
                if relevance_by_rank.get(item.rank) is RelevanceLabel.NOT_REVIEWED
                or item.rank not in relevance_by_rank
            )
        )
        if pending:
            incomplete_ranks[packet.review_item_id] = pending

    counts = Counter(item.state for item in item_validations)
    external_invalid = len(extra_ids) + len(duplicate_ids)
    if expected_count is not None and manifest.item_count != expected_count:
        external_invalid += 1
    invalid_count = sum(item.invalid for item in item_validations) + external_invalid
    if (
        counts["complete"] == manifest.item_count
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
        and (expected_count is None or manifest.item_count == expected_count)
    ):
        batch_state: Literal["unreviewed", "partial", "complete"] = "complete"
    elif (
        counts["unreviewed"] == manifest.item_count
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
        and (expected_count is None or manifest.item_count == expected_count)
    ):
        batch_state = "unreviewed"
    else:
        batch_state = "partial"
    report = ValidationReport(
        review_batch_id=manifest.review_batch_id,
        batch_state=batch_state,
        item_count=manifest.item_count,
        unreviewed_count=counts["unreviewed"],
        partial_count=counts["partial"],
        complete_count=counts["complete"],
        invalid_count=invalid_count,
        missing_response_ids=missing_ids,
        extra_response_ids=extra_ids,
        duplicate_response_ids=duplicate_ids,
        incomplete_retrieval_ranks=incomplete_ranks,
        items=tuple(item_validations),
    )
    if write_report:
        _write_json(batch_dir / VALIDATION_FILENAME, report)
    return report


def _canonical_annotations(
    mapping: ReviewMapping,
    responses: Sequence[ReviewResponse],
) -> list[tuple[HumanAnnotation, Path]]:
    response_by_id = {response.review_item_id: response for response in responses}
    annotations: list[tuple[HumanAnnotation, Path]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in mapping.items:
        response = response_by_id[item.review_item_id]
        identity = (item.run_id, item.experiment_id, item.question_id)
        if identity in seen:
            raise HumanReviewError(f"Duplicate canonical annotation identity: {identity}")
        seen.add(identity)
        annotation = HumanAnnotation(
            run_id=item.run_id,
            experiment_id=item.experiment_id,
            question_id=item.question_id,
            reviewer_id=response.reviewer_id,
            annotated_at=response.annotated_at,
            correctness=response.correctness,
            grounding=response.grounding,
            refusal=response.refusal,
            retrieval_relevance=response.retrieval_relevance,
            failure_tags=response.failure_tags,
            notes=response.notes,
        )
        annotations.append((annotation, Path(item.source_run_path)))
    return annotations


def unblind_review_batch(
    batch_dir: Path,
    *,
    write_run_annotations: bool = False,
    preview_path: Path | None = None,
) -> tuple[HumanAnnotation, ...]:
    """Verify a completed batch and emit canonical strict human annotations."""

    batch_dir = batch_dir.resolve()
    _, _, mapping = verify_batch_integrity(batch_dir)
    report = validate_review_batch(batch_dir, write_report=False)
    if report.batch_state != "complete" or report.invalid_count:
        raise HumanReviewError("Canonical unblinding requires a complete valid review batch")
    responses = _load_jsonl(batch_dir / RESPONSES_FILENAME, ReviewResponse)
    response_ids = {response.review_item_id for response in responses}
    mapping_ids = {item.review_item_id for item in mapping.items}
    if response_ids != mapping_ids or len(responses) != len(response_ids):
        raise HumanReviewError("Reviewer responses do not map one-to-one to review items")
    annotations_with_runs = _canonical_annotations(mapping, responses)
    annotations = tuple(annotation for annotation, _ in annotations_with_runs)

    if write_run_annotations:
        if preview_path is not None:
            raise HumanReviewError("preview_path cannot be used with write_run_annotations")
        grouped: dict[Path, list[HumanAnnotation]] = {}
        for annotation, run_dir in annotations_with_runs:
            grouped.setdefault(run_dir.resolve(), []).append(annotation)
        targets = {run_dir / "human_annotations.jsonl": values for run_dir, values in grouped.items()}
        existing = [path for path in targets if path.exists()]
        if existing:
            raise HumanReviewError(f"Canonical annotation target already exists: {existing[0]}")
        for path, values in targets.items():
            _write_jsonl(path, values)
    else:
        destination = (preview_path or (batch_dir / PREVIEW_FILENAME)).resolve()
        if destination.parent != batch_dir:
            raise HumanReviewError("Unblinded preview must remain inside the review batch directory")
        _write_jsonl(destination, annotations)
    return annotations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline blinded DDRAG human-review tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run", action="append", type=Path, required=True, dest="runs")
    prepare.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evaluation/benchmarks/csu_handbook_v1.jsonl"),
    )
    prepare.add_argument("--seed", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--review-batch", type=Path, required=True)
    validate.add_argument("--expect-state", choices=("unreviewed", "partial", "complete"))
    validate.add_argument("--expected-count", type=int)

    unblind = subparsers.add_parser("unblind")
    unblind.add_argument("--review-batch", type=Path, required=True)
    unblind.add_argument("--output", type=Path)
    unblind.add_argument("--write-run-annotations", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            manifest = prepare_review_batch(
                run_dirs=args.runs,
                benchmark_path=args.benchmark,
                seed=args.seed,
                output_dir=args.output,
            )
            print(f"Prepared {manifest.item_count} blinded review items")
            return 0
        if args.command == "validate":
            report = validate_review_batch(
                args.review_batch,
                expected_count=args.expected_count,
            )
            print(
                f"Review batch state={report.batch_state} items={report.item_count} "
                f"invalid={report.invalid_count}"
            )
            if args.expect_state is not None and report.batch_state != args.expect_state:
                return 1
            if report.invalid_count:
                return 1
            return 0
        unblind_review_batch(
            args.review_batch,
            write_run_annotations=args.write_run_annotations,
            preview_path=args.output,
        )
        print("Unblinding completed")
        return 0
    except (HumanReviewError, OSError) as exc:
        print(f"Human review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
