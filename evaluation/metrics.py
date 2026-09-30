"""Reusable offline metrics with explicitly supplied expected-point matchers.

Expected-point coverage is an automated answer-content proxy. It is not a
semantic correctness, grounding, or overall-accuracy judgment.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evaluation.schemas import (
    Answerability,
    BenchmarkRecord,
    DerivedMetrics,
    ExpectedBehavior,
    GenerationObservation,
    OperationalStatus,
    RawQuestionResult,
    RetrievalObservation,
)
from evaluation.validation import normalize_anchor_text

PointKey = tuple[str, str]
PointMatchFunction = Callable[[str], bool]
MetricName = Literal[
    "hit_at_k",
    "recall_at_k",
    "complete_evidence_at_k",
    "expected_point_coverage",
]

_CLAUSE_BOUNDARY = re.compile(r"[\n.?!;]+")
_DECIMAL_POINT = "\ue000"
_DASH_TRANSLATION = str.maketrans(
    {
        "-": " ",
        "‐": " ",
        "‑": " ",
        "‒": " ",
        "–": " ",
        "—": " ",
        "―": " ",
        "−": " ",
    }
)
_TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?|%|[^\W\d_]+", re.UNICODE)


class MetricContractError(ValueError):
    """Raised when benchmark metadata and the frozen metric contract differ."""


@dataclass(frozen=True)
class PointMatcher:
    """Inspectable frozen matcher metadata for one automatable expected point."""

    expected_text: str
    description: str
    match: PointMatchFunction


@dataclass(frozen=True)
class MetricAggregate:
    n: int
    mean: float | None


@dataclass(frozen=True)
class RefusalSignalAggregate:
    eligible_n: int
    true_count: int
    false_count: int
    signal_rate: float | None


@dataclass(frozen=True)
class LatencyAggregate:
    n: int
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    p95: float | None


def normalize_answer_clauses(value: str) -> tuple[tuple[str, ...], ...]:
    """Normalize an answer and return deterministic sentence/clause token groups."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_POINT, normalized)
    raw_clauses = _CLAUSE_BOUNDARY.split(normalized)
    clauses: list[tuple[str, ...]] = []
    for raw_clause in raw_clauses:
        clause = raw_clause.translate(_DASH_TRANSLATION).replace(_DECIMAL_POINT, ".")
        tokens = tuple(_TOKEN_PATTERN.findall(clause))
        if tokens:
            clauses.append(tokens)
    return tuple(clauses)


def _answer_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for clause in normalize_answer_clauses(value) for token in clause)


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    width = len(phrase)
    if width == 0 or width > len(tokens):
        return False
    expected = tuple(phrase)
    return any(tuple(tokens[index : index + width]) == expected for index in range(len(tokens) - width + 1))


Phrase = tuple[str, ...]
Component = tuple[Phrase, ...]


def same_clause_matches(value: str, *components: Component) -> bool:
    return any(
        all(any(_contains_phrase(clause, alternative) for alternative in component) for component in components)
        for clause in normalize_answer_clauses(value)
    )


def whole_token_matcher(token: str) -> PointMatchFunction:
    return lambda answer: token in _answer_tokens(answer)


def validate_matcher_registry(
    records: Sequence[BenchmarkRecord],
    matchers: Mapping[PointKey, PointMatcher],
) -> None:
    """Fail when automatable benchmark points and the frozen registry drift."""

    official = {
        (record.id, point.id): point.text
        for record in records
        for point in record.expected_points
        if point.automated_match
    }
    registered = set(matchers)
    official_keys = set(official)
    missing = sorted(official_keys - registered)
    extra = sorted(registered - official_keys)
    changed = sorted(
        key
        for key in official_keys & registered
        if official[key] != matchers[key].expected_text
    )
    if missing or extra or changed:
        raise MetricContractError(
            "Automatable expected-point registry drift: "
            f"missing={missing or 'none'}; extra={extra or 'none'}; "
            f"changed_text={changed or 'none'}"
        )


def match_expected_point(
    question_id: str,
    point_id: str,
    answer: str,
    *,
    matchers: Mapping[PointKey, PointMatcher],
) -> bool:
    """Apply one frozen deterministic expected-point matcher."""

    key = (question_id, point_id)
    try:
        matcher = matchers[key]
    except KeyError as exc:
        raise MetricContractError(f"No automatable expected-point matcher for {key}") from exc
    return matcher.match(answer)


def _validate_record_matchers(
    record: BenchmarkRecord,
    matchers: Mapping[PointKey, PointMatcher],
) -> None:
    for point in record.expected_points:
        if not point.automated_match:
            continue
        key = (record.id, point.id)
        matcher = matchers.get(key)
        if matcher is None:
            raise MetricContractError(f"No automatable expected-point matcher for {key}")
        if matcher.expected_text != point.text:
            raise MetricContractError(f"Automatable expected-point text drift for {key}")


def _retrieval_metrics(
    record: BenchmarkRecord,
    retrieval: RetrievalObservation | None,
    *,
    k: int,
) -> tuple[bool | None, float | None, bool | None]:
    if (
        record.answerability not in {Answerability.ANSWERABLE, Answerability.AMBIGUOUS}
        or not record.evidence
        or retrieval is None
        or retrieval.status is not OperationalStatus.SUCCESS
    ):
        return None, None, None

    chunks = [
        normalize_anchor_text(item.chunk_text)
        for item in sorted(retrieval.results, key=lambda item: item.rank)
        if item.rank <= k
    ]
    retrieved_count = sum(
        any(normalize_anchor_text(evidence.anchor) in chunk for chunk in chunks)
        for evidence in record.evidence
    )
    evidence_count = len(record.evidence)
    return retrieved_count > 0, retrieved_count / evidence_count, retrieved_count == evidence_count


def _expected_point_metrics(
    record: BenchmarkRecord,
    generation: GenerationObservation | None,
    matchers: Mapping[PointKey, PointMatcher],
) -> tuple[float | None, int | None, int | None]:
    automatable = tuple(point for point in record.expected_points if point.automated_match)
    if not automatable:
        return None, None, None
    _validate_record_matchers(record, matchers)
    if (
        generation is None
        or generation.status is not OperationalStatus.SUCCESS
        or generation.answer is None
    ):
        return None, None, None

    eligible_count = len(automatable)
    matched_count = sum(
        match_expected_point(record.id, point.id, generation.answer, matchers=matchers)
        for point in automatable
    )
    return matched_count / eligible_count, matched_count, eligible_count


_REFUSAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:the )?(?:handbook|corpus) (?:does not|doesn t) (?:provide|state|specify|list|establish|include|give|contain)\b",
        r"\bnot (?:provided|specified|available|given|stated|listed|established) (?:in|from) (?:the )?(?:handbook|corpus)\b",
        r"\b(?:cannot|can not|unable to) (?:be )?(?:determined|verified|established) (?:from|using|based on) (?:the )?(?:handbook|corpus)\b",
        r"\binsufficient information (?:in|from) (?:the )?(?:handbook|corpus)\b",
        r"\b(?:handbook|corpus)\b.*\b(?:missing|insufficient|does not|doesn t)\b.*\b(?:invent|guess|speculate)\b",
    )
)


def automated_refusal_signal(
    record: BenchmarkRecord,
    generation: GenerationObservation | None,
) -> bool | None:
    """Detect corpus-insufficiency language for schema-eligible unanswerable items."""

    if (
        record.answerability is not Answerability.UNANSWERABLE
        or record.expected_behavior
        is not ExpectedBehavior.REFUSE_OR_STATE_INSUFFICIENT_INFORMATION
        or generation is None
        or generation.status is not OperationalStatus.SUCCESS
        or generation.answer is None
    ):
        return None
    clauses = (" ".join(clause) for clause in normalize_answer_clauses(generation.answer))
    return any(pattern.search(clause) for clause in clauses for pattern in _REFUSAL_PATTERNS)


def calculate_question_metrics(
    record: BenchmarkRecord,
    result: RawQuestionResult,
    *,
    k: int | None = None,
    matchers: Mapping[PointKey, PointMatcher] | None = None,
) -> DerivedMetrics:
    """Calculate offline metrics; automatable points require an explicit registry.

    No benchmark-specific registry is loaded implicitly. Missing or drifted
    matcher definitions fail clearly instead of silently changing the metric.
    """

    if record.id != result.question_id:
        raise MetricContractError("Benchmark record and raw result question IDs differ")
    resolved_k = k if k is not None else (
        result.retrieval.requested_top_k if result.retrieval is not None else 1
    )
    if resolved_k < 1:
        raise MetricContractError("K must be greater than zero")

    hit, recall, complete = _retrieval_metrics(record, result.retrieval, k=resolved_k)
    coverage, matched_count, eligible_count = _expected_point_metrics(
        record, result.generation, matchers if matchers is not None else {}
    )
    return DerivedMetrics(
        run_id=result.run_id,
        experiment_id=result.experiment_id,
        question_id=result.question_id,
        hit_at_k=hit,
        recall_at_k=recall,
        complete_evidence_at_k=complete,
        expected_point_coverage=coverage,
        expected_point_matched_count=matched_count,
        expected_point_eligible_count=eligible_count,
        automated_refusal_signal=automated_refusal_signal(record, result.generation),
    )


def aggregate_question_metric(
    metrics: Iterable[DerivedMetrics],
    metric: MetricName,
) -> MetricAggregate:
    """Return a macro mean over non-null eligible question-level values."""

    values = [float(value) for item in metrics if (value := getattr(item, metric)) is not None]
    return MetricAggregate(n=len(values), mean=statistics.fmean(values) if values else None)


def aggregate_refusal_signals(metrics: Iterable[DerivedMetrics]) -> RefusalSignalAggregate:
    values = [item.automated_refusal_signal for item in metrics]
    true_count = sum(value is True for value in values)
    false_count = sum(value is False for value in values)
    eligible_n = true_count + false_count
    return RefusalSignalAggregate(
        eligible_n=eligible_n,
        true_count=true_count,
        false_count=false_count,
        signal_rate=true_count / eligible_n if eligible_n else None,
    )


def _aggregate_latency_values(values: Sequence[float]) -> LatencyAggregate:
    if not values:
        return LatencyAggregate(0, None, None, None, None, None)
    ordered = sorted(values)
    p95 = (
        ordered[0]
        if len(ordered) == 1
        else statistics.quantiles(ordered, n=100, method="inclusive")[94]
    )
    return LatencyAggregate(
        n=len(ordered),
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        min=ordered[0],
        max=ordered[-1],
        p95=p95,
    )


def aggregate_retrieval_latencies(
    results: Iterable[RawQuestionResult],
) -> LatencyAggregate:
    values = [
        observation.latency_ms
        for result in results
        if (observation := result.retrieval) is not None
        and observation.status is OperationalStatus.SUCCESS
        and observation.latency_ms is not None
    ]
    return _aggregate_latency_values(values)


def aggregate_generation_latencies(
    results: Iterable[RawQuestionResult],
) -> LatencyAggregate:
    values = [
        observation.latency_ms
        for result in results
        if (observation := result.generation) is not None
        and observation.status is OperationalStatus.SUCCESS
        and observation.latency_ms is not None
    ]
    return _aggregate_latency_values(values)


__all__ = [
    "LatencyAggregate",
    "MetricAggregate",
    "MetricContractError",
    "PointKey",
    "PointMatcher",
    "RefusalSignalAggregate",
    "aggregate_generation_latencies",
    "aggregate_question_metric",
    "aggregate_refusal_signals",
    "aggregate_retrieval_latencies",
    "automated_refusal_signal",
    "calculate_question_metrics",
    "match_expected_point",
    "normalize_answer_clauses",
    "same_clause_matches",
    "whole_token_matcher",
    "validate_matcher_registry",
]
