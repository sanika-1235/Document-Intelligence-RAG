"""Focused E5 tests for benchmark and corpus validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.schemas import BenchmarkRecord
from evaluation.validation import (
    EvaluationValidationError,
    load_benchmark_jsonl,
    load_corpus_manifest,
    validate_artifacts,
    validate_benchmark_records,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "evaluation/benchmarks/csu_handbook_v1.jsonl"
MANIFEST = ROOT / "evaluation/corpus/corpus_manifest.json"
CORPUS = ROOT / "evaluation/corpus/CSU-Student-Handbook.pdf"

pytestmark = pytest.mark.skipif(
    not (BENCHMARK.is_file() and CORPUS.is_file()),
    reason="requires authorized local CSU benchmark and corpus artifacts",
)


def _record_data(record: BenchmarkRecord) -> dict[str, object]:
    return record.model_dump(mode="json")


def _parse(data: dict[str, object]) -> BenchmarkRecord:
    return BenchmarkRecord.model_validate_json(json.dumps(data))


def test_official_benchmark_validates() -> None:
    manifest, records = validate_artifacts(
        benchmark_path=BENCHMARK,
        manifest_path=MANIFEST,
        corpus_path=CORPUS,
        strict_csu_v1=True,
    )
    assert manifest.corpus_id == "csu_student_handbook_2019"
    assert len(records) == 40


def test_duplicate_question_id_rejected() -> None:
    records = load_benchmark_jsonl(BENCHMARK)
    duplicate = _record_data(records[1])
    duplicate["id"] = records[0].id
    with pytest.raises(EvaluationValidationError, match="Duplicate question IDs"):
        validate_benchmark_records(
            [records[0], _parse(duplicate)], corpus_page_count=93
        )


def test_duplicate_evidence_id_rejected() -> None:
    records = load_benchmark_jsonl(BENCHMARK)
    duplicate = _record_data(records[1])
    duplicate["evidence"][0]["evidence_id"] = records[0].evidence[0].evidence_id  # type: ignore[index]
    with pytest.raises(EvaluationValidationError, match="Duplicate evidence IDs"):
        validate_benchmark_records(
            [records[0], _parse(duplicate)], corpus_page_count=93
        )


def test_invalid_category_rejected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["category"] = "not_a_category"
    with pytest.raises(ValidationError):
        _parse(data)


def test_invalid_answerability_rejected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["answerability"] = "sometimes"
    with pytest.raises(ValidationError):
        _parse(data)


def test_answerable_without_evidence_rejected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["evidence"] = []
    with pytest.raises(ValidationError, match="require gold evidence"):
        _parse(data)


def test_unanswerable_with_evidence_rejected() -> None:
    records = load_benchmark_jsonl(BENCHMARK)
    unanswerable = _record_data(next(record for record in records if record.id == "Q046"))
    unanswerable["evidence"] = [_record_data(records[0])["evidence"][0]]  # type: ignore[index]
    with pytest.raises(ValidationError, match="empty evidence"):
        _parse(unanswerable)


def test_unanswerable_with_reference_answer_rejected() -> None:
    records = load_benchmark_jsonl(BENCHMARK)
    unanswerable = _record_data(next(record for record in records if record.id == "Q046"))
    unanswerable["reference_answer"] = "Invented answer"
    with pytest.raises(ValidationError, match="reference_answer=null"):
        _parse(unanswerable)


def test_invalid_pdf_page_rejected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["evidence"][0]["pdf_page"] = 94  # type: ignore[index]
    with pytest.raises(EvaluationValidationError, match="exceeds corpus page count"):
        validate_benchmark_records([_parse(data)], corpus_page_count=93)


def test_missing_anchor_rejected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["evidence"][0]["anchor"] = "   "  # type: ignore[index]
    with pytest.raises(ValidationError, match="anchor must not be blank"):
        _parse(data)


def test_anchor_mismatch_detected() -> None:
    data = _record_data(load_benchmark_jsonl(BENCHMARK)[0])
    data["evidence"][0]["anchor"] = "This phrase is not in the handbook"
    with pytest.raises(EvaluationValidationError, match="anchor not found"):
        validate_benchmark_records(
            [_parse(data)], corpus_page_count=93, page_texts=("",) * 93
        )


def test_manifest_loads_with_existing_field_names() -> None:
    manifest = load_corpus_manifest(MANIFEST)
    assert manifest.manifest_version == "1.0"
    assert manifest.documents[0].expected_filename == "CSU-Student-Handbook.pdf"
