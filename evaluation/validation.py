"""Benchmark/corpus validation CLI that does not start DDRAG services."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import ValidationError
from pypdf import PdfReader

from evaluation.schemas import BenchmarkRecord, CorpusManifest, ReviewStatus

CSU_V1_APPROVED_IDS = (
    "Q002", "Q003", "Q004", "Q005", "Q006", "Q007", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q016", "Q017", "Q018", "Q019",
    "Q020", "Q022", "Q023", "Q024", "Q025", "Q026", "Q027", "Q028",
    "Q029", "Q030", "Q032", "Q033", "Q034", "Q035", "Q036", "Q038",
    "Q039", "Q041", "Q042", "Q043", "Q045", "Q046", "Q047", "Q048",
)


class EvaluationValidationError(ValueError):
    """One or more evaluation artifacts failed validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def normalize_anchor_text(value: str) -> str:
    """Normalize Unicode/case and collapse whitespace; perform no fuzzy repair."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def load_corpus_manifest(path: Path) -> CorpusManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CorpusManifest.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvaluationValidationError((f"Invalid corpus manifest {path}: {exc}",)) from exc


def load_benchmark_jsonl(path: Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationValidationError((f"Unable to read benchmark {path}: {exc}",)) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}:{line_number}: blank JSONL lines are not allowed")
            continue
        try:
            records.append(BenchmarkRecord.model_validate_json(line))
        except ValidationError as exc:
            errors.append(f"{path}:{line_number}: {exc}")
    if errors:
        raise EvaluationValidationError(errors)
    return records


def validate_corpus_file(manifest: CorpusManifest, corpus_path: Path) -> tuple[int, tuple[str, ...]]:
    """Validate the single local corpus file and return page count plus page texts."""

    errors: list[str] = []
    if len(manifest.documents) != 1:
        raise EvaluationValidationError(("This validator currently requires a one-document corpus",))
    document = manifest.documents[0]
    try:
        payload = corpus_path.read_bytes()
    except OSError as exc:
        raise EvaluationValidationError((f"Unable to read corpus file {corpus_path}: {exc}",)) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != document.sha256:
        errors.append(f"Corpus SHA-256 mismatch: expected {document.sha256}, got {digest}")
    if len(payload) != document.size_bytes:
        errors.append(f"Corpus size mismatch: expected {document.size_bytes}, got {len(payload)}")
    if corpus_path.name != document.expected_filename:
        errors.append(
            f"Corpus filename mismatch: expected {document.expected_filename}, got {corpus_path.name}"
        )
    try:
        reader = PdfReader(corpus_path)
        page_texts = tuple(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        errors.append(f"Unable to parse corpus PDF {corpus_path}: {exc}")
        page_texts = ()
    if len(page_texts) != document.page_count:
        errors.append(f"Corpus page-count mismatch: expected {document.page_count}, got {len(page_texts)}")
    if errors:
        raise EvaluationValidationError(errors)
    return len(page_texts), page_texts


def validate_benchmark_records(
    records: Sequence[BenchmarkRecord],
    *,
    corpus_page_count: int,
    page_texts: Sequence[str] | None = None,
    strict_csu_v1: bool = False,
) -> None:
    """Apply reusable collection checks plus optional frozen CSU-v1 invariants."""

    errors: list[str] = []
    ids = [record.id for record in records]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        errors.append(f"Duplicate question IDs: {', '.join(duplicate_ids)}")

    evidence_ids = [item.evidence_id for record in records for item in record.evidence]
    duplicate_evidence = sorted({item for item in evidence_ids if evidence_ids.count(item) > 1})
    if duplicate_evidence:
        errors.append(f"Duplicate evidence IDs: {', '.join(duplicate_evidence)}")

    benchmark_ids = {record.benchmark_id for record in records}
    if len(benchmark_ids) != 1:
        errors.append(f"Benchmark IDs are inconsistent: {sorted(benchmark_ids)}")
    benchmark_versions = {record.benchmark_version for record in records}
    if len(benchmark_versions) != 1:
        errors.append(f"Benchmark versions are inconsistent: {sorted(benchmark_versions)}")
    review_statuses = {record.review_status for record in records}
    if len(review_statuses) != 1:
        errors.append(f"Review statuses are inconsistent: {sorted(item.value for item in review_statuses)}")

    normalized_pages = (
        tuple(normalize_anchor_text(page) for page in page_texts) if page_texts is not None else None
    )
    for record in records:
        for evidence in record.evidence:
            if evidence.pdf_page > corpus_page_count:
                errors.append(
                    f"{record.id}/{evidence.evidence_id}: pdf_page {evidence.pdf_page} exceeds "
                    f"corpus page count {corpus_page_count}"
                )
                continue
            if normalized_pages is not None:
                anchor = normalize_anchor_text(evidence.anchor)
                if anchor not in normalized_pages[evidence.pdf_page - 1]:
                    errors.append(
                        f"{record.id}/{evidence.evidence_id}: anchor not found on PDF page "
                        f"{evidence.pdf_page} after conservative normalization"
                    )

    if strict_csu_v1:
        if len(records) != 40:
            errors.append(f"CSU v1 requires exactly 40 records, got {len(records)}")
        if tuple(ids) != CSU_V1_APPROVED_IDS:
            missing = sorted(set(CSU_V1_APPROVED_IDS) - set(ids))
            unexpected = sorted(set(ids) - set(CSU_V1_APPROVED_IDS))
            errors.append(
                "CSU v1 approved IDs/order mismatch"
                f"; missing={missing or 'none'}; unexpected={unexpected or 'none'}"
            )
        if benchmark_ids != {"csu_handbook"}:
            errors.append("CSU v1 requires benchmark_id='csu_handbook'")
        if benchmark_versions != {"1.0.0"}:
            errors.append("CSU v1 requires benchmark_version='1.0.0'")
        if review_statuses != {ReviewStatus.REVIEWED}:
            errors.append("CSU v1 requires every record to be reviewed")

    if errors:
        raise EvaluationValidationError(errors)


def validate_artifacts(
    *,
    benchmark_path: Path,
    manifest_path: Path,
    corpus_path: Path,
    strict_csu_v1: bool = False,
) -> tuple[CorpusManifest, list[BenchmarkRecord]]:
    manifest = load_corpus_manifest(manifest_path)
    records = load_benchmark_jsonl(benchmark_path)
    page_count, page_texts = validate_corpus_file(manifest, corpus_path)
    validate_benchmark_records(
        records,
        corpus_page_count=page_count,
        page_texts=page_texts,
        strict_csu_v1=strict_csu_v1,
    )
    return manifest, records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DDRAG evaluation benchmark and corpus artifacts")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evaluation/benchmarks/csu_handbook_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/corpus/corpus_manifest.json"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/corpus/CSU-Student-Handbook.pdf"),
    )
    parser.add_argument("--strict-csu-v1", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        manifest, records = validate_artifacts(
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            corpus_path=args.corpus,
            strict_csu_v1=args.strict_csu_v1,
        )
    except EvaluationValidationError as exc:
        print("Validation failed:", file=sys.stderr)
        for error in exc.errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Validated {len(records)} benchmark records for corpus {manifest.corpus_id} "
        f"({manifest.documents[0].page_count} PDF pages)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
