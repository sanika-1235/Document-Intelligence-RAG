"""Synthetic artifact builders shared by offline human-review tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from evaluation.human_review import (
    ReviewPacketItem,
    ReviewResponse,
    prepare_review_batch,
    sha256_file,
)
from evaluation.schemas import (
    Answerability,
    BenchmarkCategory,
    BenchmarkIdentity,
    CitationValidationObservation,
    ChunkingConfig,
    CorpusIdentity,
    CorrectnessLabel,
    EmbeddingConfig,
    ErrorDetail,
    ExperimentConfig,
    GenerationConfig,
    GenerationObservation,
    GenerationSource,
    GroundingLabel,
    OperationalStatus,
    PromptConfig,
    RawQuestionResult,
    RefusalLabel,
    RelevanceLabel,
    RetrievalConfig,
    RetrievalObservation,
    RetrievalRelevanceAnnotation,
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


def benchmark_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "Q020",
            "benchmark_id": "synthetic_review",
            "benchmark_version": "1.0.0",
            "review_status": "reviewed",
            "category": "direct_factual",
            "question": "What does the synthetic policy require?",
            "answerability": "answerable",
            "expected_behavior": "answer",
            "reference_answer": "The policy requires the alpha condition.",
            "expected_points": [
                {"id": "P1", "text": "Alpha condition", "automated_match": False}
            ],
            "evidence": [
                {
                    "evidence_id": "Q020-E1",
                    "pdf_page": 1,
                    "printed_page_label": "1",
                    "section": "Synthetic section",
                    "anchor": "The policy requires the alpha condition.",
                    "notes": None,
                }
            ],
            "notes": "Synthetic scoring-scope note; excluded clauses must not be restored.",
        },
        {
            "id": "Q021",
            "benchmark_id": "synthetic_review",
            "benchmark_version": "1.0.0",
            "review_status": "reviewed",
            "category": "direct_factual",
            "question": "What is the synthetic duration?",
            "answerability": "answerable",
            "expected_behavior": "answer",
            "reference_answer": "The duration is two days.",
            "expected_points": [
                {"id": "P1", "text": "Two days", "automated_match": False}
            ],
            "evidence": [
                {
                    "evidence_id": "Q021-E1",
                    "pdf_page": 2,
                    "printed_page_label": "2",
                    "section": None,
                    "anchor": "The duration is two days.",
                    "notes": "Synthetic evidence note",
                }
            ],
            "notes": None,
        },
        {
            "id": "Q046",
            "benchmark_id": "synthetic_review",
            "benchmark_version": "1.0.0",
            "review_status": "reviewed",
            "category": "unanswerable",
            "question": "Who holds the synthetic office in 2030?",
            "answerability": "unanswerable",
            "expected_behavior": "refuse_or_state_insufficient_information",
            "reference_answer": None,
            "expected_points": [
                {
                    "id": "P1",
                    "text": "State that the corpus cannot establish the office-holder",
                    "automated_match": False,
                }
            ],
            "evidence": [],
            "notes": "Synthetic unanswerable-scope note.",
        },
    ]


def write_benchmark(root: Path) -> Path:
    path = root / "benchmark.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark_rows()),
        encoding="utf-8",
    )
    return path


def _config(benchmark_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        corpus=CorpusIdentity(corpus_id="synthetic", corpus_sha256="a" * 64),
        benchmark=BenchmarkIdentity(
            benchmark_id="synthetic_review",
            benchmark_version="1.0.0",
            benchmark_sha256=sha256_file(benchmark_path),
        ),
        generation=GenerationConfig(generation_model="SECRET_MODEL_SENTINEL"),
        embedding=EmbeddingConfig(
            embedding_model="SECRET_EMBEDDING_SENTINEL",
            embedding_dimension=8,
        ),
        prompt=PromptConfig(prompt_id="synthetic", prompt_sha256="b" * 64),
        chunking=ChunkingConfig(
            chunk_size_chars=100,
            chunk_overlap_chars=10,
            chunking_version="synthetic-v1",
        ),
        retrieval=RetrievalConfig(top_k=1, retrieval_strategy="synthetic"),
        sampling=SamplingConfig(
            mode=SamplingMode.RUNTIME_DEFAULT,
            temperature=None,
            top_p=None,
            seed=None,
        ),
        timeouts=TimeoutConfig(
            request_timeout_seconds=10.0,
            retrieval_timeout_seconds=5.0,
            generation_timeout_seconds=5.0,
        ),
        warmup=WarmupConfig(required=False),
    )


def create_run(
    root: Path,
    benchmark_path: Path,
    run_name: str,
    *,
    question_ids: tuple[str, ...] = ("Q020", "Q021", "Q046"),
    missing_source_questions: frozenset[str] = frozenset(),
    no_source_questions: frozenset[str] = frozenset(),
    failed_questions: frozenset[str] = frozenset(),
) -> Path:
    run_dir = root / run_name
    run_dir.mkdir()
    config = _config(benchmark_path)
    exp_id = experiment_id(config)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    rows = {str(row["id"]): row for row in benchmark_rows()}
    results: list[RawQuestionResult] = []
    for offset, question_id in enumerate(question_ids, 1):
        row = rows[question_id]
        failed = question_id in failed_questions
        variant = run_name.rsplit("_", 1)[-1]
        retrieval_result = RetrievedResult(
            rank=1,
            document_id=77,
            chunk_id=100 + offset,
            chunk_order=offset,
            distance=0.0123,
            chunk_text=f"Exact retrieved text for {question_id}.",
        )
        retrieval = RetrievalObservation(
            status=OperationalStatus.SUCCESS,
            latency_ms=12.34,
            requested_top_k=1,
            candidate_count=1,
            result_count=1,
            embedding_model="SECRET_EMBEDDING_SENTINEL",
            embedding_dimension=8,
            results=(retrieval_result,),
            error=None,
        )
        source_chunk_id = (
            9999 if question_id in missing_source_questions else retrieval_result.chunk_id
        )
        generation = GenerationObservation(
            status=(
                OperationalStatus.GENERATION_FAILURE
                if failed
                else OperationalStatus.SUCCESS
            ),
            latency_ms=56.78,
            generation_model="SECRET_MODEL_SENTINEL",
            answer=(
                None
                if failed
                else f"Exact observed answer variant {variant} for {question_id}.\n[1]"
            ),
            context_available=None if failed else True,
            sources=(
                ()
                if failed or question_id in no_source_questions
                else (
                    GenerationSource(
                        citation_id=1,
                        document_id=77,
                        chunk_id=source_chunk_id,
                        chunk_order=offset,
                        distance=0.0123,
                    ),
                )
            ),
            citation_validation=(
                None
                if failed
                else CitationValidationObservation(
                    cited_ids=(1,),
                    valid=False,
                    invalid_citations=(99,),
                )
            ),
            error=(
                ErrorDetail(
                    code="request_error",
                    message=(
                        "password=RAW_ERROR_PASSWORD; Authorization: Bearer "
                        "eyJRAW.ERROR.TOKEN; email=raw-error@example.invalid"
                    ),
                    details={"credential": "RAW_ERROR_SECRET"},
                )
                if failed
                else None
            ),
        )
        results.append(
            RawQuestionResult(
                run_id=run_name,
                experiment_id=exp_id,
                question_id=question_id,
                category=BenchmarkCategory(str(row["category"])),
                answerability=Answerability(str(row["answerability"])),
                started_at=now,
                completed_at=now,
                overall_status=generation.status,
                retrieval=retrieval,
                generation=generation,
                retrieval_observation_mismatch=question_id in missing_source_questions,
            )
        )
    manifest = RunManifest(
        run_id=run_name,
        experiment_id=exp_id,
        config_hash=configuration_hash(config),
        started_at=now,
        completed_at=now,
        status=RunStatus.COMPLETED,
        corpus=config.corpus,
        benchmark=config.benchmark,
        git_commit_sha="c" * 40,
        git_dirty=False,
        python_version="3.13",
        platform=(
            "password=SECRET_PASSWORD_SENTINEL; "
            "Authorization: Bearer eyJSECRET.JWT.SENTINEL; "
            "email=secret@example.invalid"
        ),
        project_version="0.1.0",
        ollama_version=None,
        hardware=None,
        warmup_status=WarmupStatus.NOT_REQUIRED,
        question_total=len(results),
        success_total=sum(result.overall_status is OperationalStatus.SUCCESS for result in results),
        failure_total=sum(result.overall_status is not OperationalStatus.SUCCESS for result in results),
    )
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "raw_results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results),
        encoding="utf-8",
    )
    return run_dir


def create_batch(
    root: Path,
    *,
    run_count: int = 1,
    seed: str = "synthetic-seed",
    question_ids: tuple[str, ...] = ("Q020", "Q021", "Q046"),
    missing_source_questions: frozenset[str] = frozenset(),
    failed_questions: frozenset[str] = frozenset(),
    batch_name: str = "review_batch",
) -> tuple[Path, Path, list[Path]]:
    benchmark = write_benchmark(root)
    runs = [
        create_run(
            root,
            benchmark,
            f"run_synthetic_{index}",
            question_ids=question_ids,
            missing_source_questions=missing_source_questions,
            failed_questions=failed_questions,
        )
        for index in range(1, run_count + 1)
    ]
    batch = root / batch_name
    prepare_review_batch(
        run_dirs=runs,
        benchmark_path=benchmark,
        seed=seed,
        output_dir=batch,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    return batch, benchmark, runs


def read_packets(batch: Path) -> list[ReviewPacketItem]:
    return [
        ReviewPacketItem.model_validate_json(line)
        for line in (batch / "packet.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def read_responses(batch: Path) -> list[ReviewResponse]:
    return [
        ReviewResponse.model_validate_json(line)
        for line in (batch / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def write_responses(batch: Path, responses: list[ReviewResponse]) -> None:
    (batch / "responses.jsonl").write_text(
        "".join(response.model_dump_json() + "\n" for response in responses),
        encoding="utf-8",
    )


def completed_response(packet: ReviewPacketItem, response: ReviewResponse) -> ReviewResponse:
    assessable = packet.grounding_assessable
    return response.model_copy(
        update={
            "reviewer_id": "synthetic-reviewer",
            "annotated_at": datetime(2026, 1, 3, tzinfo=UTC),
            "correctness": CorrectnessLabel.CORRECT,
            "grounding": (
                GroundingLabel.GROUNDED if assessable else GroundingLabel.NOT_APPLICABLE
            ),
            "refusal": RefusalLabel.NOT_APPLICABLE,
            "retrieval_relevance": tuple(
                item.model_copy(update={"relevance": RelevanceLabel.RELEVANT})
                for item in response.retrieval_relevance
            ),
            "failure_tags": (),
            "notes": None if assessable else "Exact generation source text was unavailable.",
        }
    )
