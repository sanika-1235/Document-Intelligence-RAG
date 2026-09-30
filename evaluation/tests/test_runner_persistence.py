"""E6A tests for incremental persistence, interruption, resume, and secrets."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import pytest

from evaluation.runner import (
    AuthenticationError,
    DuplicateQuestionResultError,
    EvaluationRunner,
    ResultJournal,
    RunStore,
    RunnerConfigurationError,
    RunnerError,
    sha256_file,
)
from evaluation.schemas import (
    Answerability,
    BenchmarkCategory,
    BenchmarkRecord,
    ExperimentConfig,
    GenerationObservation,
    GenerationSource,
    OperationalStatus,
    RawQuestionResult,
    RetrievalObservation,
    RetrievedResult,
    RunStatus,
    WarmupStatus,
)
from evaluation.validation import load_benchmark_jsonl

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "evaluation/benchmarks/csu_handbook_v1.jsonl"

pytestmark = pytest.mark.skipif(
    not BENCHMARK.is_file(),
    reason="requires authorized local CSU benchmark artifact",
)


def _config(*, warmup: bool = False, top_k: int = 2) -> ExperimentConfig:
    data = {
        "schema_version": "1.0.0",
        "corpus": {"corpus_id": "fixture", "corpus_sha256": "a" * 64},
        "benchmark": {
            "benchmark_id": "csu_handbook",
            "benchmark_version": "1.0.0",
            "benchmark_sha256": sha256_file(BENCHMARK),
        },
        "generation": {"generation_model": "fixture-generation"},
        "embedding": {"embedding_model": "fixture-embedding", "embedding_dimension": 8},
        "prompt": {"prompt_id": "fixture", "prompt_sha256": "b" * 64},
        "chunking": {
            "chunk_size_chars": 100,
            "chunk_overlap_chars": 10,
            "chunking_version": "fixture",
        },
        "retrieval": {"top_k": top_k, "retrieval_strategy": "fixture"},
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
        "warmup": {"required": warmup},
    }
    return ExperimentConfig.model_validate_json(json.dumps(data))


def _times() -> Callable[[], datetime]:
    current = datetime(2030, 1, 1, tzinfo=UTC)

    def next_time() -> datetime:
        nonlocal current
        value = current
        current += timedelta(seconds=1)
        return value

    return next_time


class FakeApi:
    def __init__(
        self,
        *,
        interrupt_on_retrieval: int | None = None,
        retrieval_status: OperationalStatus = OperationalStatus.SUCCESS,
        generation_status: OperationalStatus = OperationalStatus.SUCCESS,
        authentication_failure: bool = False,
        warmup_status: OperationalStatus = OperationalStatus.SUCCESS,
    ) -> None:
        self.interrupt_on_retrieval = interrupt_on_retrieval
        self.retrieval_status = retrieval_status
        self.generation_status = generation_status
        self.authentication_failure = authentication_failure
        self.warmup_status = warmup_status
        self.retrieve_calls = 0
        self.generate_calls = 0
        self.warmup_calls = 0
        self.access_token = "sentinel-token"

    def authenticate(self, email: str, password: str) -> None:
        if self.authentication_failure:
            raise AuthenticationError("Authentication failed safely")

    def retrieve(self, question: str) -> RetrievalObservation:
        self.retrieve_calls += 1
        if self.interrupt_on_retrieval == self.retrieve_calls:
            raise RuntimeError("simulated interruption")
        results = (
            RetrievedResult(
                rank=1,
                document_id=1,
                chunk_id=2,
                chunk_order=3,
                distance=0.1,
                chunk_text="complete retrieved text",
            ),
        )
        return RetrievalObservation(
            status=self.retrieval_status,
            latency_ms=1.0,
            requested_top_k=2,
            candidate_count=1,
            result_count=1,
            embedding_model="fixture-embedding",
            embedding_dimension=8,
            results=results,
        )

    def generate(self, question: str) -> GenerationObservation:
        self.generate_calls += 1
        return GenerationObservation(
            status=self.generation_status,
            latency_ms=2.0,
            generation_model="fixture-generation",
            answer="answer",
            context_available=True,
            sources=(
                GenerationSource(
                    citation_id=1,
                    document_id=1,
                    chunk_id=2,
                    chunk_order=3,
                    distance=9.9,
                ),
            ),
        )

    def warmup(self, question: str) -> GenerationObservation:
        self.warmup_calls += 1
        return GenerationObservation(
            status=self.warmup_status,
            latency_ms=2.0,
            generation_model="fixture-generation",
        )


def _result(run_id: str, experiment_id: str, question_id: str) -> RawQuestionResult:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RawQuestionResult(
        run_id=run_id,
        experiment_id=experiment_id,
        question_id=question_id,
        category=BenchmarkCategory.DIRECT_FACTUAL,
        answerability=Answerability.ANSWERABLE,
        started_at=now,
        completed_at=now,
        overall_status=OperationalStatus.SUCCESS,
    )


def test_incremental_jsonl_and_duplicate_rejection(tmp_path: Path) -> None:
    path = tmp_path / "raw_results.jsonl"
    journal = ResultJournal(path, run_id="run_fixture", experiment_id_value="exp_fixture")
    journal.append(_result("run_fixture", "exp_fixture", "Q001"))
    journal.append(_result("run_fixture", "exp_fixture", "Q002"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [RawQuestionResult.model_validate_json(line).question_id for line in lines] == [
        "Q001",
        "Q002",
    ]
    with pytest.raises(DuplicateQuestionResultError):
        journal.append(_result("run_fixture", "exp_fixture", "Q002"))


def test_interruption_preserves_result_and_resume_skips_it(tmp_path: Path) -> None:
    records = load_benchmark_jsonl(BENCHMARK)[:3]
    config = _config()
    store = RunStore.create(
        runs_root=tmp_path,
        config=config,
        question_total=3,
        run_id="run_resume_fixture",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    interrupted_api = FakeApi(interrupt_on_retrieval=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        EvaluationRunner(
            store=store,
            records=records,
            api=interrupted_api,
            email="eval@example.invalid",
            password="sentinel-password",
            now=_times(),
        ).run()
    assert len((store.run_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert store.manifest.status is RunStatus.PARTIAL

    resumed = RunStore.resume(
        run_dir=store.run_dir,
        expected_config=config,
        benchmark_path=BENCHMARK,
    )
    resumed_api = FakeApi()
    final = EvaluationRunner(
        store=resumed,
        records=records,
        api=resumed_api,
        email="eval@example.invalid",
        password="sentinel-password",
        now=_times(),
    ).run()
    lines = (store.run_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert len({RawQuestionResult.model_validate_json(line).question_id for line in lines}) == 3
    assert resumed_api.retrieve_calls == 2
    assert final.status is RunStatus.COMPLETED


def test_resume_rejects_changed_config_and_benchmark(tmp_path: Path) -> None:
    store = RunStore.create(
        runs_root=tmp_path,
        config=_config(),
        question_total=1,
        run_id="run_reject_fixture",
    )
    with pytest.raises(RunnerConfigurationError, match="configuration differs"):
        RunStore.resume(
            run_dir=store.run_dir,
            expected_config=_config(top_k=3),
            benchmark_path=BENCHMARK,
        )
    changed_benchmark = tmp_path / "changed.jsonl"
    changed_benchmark.write_bytes(BENCHMARK.read_bytes() + b"\n")
    with pytest.raises(RunnerConfigurationError, match="benchmark bytes differ"):
        RunStore.resume(
            run_dir=store.run_dir,
            expected_config=_config(),
            benchmark_path=changed_benchmark,
        )


def test_resume_rejects_changed_persisted_experiment_id(tmp_path: Path) -> None:
    store = RunStore.create(
        runs_root=tmp_path,
        config=_config(),
        question_total=1,
        run_id="run_exp_fixture",
    )
    manifest_path = store.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_id"] = "exp_changed0000"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunnerConfigurationError, match="experiment ID"):
        RunStore.resume(
            run_dir=store.run_dir,
            expected_config=_config(),
            benchmark_path=BENCHMARK,
        )


@pytest.mark.parametrize(
    ("retrieval_status", "generation_status", "expected"),
    [
        (
            OperationalStatus.RETRIEVAL_FAILURE,
            OperationalStatus.SUCCESS,
            OperationalStatus.RETRIEVAL_FAILURE,
        ),
        (
            OperationalStatus.SUCCESS,
            OperationalStatus.GENERATION_FAILURE,
            OperationalStatus.GENERATION_FAILURE,
        ),
        (
            OperationalStatus.TIMEOUT,
            OperationalStatus.GENERATION_FAILURE,
            OperationalStatus.TIMEOUT,
        ),
    ],
)
def test_operational_failures_are_raw_statuses_not_scores(
    tmp_path: Path,
    retrieval_status: OperationalStatus,
    generation_status: OperationalStatus,
    expected: OperationalStatus,
) -> None:
    record = load_benchmark_jsonl(BENCHMARK)[0]
    store = RunStore.create(
        runs_root=tmp_path,
        config=_config(),
        question_total=1,
        run_id=f"run_{expected.value}",
    )
    api = FakeApi(retrieval_status=retrieval_status, generation_status=generation_status)
    EvaluationRunner(
        store=store,
        records=[record],
        api=api,
        email="eval@example.invalid",
        password="sentinel-password",
        now=_times(),
    ).run()
    result = RawQuestionResult.model_validate_json(
        (store.run_dir / "raw_results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert result.overall_status is expected
    assert "correctness" not in RawQuestionResult.model_fields
    assert api.retrieve_calls == 1
    assert api.generate_calls == 1


def test_authentication_failure_records_no_question_result(tmp_path: Path) -> None:
    store = RunStore.create(
        runs_root=tmp_path,
        config=_config(),
        question_total=1,
        run_id="run_auth_failure",
    )
    with pytest.raises(AuthenticationError):
        EvaluationRunner(
            store=store,
            records=load_benchmark_jsonl(BENCHMARK)[:1],
            api=FakeApi(authentication_failure=True),
            email="eval@example.invalid",
            password="sentinel-password",
        ).run()
    assert (store.run_dir / "raw_results.jsonl").read_text(encoding="utf-8") == ""
    assert store.manifest.status is RunStatus.FAILED


def test_warmup_is_unscored_and_failure_prevents_questions(tmp_path: Path) -> None:
    records = load_benchmark_jsonl(BENCHMARK)[:1]
    successful_store = RunStore.create(
        runs_root=tmp_path,
        config=_config(warmup=True),
        question_total=1,
        run_id="run_warmup_success",
    )
    success_api = FakeApi()
    manifest = EvaluationRunner(
        store=successful_store,
        records=records,
        api=success_api,
        email="eval@example.invalid",
        password="sentinel-password",
        warmup_question="not an official benchmark question",
        now=_times(),
    ).run()
    assert success_api.warmup_calls == 1
    assert len((successful_store.run_dir / "raw_results.jsonl").read_text().splitlines()) == 1
    assert manifest.warmup_status is WarmupStatus.COMPLETED
    assert manifest.question_total == 1

    failed_store = RunStore.create(
        runs_root=tmp_path,
        config=_config(warmup=True),
        question_total=1,
        run_id="run_warmup_failure",
    )
    with pytest.raises(RunnerError, match="Warm-up failed"):
        EvaluationRunner(
            store=failed_store,
            records=records,
            api=FakeApi(warmup_status=OperationalStatus.GENERATION_FAILURE),
            email="eval@example.invalid",
            password="sentinel-password",
            warmup_question="not an official benchmark question",
        ).run()
    assert (failed_store.run_dir / "raw_results.jsonl").read_text() == ""


def test_secret_sentinels_never_enter_artifacts(tmp_path: Path) -> None:
    store = RunStore.create(
        runs_root=tmp_path,
        config=_config(),
        question_total=1,
        run_id="run_secret_fixture",
        runtime_metadata={"platform": "safe-platform"},
    )
    EvaluationRunner(
        store=store,
        records=load_benchmark_jsonl(BENCHMARK)[:1],
        api=FakeApi(),
        email="sentinel-email@example.invalid",
        password="sentinel-password",
        now=_times(),
    ).run()
    artifacts = "\n".join(
        (store.run_dir / filename).read_text(encoding="utf-8")
        for filename in ("config.json", "run_manifest.json", "raw_results.jsonl")
    )
    assert "sentinel-password" not in artifacts
    assert "sentinel-token" not in artifacts
    assert "sentinel-email@example.invalid" not in artifacts
