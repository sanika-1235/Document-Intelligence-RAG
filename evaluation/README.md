# DDRAG Evaluation

DDRAG evaluates how a local document-question-answering system retrieves required evidence and turns it into supported answers. Start with the [evaluation article](reports/ddrag_evaluation_article.md) for the findings; this guide explains the framework and methodology. The source PDF and handbook-derived benchmark are intentionally not redistributed.

## What is evaluated

The framework keeps these dimensions distinct:

- retrieval coverage;
- answer correctness;
- grounding in retrieved context;
- refusal behavior;
- operational reliability;
- retrieval and generation latency.

DDRAG does not report one overall accuracy score or a composite model/prompt winner score.

## Corpus

The official corpus is the 93-page Caraga State University Student Handbook identified as `csu_student_handbook_2019`. Its PDF is not redistributed. The public [corpus policy and hash](corpus/README.md) explain the authorized-local-input requirement.

## Benchmark

The official frozen benchmark has ID `csu_handbook`, version `1.0.0`, 40 questions, and SHA-256 `f2d5e90b252f288572f617d56e27f1a003e18b1171421881c636793e32137ae0`. It contains 34 answerable, 3 ambiguous, and 3 unanswerable questions.

The public repository does not redistribute the questions, reference answers, evidence text, quoted handbook anchors, or substantive review notes. See [benchmark metadata](benchmarks/README.md). Exact official-result reproduction requires an authorized local benchmark artifact matching the published hash.

## Evaluation dimensions

- Hit@5 asks whether any required evidence unit appears in the first five retrieved chunks.
- Recall@5 measures the fraction of required evidence units retrieved in those chunks.
- CompleteEvidence@5 requires all required evidence units to be present in those chunks.
- Correctness, grounding, and refusal are independent qualitative judgments.
- ExpectedPointCoverage is an **automated answer-content proxy**, not semantic correctness, grounding, or overall accuracy.
- Operational reliability records successful and failed requests independently of answer quality.
- Retrieval latency and generation-endpoint latency are recorded separately; the latter includes retrieval within `/generation` and is not pure model-compute time. The runner's combined/end-to-end observation additionally includes its separate retrieval observation; it is not browser-to-answer latency.

Evidence matching uses Unicode normalization, case folding, whitespace collapse, and complete benchmark anchors. Retrieval averages use 37 eligible questions, including the 3 ambiguous questions and excluding the 3 unanswerable questions with no gold evidence. Complete evidence means coverage under those anchors, not a guarantee of usable context or correct answers. The code accepts other values of K, but the reported studies used K=5. Null or ineligible values are not converted to zero.

## Configuration identity

Experiment configuration uses immutable Pydantic contracts. Canonical UTF-8 JSON with stable key ordering is SHA-256 hashed, and the experiment ID uses the first 12 digest characters. Runtime metadata is deliberately excluded from configuration identity.

Prompt identity is resolved through the same internal allowlisted registry used by the backend. Its hash covers the selected system prompt and stable user-message framing. The retained production prompt is `ddrag-production-v1`; prompt selection is not exposed through the public API.

The retained official configuration resolves to experiment ID `exp_eb14097ede63` and configuration hash `eb14097ede63989684c2c4ff06268985318ab580784658c0c2188b6acc979e8d` when the matching private benchmark and corpus artifacts are present.

## Runner

The live runner uses `/retrieval` for an independent retrieval observation and `/generation` for the full grounded-generation observation. It is guarded by an explicit `--execute` flag:

```powershell
python -m evaluation.runner --help
```

Evaluation credentials are supplied through local environment variables and are never serialized. A required warm-up question is supplied separately and is never written as an official benchmark result. Generated runs are stored under ignored `evaluation/runs/<run_id>/` directories with immutable configuration, manifest, and append-only raw-result artifacts.

Resume checks the configuration, experiment ID, benchmark hash, and existing result identities. Completed results, including failures, are not silently retried.

## Automated metrics

`evaluation.metrics` calculates deterministic metrics offline from benchmark records and immutable observations; it does not call DDRAG services. Retrieval and generation outputs, derived metrics, and qualitative annotations remain separate artifacts.

ExpectedPointCoverage is an automated answer-content proxy using an explicitly supplied point-specific matcher registry for automatable expected points. The public module contains reusable normalization, token/clause matching, coverage, and aggregation algorithms. Callers supply a mapping from question/point IDs to `PointMatcher` instances, validate it with `validate_matcher_registry(records, matchers)`, and pass it to `calculate_question_metrics(..., matchers=matchers)`. Missing or drifted matcher definitions fail clearly; there is no implicit benchmark-specific registry or private import.

The official matcher includes supplemental clause associations and permitted alternatives not fully represented by benchmark expected-point text. Those rules and their official answer fixtures remain private, while public tests use fictional examples. Exact official metric replay requires that matching supplemental registry as well as the authorized benchmark and persisted observations.

The reported ExpectedPointCoverage mean is the macro average of per-question coverage ratios across 11 eligible questions. The separate matched/eligible point totals cover 18 points; their pooled ratio is not the macro mean. For example, the baseline's 10 matched points out of 18 accompany a macro mean of 0.5455. Conservative lexical matching may diverge from qualitative answer correctness. The automated refusal signal is similarly narrow and does not establish correctness, grounding, or freedom from later unsupported claims; eligibility follows each record's answerability and expected behavior.

## Qualitative review

The Baseline Evaluation used **blinded AI-assisted qualitative evaluation with independent secondary adjudication and human final review/sign-off**.

The controlled paired comparisons used **blinded dual-reviewer qualitative evaluation with independent disagreement adjudication and post-freeze unblinding**. The independent review and adjudication were AI-assisted/AI-based. These judgments complement deterministic measurements; blinding and adjudication do not make them objective ground truth or substitute for a conventional human expert-panel study.

`evaluation.human_review` provides offline batch preparation, validation, and unblinding mechanics. Private packets, mappings, reviewer responses, prompts/instructions, adjudication material, and final reviewer-level provenance are not published. Consequently, the public framework is testable, but the exact qualitative workflow cannot be replayed from public files alone.

## Controlled comparisons

The Controlled Generation-Model Comparison compared a fresh `qwen2.5:1.5b` control with `qwen2.5:3b` while holding corpus, benchmark, chunking, retrieval, and prompt configuration fixed. The candidate was not adopted for this tested setup. The historical baseline used a different database/index identity, so each comparison used its own fresh control.

The Controlled Prompt Intervention compared `ddrag-production-v1` with `ddrag-answer-completeness-v1` while holding the generation model and retrieval pipeline fixed. The candidate prompt did not improve the predeclared complete-retrieval generation-failure population, so the production prompt was retained. Both decisions are limited to the tested setup.

## Failure analysis

The failure taxonomy separates complete-retrieval generation failures, incomplete-retrieval answer failures, ambiguous/unanswerable edge cases, operational failures, and successes/near-successes. This avoids treating every weak answer as a retrieval failure and preserves grounding/refusal tags as separate dimensions.

## Reproducing the public evaluation framework

Install the development dependencies, then run the public framework tests from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest evaluation/tests -q -p no:cacheprovider
```

Generic metric and refusal tests use synthetic fixtures and do not require private files. Integration tests that require the private official benchmark or corpus skip clearly when those artifacts are absent. Explicit strict official validation still fails when its required artifacts are unavailable. The framework never substitutes synthetic records for official benchmark inputs.

## Reproducing the exact official experiment

Exact official-result reproduction requires all of the following local inputs and services:

- the authorized corpus PDF matching the published filename, size, page count, and SHA-256;
- the authorized official benchmark matching the published benchmark SHA-256;
- the private supplemental matcher rules for exact official answer-content metric replay;
- the documented Ollama models and retained configuration;
- DDRAG backend/database services and a dedicated evaluation account;
- the warm-up policy and runner invocation used by the experiment.

After placing the authorized private artifacts at their documented local paths, validate them before execution:

```powershell
.venv\Scripts\python.exe -m evaluation.validation --strict-csu-v1
```

This command fails if the required local artifacts are absent or do not match. Matching artifacts permit a repeat of the protocol, not a guarantee of identical generated answers: the original runs used runtime-default sampling. Reconstructing the published qualitative judgments also requires the private review inputs and workflow.

## Results

The [evaluation article](reports/ddrag_evaluation_article.md) is the reader-facing synthesis. The [frozen evidence report](reports/ddrag_evaluation_report.md) remains the authoritative record of the numerical findings. Its structured summary and traceability matrix preserve machine-readable provenance.

Technical appendices provide the [Controlled Generation-Model Comparison](experiments/e10_final_comparison.md), [Controlled Prompt Intervention](experiments/e11_prompt_final_comparison.md), and [tested prompt text](prompts/ddrag-answer-completeness-v1.txt). The two comparison reports retain their original titles, identifiers, and bytes for hash verification; their categorical tables add detail beyond the article. Read their qualitative labels using the AI-assisted methodology disclosed above. Execution logs and the intermediate failure-analysis planning record remain private; their aggregate findings are synthesized in the article and frozen report.

Historical baseline retrieval results were Hit@5 `0.973`, Recall@5 `0.851`, and CompleteEvidence@5 `0.730` over 37 retrieval-eligible questions. These are not overall accuracy scores.

## Limitations

- one corpus and one 40-question benchmark limit generalization;
- runtime-default sampling permits variation;
- local hardware bounds latency observations;
- complete evidence retrieval is weaker than Hit@5;
- generation failures remain possible despite complete retrieval;
- page metadata and a retrieval relevance threshold are absent;
- retrieval/chunking/embedding settings were not established as optimal;
- exact qualitative replay requires private reviewer artifacts and workflow context.

## Data and artifact availability

The public documentation includes methodology, corpus and benchmark identities and hashes, selected aggregate comparison evidence, the tested prompt intervention, and final reports. Generic framework code and synthetic metric tests support inspection of the evaluation procedure. The source PDF, handbook-derived official benchmark, and official matcher rules/answer fixtures are intentionally not redistributed because the repository does not establish redistribution permission. Raw runs, private review packets and mappings, reviewer response/adjudication files, and operational audit records remain local. Repeating the official protocol requires authorized matching corpus and benchmark inputs; exact metric replay also requires the private matcher registry, and exact qualitative replay requires the private AI-assisted review workflow. The [public manifest](reports/ddrag_public_evaluation_manifest.json) records the publication boundary.
