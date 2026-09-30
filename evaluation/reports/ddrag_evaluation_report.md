# DDRAG Evaluation Report

## Executive Summary

Under the frozen Caraga State University (CSU) Student Handbook benchmark and tested local DDRAG environment, retrieval usually found at least one required evidence unit, but it less consistently retrieved all required evidence. The historical E9 baseline recorded Hit@5 of 0.9730, Recall@5 of 0.8509, and CompleteEvidence@5 of 0.7297. Qualitative answer behavior remained a separate concern: E11A identified 12 answer failures despite complete retrieval, while 7 additional answer failures involved incomplete retrieval and were concentrated in multi-chunk and summarization questions.

The controlled studies did not support either tested production change. In the fresh paired E10 comparison, `qwen2.5:3b` systematically over-refused under the fixed prompt and context, with 34 over-refusals versus 1 for `qwen2.5:1.5b`; its correctness was lower on 29 comparable questions and higher on none. In E11, the `ddrag-answer-completeness-v1` prompt improved some ambiguity/refusal behavior, notably Q046, but it did not show a net correctness improvement on the predeclared 12-question complete-retrieval generation-failure subset: 2 transitions were higher, 7 unchanged, and 3 lower.

DDRAG therefore retains `qwen2.5:1.5b`, `ddrag-production-v1`, `qwen3-embedding:0.6b`, top-K 5, and 1000/200 character chunking. Only the generation model and generation prompt were experimentally compared in the completed controlled optimization studies. No controlled study established that the unchanged retrieval settings should be preferred over alternatives. No single DDRAG accuracy score or composite model/prompt score is reported.

## Evaluation Objective

This report asks: How reliably does DDRAG retrieve the evidence needed to answer questions and generate responses that remain supported by the fixed CSU Student Handbook corpus, and what did controlled evaluation reveal about its remaining failure modes and tested alternatives?

The answer is organized across separate dimensions: retrieval, generation/correctness, grounding, refusal, operational reliability, and latency.

## System Under Evaluation

The retained configuration is:

| Component | Retained value |
| --- | --- |
| Generation model | `qwen2.5:1.5b` |
| Generation prompt | `ddrag-production-v1` |
| Embedding model | `qwen3-embedding:0.6b` |
| Retrieval | pgvector cosine, owner-scoped, top-K 5 |
| Chunking | 1000 characters with 200-character overlap |
| Sampling | runtime defaults; no explicit temperature, top-p, or seed |

The corpus identity is `csu_student_handbook_2019`, SHA-256 `ce3c2818ea5ba171a2e75b51d3db14970f57d4d54515eccc0c0a275470be4841`. The benchmark identity is `csu_handbook` version 1.0.0, SHA-256 `f2d5e90b252f288572f617d56e27f1a003e18b1171421881c636793e32137ae0`.

## Corpus and Benchmark

The evaluation uses one fixed 93-page CSU Student Handbook corpus and a frozen 40-question benchmark. The benchmark includes direct factual, procedural, multi-condition, multi-chunk, policy-eligibility, summarization, ambiguous, and unanswerable questions. Retrieval metrics exclude the 3 unanswerable zero-evidence questions, leaving 37 eligible questions.

The source PDF is local-only. Its manifest preserves corpus identity without distributing the document. PDF ingestion does not preserve page metadata in retrieval results, so the evaluation traces evidence through document/chunk identity and text anchors rather than page citations.

## Evaluation Methodology

The authoritative evidence chain is benchmark, experiment configuration, immutable raw question results, frozen qualitative annotations, derived metrics, comparison artifacts, and this report. Persisted artifacts take precedence over conversational summaries. Retrieval and generation observations, qualitative labels, deterministic proxies, operations, and latency are kept distinct.

E9 used **blinded AI-assisted qualitative evaluation with independent secondary adjudication and human final review/sign-off**. It was not a purely manual human annotation process.

E10 and E11 used **blinded dual-reviewer qualitative evaluation with independent disagreement adjudication and post-freeze unblinding**. The independent review and adjudication were AI-assisted/AI-based; this was not a conventional human-panel study. Private reviewer prompts, hidden mappings, and review packets are not reproduced here.

Historical E9 remains valid frozen evidence. A laptop reset did not preserve its PostgreSQL/index identity, so E10 used a fresh post-recovery shared environment and a fresh `qwen2.5:1.5b` paired control. E11 likewise used a fresh paired control. E9 is historical context, not the live causal control for E10 or E11.

## Metrics and Qualitative Review

- Hit@5 asks whether at least one gold evidence unit appears in the first 5 retrieved chunks.
- Recall@5 measures the fraction of gold evidence units retrieved.
- CompleteEvidence@5 requires all gold evidence units to be present in the first 5 chunks.
- Correctness, grounding, and refusal are separate frozen qualitative dimensions.
- ExpectedPointCoverage is a conservative automated answer-content proxy. It is not semantic correctness, grounding, or overall accuracy.
- Operational reliability counts completed and failed requests independently of answer quality.
- Retrieval and generation latency are reported separately where available; generation-endpoint latency is not pure model-compute time.

## Hardware and Runtime

The controlled work ran locally on a recovered Windows 11 laptop with an AMD Ryzen 5 3550H CPU, approximately 17.94 GiB of RAM, integrated Radeon Vega 8 graphics, and a Radeon RX 560X. E10 recorded Ollama 0.34.4. These details bound the latency observations; they do not support performance claims for other hardware. Generation used runtime-default sampling, so repeated outputs are not guaranteed to be identical.

## E9 Historical Baseline

E9 completed 40/40 questions successfully. Retrieval and the automated proxy were:

| Measure | Frozen E9 result |
| --- | ---: |
| Hit@5 | 0.9730 |
| Recall@5 | 0.8509 |
| CompleteEvidence@5 | 0.7297 |
| ExpectedPointCoverage | 10/18 (0.5455) |
| Mean retrieval latency | 190.976 ms |
| Mean generation-path latency | 14952.937 ms |

Frozen E9 qualitative distributions were:

| Dimension | Distribution |
| --- | --- |
| Correctness | correct 15; mostly_correct 7; partial 14; incorrect 4 |
| Grounding | grounded 26; partially_grounded 11; unsupported 2; not_applicable 1 |
| Refusal | appropriate 2; failed_refusal 1; over_refusal 1; not_applicable 36 |

Hit@5 shows that at least some required evidence was usually present. The lower CompleteEvidence@5 shows a material gap between retrieving any evidence and retrieving all benchmark-required evidence. E9 also shows that high retrieval hit rate did not eliminate partial, incorrect, or unsupported answers.

## E10 Generation-Model Comparison

E10 compared a fresh `qwen2.5:1.5b` control with a `qwen2.5:3b` candidate in one shared post-recovery environment. The ordered retrieval results were identical for 40/40 questions. Operational success was 39/40 for each arm, with failures on different questions.

| Dimension | Control: 1.5B | Candidate: 3B |
| --- | --- | --- |
| Correctness | correct 12; mostly_correct 5; partial 18; incorrect 4; not_applicable 1 | correct 4; mostly_correct 0; partial 2; incorrect 33; not_applicable 1 |
| Grounding | grounded 27; partially_grounded 11; unsupported 1; not_applicable 1 | grounded 6; partially_grounded 5; unsupported 28; not_applicable 1 |
| Over-refusal | 1 | 34 |

Paired correctness transitions were candidate higher 0, same 9, candidate lower 29, and not comparable 2. Mean generation-path latency was 14178.546 ms for control and 19918.148 ms for candidate, a candidate increase of 5739.601 ms.

The unchanged retrieval context and widespread candidate over-refusal show that the larger model systematically avoided answering under the tested prompt/context. The scoped decision is to retain `qwen2.5:1.5b`. This is a local engineering decision, not a general model ranking.

## E11 Failure Analysis

E11A mechanically classified the fresh E10 control without changing frozen labels:

| Primary root-cause bucket | Count |
| --- | ---: |
| Complete retrieval / generation failure | 12 |
| Incomplete retrieval with answer failure | 7 |
| Ambiguous or unanswerable edge case | 6 |
| Operational failure | 1 |
| Success or near-success | 14 |

Among 26 answerable complete-retrieval questions, 12 were partial or incorrect, a descriptive complete-retrieval answer-failure rate of 46.15%. The `partial_answer` tag appeared 23 times: 13 with complete retrieval and 10 with incomplete retrieval. This made prompt behavior a direct, isolated next hypothesis, while not proving that the prompt caused each failure.

The 7 incomplete-retrieval answer failures remain a separate limitation. They were concentrated in multi-chunk and summarization questions, with 3 cases in each category and 1 procedural case.

## E11 Prompt Comparison

E11 compared `ddrag-production-v1` with `ddrag-answer-completeness-v1`; both arms used `qwen2.5:1.5b`, and only prompt ID/hash differed. Ordered retrieval identity matched for 40/40 questions, and both arms completed 40/40 successfully.

### Primary target subset

The predeclared 12-question target subset was the primary result:

| Correctness | Control | Candidate |
| --- | ---: | ---: |
| mostly_correct | 4 | 2 |
| partial | 4 | 7 |
| incorrect | 4 | 3 |

Candidate transitions were higher for 2 questions (Q020 and Q024), unchanged for 7, and lower for 3 (Q010, Q019, and Q027). The answer-completeness prompt did not show a net correctness improvement on the predeclared complete-retrieval generation-failure subset.

### Complete-retrieval and overall results

For the 26 answerable complete-retrieval questions, partial-or-incorrect outcomes were 12/26 for both control and candidate. The `partial_answer` tag was present in 12/26 control answers and 13/26 candidate answers.

| Dimension | Control | Candidate |
| --- | --- | --- |
| Overall correctness | correct 12; mostly_correct 7; partial 13; incorrect 8 | correct 8; mostly_correct 12; partial 16; incorrect 4 |
| Correctness transitions | candidate higher 7; same 24; candidate lower 9 | paired result |
| Grounding transitions | candidate higher 8; same 22; candidate lower 10 | paired result |

Candidate-minus-control failure-tag changes were hallucination +1, unsupported claim +2, partial answer +2, retrieval failure +1, over-refusal +1, failed refusal -2, and citation issue +4. This mixed pattern is not reduced to a composite score.

The candidate nevertheless produced a useful refusal improvement. Q046 changed from incorrect/unsupported/failed refusal to correct/grounded/appropriate; Q047 and Q048 remained appropriate. The candidate therefore improved some refusal and ambiguity behavior even though it did not solve the primary completeness target.

ExpectedPointCoverage was 13/18 (0.681818) for control and 11/18 (0.515152) for candidate. Some proxy changes aligned with semantic-label changes and others did not, so this lexical proxy is not used as the sole reason for rejecting the prompt.

Mean generation-endpoint latency was 14551.198 ms for control and 15091.539 ms for candidate, a 540.340 ms or 3.713% increase. Mean combined/end-to-end latency was 14727.746 ms and 15259.941 ms, respectively, a 532.195 ms or 3.614% increase. Prompt length increased from 2022 to 3320 characters, or 64.193867%; prompt length was part of the intervention and was not independently isolated.

The scoped decision is to retain `ddrag-production-v1`. The candidate did not improve the predeclared primary failure population and introduced additional grounding, failure-mode, and latency trade-offs.

## Cross-Experiment Findings

1. E9 Hit@5 of approximately 0.973 shows that DDRAG usually retrieved at least some required evidence on this benchmark.
2. E9 CompleteEvidence@5 of approximately 0.730 shows materially weaker complete evidence coverage.
3. Retrieval was not the only bottleneck: E11A found 12 answer failures despite complete retrieval.
4. Multi-chunk and summarization questions remain retrieval-coverage weaknesses.
5. Increasing the generator from 1.5B to 3B did not improve this tested setup and produced widespread over-refusal under the fixed prompt/context.
6. The answer-completeness prompt improved some refusal/ambiguity behavior but did not improve the predeclared completeness-failure population.
7. Operational reliability was generally strong: E9 and both E11 arms completed all 40 questions, while both E10 arms completed 39.
8. Local generation latency remained substantial, with arm means ranging from roughly 14.2 to 19.9 seconds across the controlled comparisons.

## Remaining Failure Modes

Generation completeness remains material even when retrieval is complete: E11A found 12 such failures, and neither E11 prompt reduced the 12/26 partial-or-incorrect count among answerable complete-retrieval questions. Grounding and refusal also remain distinct concerns; improvements in one dimension sometimes coincided with regressions in another.

Retrieval coverage remains incomplete for complex evidence needs. The 7 E11A incomplete-retrieval answer failures, particularly those in multi-chunk and summarization categories, justify future controlled retrieval study. The evidence does not establish that a larger top-K would necessarily solve them.

There is no configured relevance threshold, top-K remains fixed at 5, and page metadata is unavailable in retrieval output. These constraints limit abstention/ranking and page-level evidence traceability.

## Engineering Decisions

- **Generation model:** retain `qwen2.5:1.5b`; fresh paired E10 evidence did not support replacement by `qwen2.5:3b`.
- **Generation prompt:** retain `ddrag-production-v1`; the E11 candidate did not improve the predeclared primary population and introduced trade-offs.
- **Retrieval:** retain top-K 5, 1000/200 chunking, `qwen3-embedding:0.6b`, and pgvector cosine retrieval because no retrieval-setting optimization experiment was executed. This is continuity, not comparative evidence favoring these settings over alternatives.

## Limitations

- One fixed corpus and one 40-question benchmark limit generalization.
- One full run per configuration was executed; runtime-default sampling permits run-to-run variation.
- Results reflect local hardware and runtime.
- No statistical-significance claims or confidence intervals are provided.
- Qualitative review was AI-assisted/AI-based, with the disclosed blinded review and adjudication procedures rather than a conventional human panel.
- ExpectedPointCoverage is lexical and can diverge from semantic correctness.
- PDF ingestion does not retain page metadata in retrieval.
- Retrieval has no relevance threshold and uses fixed top-K 5.
- Retrieval-setting, chunking, re-ranking, hybrid retrieval, and embedding optimization were not exhaustively explored.
- Historical E9 remains valid, but the post-laptop-reset database/index identity changed; fresh post-recovery controls, not E9, are the causal references for E10 and E11.
- All conclusions are descriptive and scoped to the frozen benchmark, corpus, application, configurations, and local environment.

## Future Work

Retrieval-coverage optimization remains a justified future experiment, particularly for multi-chunk and summarization questions, but was not pursued in M13 to avoid turning the evaluation into broad hyperparameter search. A first controlled study could compare top-K 5 with one predeclared larger K while holding other factors fixed. That experiment was not executed, and no improvement is assumed.

Explicit sampling/seed control may be studied independently for reproducibility. Chunking, re-ranking, hybrid retrieval, and alternative retrieval representations or embeddings may also be studied only through future controlled experiments.

## Reproducibility

The evaluation package supports offline strict benchmark/corpus validation and preserves configurations, raw observations, frozen annotations, deterministic summaries, comparisons, and hashes as separate artifacts. Raw run directories are intentionally Git-ignored, and the source PDF is local-only. Reproduction requires the exact corpus identified by the manifest, the frozen benchmark, the recorded configuration, a dedicated evaluation account/environment, and equivalent local services.

The report does not expose private reviewer prompts, mappings, credentials, tokens, or environment data.

## Artifact Traceability

The machine-readable summary is `evaluation/reports/ddrag_evaluation_summary.json`. Significant claims and metrics are mapped to frozen source artifacts in `evaluation/reports/ddrag_evaluation_traceability.csv`. The packaging/classification draft is `evaluation/reports/e12_artifact_classification.md`.

The frozen evidence integrity gate re-hashed E9 raw and baseline report artifacts, E10 control/candidate raw results and final comparison, E11 failure analysis, and E11 control/candidate raw results, deterministic summary, and final comparison. All matched their expected SHA-256 digests before this report was created.
