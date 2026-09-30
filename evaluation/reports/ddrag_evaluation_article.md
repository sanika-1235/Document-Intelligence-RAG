# Evaluating DDRAG: Evidence Retrieval, Answer Quality, and Controlled Interventions

## Abstract

DDRAG is a local document-question-answering application that combines vector retrieval with language-model generation. Its evaluation asks whether retrieving relevant evidence is sufficient to produce complete, supported answers, and whether a larger generator or a targeted prompt improves the tested system. On a fixed 40-question Caraga State University (CSU) Student Handbook benchmark, baseline retrieval achieved Hit@5 of 0.973, Recall@5 of 0.851, and CompleteEvidence@5 of 0.730 across 37 retrieval-eligible questions. Failure analysis identified answer failures even when all benchmark-required evidence was retrieved. Two controlled paired comparisons did not support adopting the tested larger generator or answer-completeness prompt. These descriptive findings support retaining the tested configuration while motivating further study of retrieval coverage and generation completeness.

This article reorganizes the [frozen evaluation report](ddrag_evaluation_report.md) for readers. The report, [structured summary](ddrag_evaluation_summary.json), and [claim traceability matrix](ddrag_evaluation_traceability.csv) remain the numerical evidence sources. Original identifiers in those appendices are provenance labels, not prerequisites for understanding the study.

## System and evaluation objective

DDRAG accepts PDF, DOCX, Markdown, and text documents. Its FastAPI backend extracts text, creates overlapping character chunks, stores local Ollama embeddings in PostgreSQL with pgvector, retrieves chunks belonging to the authenticated user, and generates answers with ordinal source references. The React interface displays answers and preserves chat history. The [architecture guide](../../docs/ARCHITECTURE.md) describes the implementation.

The evaluation separates retrieval coverage, answer correctness, grounding, refusal, operational reliability, and latency. It asks which failures remain under a fixed setup and whether a specific change improves them. No overall accuracy or composite selection score is defined.

| Component | Retained tested configuration |
| --- | --- |
| Generation model | `qwen2.5:1.5b` |
| Prompt | `ddrag-production-v1` |
| Embedding model | `qwen3-embedding:0.6b` |
| Retrieval | owner-scoped pgvector cosine distance, top-K 5 |
| Chunking | 1000 characters, 200-character overlap |
| Sampling | runtime defaults; no explicit temperature, top-p, or seed |

Only the generation model and prompt received completed controlled comparisons. Retaining the other settings expresses continuity, not evidence that they outperform alternatives.

## Corpus and benchmark

The study uses one 93-page CSU Student Handbook and 40 questions: 34 answerable, 3 ambiguous, and 3 unanswerable. Categories cover direct factual, procedural, multi-condition, multi-chunk, policy-eligibility, summarization, ambiguous, and unanswerable questions. The [benchmark metadata](../benchmarks/README.md) describes its schema, composition, identity, and hash; the [corpus metadata](../corpus/README.md) identifies the source document.

Evidence is specified through stable page, section, and text anchors rather than database chunk IDs. Retrieval metrics include the 34 answerable and 3 ambiguous questions; the 3 unanswerable questions have no required evidence and are excluded. The PDF and official handbook-derived benchmark are intentionally not redistributed.

## Methods and interpretation

The runner observes `/retrieval` independently and `/generation` for the full retrieval-plus-generation path. Each controlled comparison used a fresh control and candidate with a shared corpus/index, benchmark, and retrieval configuration; ordered retrieval identities matched on all 40 questions within each pair. The generator comparison changed only the generation model, and the prompt intervention changed only prompt identity and content. Each arm received one unscored warm-up.

The historical baseline's database/index identity was not preserved after a local environment reset. Consequently, its results provide context; fresh controls are the reference for the paired comparisons. One run per arm and runtime-default sampling limit causal attribution and precision. The comparisons are descriptive, with no confidence intervals or significance analysis.

The Baseline Evaluation used **blinded AI-assisted qualitative evaluation with independent secondary adjudication and human final review/sign-off**. The controlled comparisons used **blinded dual-reviewer qualitative evaluation with independent disagreement adjudication and post-freeze unblinding**. Independent review and adjudication were AI-assisted/AI-based. These labels complement deterministic metrics; they are not automatically objective judgments or a conventional human expert-panel study.

## Metrics and denominators

| Measure | Interpretation |
| --- | --- |
| Hit@5 | Whether any required evidence unit appears in the first five chunks; mean across 37 eligible questions |
| Recall@5 | Fraction of required evidence units retrieved per question; macro mean across those 37 questions |
| CompleteEvidence@5 | Whether all required evidence units are retrieved; mean across those 37 questions |
| Answer correctness | Qualitative `correct`, `mostly_correct`, `partial`, or `incorrect` judgment; unavailable outputs remain `not_applicable` |
| Grounding | Qualitative support of answer claims by retrieved context, assessed separately from correctness |
| Refusal | Appropriate refusal, failed refusal, or over-refusal; the latter means declining an answerable question |
| ExpectedPointCoverage | An automated answer-content proxy: the macro mean of per-question lexical coverage ratios over eligible questions |
| Operational reliability | Successful versus failed requests, independently of answer quality |

ExpectedPointCoverage covers 18 eligible points across 11 questions. Point totals and macro means use different denominators: the baseline matched 10 of 18 points and had a macro mean of 0.5455; 0.5455 is not the pooled fraction 10/18. The frozen tables place these quantities together, so they should be read as separate observations. This lexical proxy does not establish semantic correctness.

Complete evidence means coverage under the benchmark's anchor-matching rules, not proof that context is usable or sufficient in every semantic sense. Citation validation checks source identifiers, not whether every claim is supported. Non-exclusive failure tags can overlap and should not be added into an overall error rate.

Generation-endpoint latency includes retrieval within the generation path and is not pure model-compute time. The runner's combined/end-to-end observation also includes its separate retrieval observation; it does not measure browser-to-answer latency. Latencies reflect the tested local Windows laptop and Ollama runtime, not a hardware-independent throughput claim.

## Baseline Evaluation

All 40 baseline questions completed operationally. Retrieval found at least one required evidence unit much more often than it found every required unit:

| Retrieval metric | Baseline result |
| --- | ---: |
| Hit@5 | 0.973 |
| Recall@5 | 0.851 |
| CompleteEvidence@5 | 0.730 |

Qualitative correctness was 15 correct, 7 mostly correct, 14 partial, and 4 incorrect. Grounding was 26 grounded, 11 partially grounded, 2 unsupported, and 1 not applicable. A high retrieval hit rate therefore did not imply uniformly complete or supported answers.

## Controlled Generation-Model Comparison

The paired study compared a fresh `qwen2.5:1.5b` control with `qwen2.5:3b`, holding retrieval, corpus, benchmark, chunking, prompt, and sampling policy constant. Each arm completed 39 of 40 questions; failures affected different questions, leaving 38 comparable pairs.

The candidate had lower correctness on 29 pairs, the same label on 9, and a higher label on none. It received 34 over-refusal labels versus 1 for the control. Mean generation-endpoint latency was 14178.546 ms for the control and 19918.148 ms for the candidate, measured over 39 successful generation observations per arm.

Under the fixed benchmark and tested runtime, these observations did not support replacing the retained generator. They do not establish a general ranking of model sizes. The [generation-model comparison appendix](../experiments/e10_final_comparison.md) preserves detailed category results, distributions, and hashes.

## Failure Analysis

Analysis of the fresh generation-model control separated 12 complete-retrieval answer failures, 7 incomplete-retrieval answer failures, 6 ambiguous/unanswerable edge cases, 1 operational failure, and 14 successes or near-successes. Among 26 answerable questions with complete retrieval, 12 answers were partial or incorrect: 46.15% descriptively.

The seven incomplete-retrieval failures comprised three multi-chunk, three summarization, and one procedural case. These associations distinguish retrieval coverage from generation completeness; they do not prove that the prompt caused failures. The complete-retrieval failures motivated a targeted prompt hypothesis with a predeclared 12-question diagnostic subset.

## Controlled Prompt Intervention

Both arms used `qwen2.5:1.5b`. The candidate, [ddrag-answer-completeness-v1](../prompts/ddrag-answer-completeness-v1.txt), emphasized covering all supported requested parts and avoiding premature refusal. The production prompt was the control. Both arms completed all 40 questions.

| Primary diagnostic observation | Result |
| --- | --- |
| Candidate correctness transitions on the 12-question target subset | 2 higher, 7 unchanged, 3 lower |
| Partial or incorrect answers among 26 answerable complete-retrieval questions | 12/26 in each arm |
| Complete-retrieval `partial_answer` tags | 12 control; 13 candidate |

The candidate corrected one failed refusal on an unanswerable question while the other two unanswerable questions remained appropriate. This improvement accompanied mixed changes elsewhere: overall correctness transitions were 7 higher, 24 unchanged, and 9 lower; grounding transitions were 8 higher, 22 unchanged, and 10 lower. Failure tags remained non-exclusive.

ExpectedPointCoverage macro means were 0.681818 for control and 0.515152 for candidate, with separate point totals of 13/18 and 11/18. Proxy and qualitative-label changes sometimes diverged. Mean generation-endpoint latency increased from 14551.198 ms to 15091.539 ms, or 3.713%. Prompt length increased from 2022 to 3320 characters as part of the intervention and was not independently isolated.

The candidate did not improve the predeclared completeness-failure population, so `ddrag-production-v1` was retained. The [prompt comparison appendix](../experiments/e11_prompt_final_comparison.md) provides the categorical transition matrices and detailed findings.

## Engineering decisions, limitations, and future work

The studies support retaining the tested generator and prompt for this setup. They also show that incomplete evidence coverage and incomplete generation remain distinct limitations. A refusal improvement alone did not justify the prompt change, and fewer hallucination tags from a frequently refusing generator did not establish better answer quality.

The study is limited to one corpus, a small benchmark, one run per comparison arm, runtime-default sampling, local hardware, and AI-assisted qualitative judgments. Several categories contain only three questions, so category-level findings are especially sensitive to individual cases. The predeclared prompt subset was selected from earlier failures on the same benchmark; it was not an independent held-out evaluation set. Private benchmark and reviewer inputs restrict independent audit and exact replay. Model tags and runtime versions alone do not guarantee identical artifacts or outputs.

Retrieval currently has no relevance threshold, and persisted retrieval records lack PDF page metadata. Future work could compare top-K 5 with one predeclared larger value, followed by controlled studies of chunking, re-ranking, hybrid retrieval, or alternative embeddings. Explicit sampling control and repeated trials would address different reproducibility questions. None of those changes has been shown to improve the reported system.

## Reproducibility and data availability

The [evaluation guide](../README.md) describes the framework, validation, runner, and current distribution limits. Matching authorized corpus and benchmark copies are necessary to repeat the official protocol; runtime-default sampling prevents a guarantee of identical answers. Reconstructing the published qualitative judgments additionally requires private reviewer inputs, mappings, adjudication, and workflow context.

Methodology, identity metadata, generic framework code and synthetic tests, selected aggregate comparison evidence, this article, and the frozen results are provided publicly. The source PDF, official handbook-derived benchmark and review artifact, benchmark-specific matcher rules and answer fixtures, raw runs, and private qualitative-review materials are not redistributed. The repository does not establish permission to redistribute the excluded handbook-derived content. Exact official metric replay requires the private supplemental matcher registry; public algorithms receive it explicitly rather than embedding official answers. The [publication manifest](ddrag_public_evaluation_manifest.json) records the artifact boundary.

## Conclusion

Under the fixed CSU Student Handbook benchmark and tested local runtime, DDRAG retrieved some required evidence reliably but retrieved complete evidence less consistently, and complete retrieval did not eliminate answer failures. The controlled generator and prompt changes did not resolve the observed weaknesses. The retained configuration is an evidence-scoped engineering decision, with retrieval coverage, generation completeness, repeated-trial evaluation, and independent review remaining substantive areas for further work.
