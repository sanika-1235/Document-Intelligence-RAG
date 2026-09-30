# DDRAG E10 Paired Generation-Model Comparison

## Executive Summary

On the frozen paired local experiment, the preferred generation configuration for the tested DDRAG setup is `qwen2.5:1.5b`. This scoped engineering preference is supported independently by the frozen qualitative labels, deterministic answer-content proxy, refusal behavior, and latency. Operational reliability was tied. It is not a universal model ranking.

The candidate had lower correctness on 29 comparable questions, the same label on 9, and no higher-correctness transitions; 2 pairs were not comparable because one response was unavailable. Grounding was lower on 30 comparable questions, the same on 7, and higher on 1; 2 were not comparable. The candidate received 34 `over_refusal` labels versus 1 for control. Mean generation latency increased by 40.481%.

## Why a Fresh Paired Control Was Required

Post-reset recovery did not preserve the PostgreSQL/index identity that produced historical E9. E10 therefore used a fresh shared index and fresh control run. Historical E9 is context only; the causal paired comparison is the fresh control versus fresh candidate.

## Experimental Controls

- Shared index content SHA-256: `25287b54a8c147ae48994e2a67af10c2f0d0b6bf48cfbb22e740f14d55c8c409`
- Same benchmark, question-ID set, corpus/index, retrieval strategy, top-K, prompt, sampling mode, timeouts, and evaluation method: PASS
- Identical ordered retrieval observations: 40/40
- Sole intentional canonical change: `generation.generation_model`
- Control: `qwen2.5:1.5b`; candidate: `qwen2.5:3b`

## Deterministic Results

| Metric | Control | Candidate | Candidate minus control |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.972973 (n=37) | 0.972973 (n=37) | 0.000000 |
| Recall@5 | 0.850901 (n=37) | 0.850901 (n=37) | 0.000000 |
| CompleteEvidence@5 | 0.729730 (n=37) | 0.729730 (n=37) | 0.000000 |
| ExpectedPointCoverage | 0.742424 (13/18; n=11) | 0.000000 (0/18; n=11) | -0.742424 |
| Automated refusal signal | 0/3 | 0/3 | 0 |
| Operational success | 39/40 | 39/40 | 0 |
| Operational failure | 1/40 | 1/40 | 0 |
| Retrieval mismatch | 0/39 observed | 0/39 observed | 0 |

Retrieval metrics serve as a control check and were identical. ExpectedPointCoverage remains a lexical answer-content proxy, not semantic correctness.

## Blinded Qualitative Method

The frozen method was **blinded dual-reviewer qualitative evaluation with independent disagreement adjudication and post-freeze unblinding**. Two independent reviewers assessed 40 paired items / 80 anonymous responses. Inter-reviewer disagreements were isolated mechanically; 25 items contained at least one disputed field. Of 520 final label fields, 456 were resolved by reviewer consensus and 64 by independent blinded adjudication.

## Qualitative Results

All distributions use n=40 responses per configuration. Failure tags are counted independently and may sum beyond 40.

| Dimension | Control | Candidate |
| --- | --- | --- |
| Correctness | correct=12, mostly_correct=5, partial=18, incorrect=4, not_applicable=1 | correct=4, partial=2, incorrect=33, not_applicable=1 |
| Grounding | grounded=27, partially_grounded=11, unsupported=1, not_applicable=1 | grounded=6, partially_grounded=5, unsupported=28, not_applicable=1 |
| Refusal | appropriate=3, over_refusal=1, not_applicable=36 | appropriate=3, over_refusal=34, not_applicable=3 |
| Failure tags | hallucination=9, unsupported_claim=3, partial_answer=23, retrieval_failure=5, over_refusal=1, ambiguity_issue=3, request_error=1 | partial_answer=2, retrieval_failure=6, over_refusal=34, ambiguity_issue=3, request_error=1 |

## Paired Question-Level Changes

Correctness transitions: candidate higher=0, same=9, lower=29, not comparable=2.

### Correctness transition matrix

| Control \ Candidate | correct | mostly_correct | partial | incorrect | not_applicable |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 4 | 0 | 0 | 8 | 0 |
| mostly_correct | 0 | 0 | 1 | 4 | 0 |
| partial | 0 | 0 | 1 | 16 | 1 |
| incorrect | 0 | 0 | 0 | 4 | 0 |
| not_applicable | 0 | 0 | 0 | 1 | 0 |

Grounding transitions: candidate higher=1, same=7, lower=30, not comparable=2.

### Grounding transition matrix

| Control \ Candidate | grounded | partially_grounded | unsupported | not_applicable |
| --- | ---: | ---: | ---: | ---: |
| grounded | 5 | 4 | 18 | 0 |
| partially_grounded | 1 | 1 | 8 | 1 |
| unsupported | 0 | 0 | 1 | 0 |
| not_applicable | 0 | 0 | 1 | 0 |

### Refusal transitions

No universal ordinal ordering was applied.

| Control -> Candidate | Count |
| --- | ---: |
| appropriate -> appropriate | 3 |
| not_applicable -> not_applicable | 3 |
| not_applicable -> over_refusal | 33 |
| over_refusal -> over_refusal | 1 |

For the frozen ambiguous cases: Q042 and Q043 changed `not_applicable -> over_refusal`; Q045 remained `over_refusal -> over_refusal`. For the frozen unanswerable cases Q046, Q047, and Q048, both configurations remained `appropriate -> appropriate`.

## Failure Modes

The candidate's dominant frozen failure tag was `over_refusal` (34 versus 1 control). Control had more `partial_answer` (23 versus 2), `hallucination` (9 versus 0), and `unsupported_claim` (3 versus 0). These counts describe different observed failure modes; the absence of hallucination tags in a configuration dominated by refusal does not imply higher overall answer quality.

Operational failures are separate: control Q032 and candidate Q025. Each unavailable response was frozen as correctness/grounding/refusal `not_applicable` with `request_error`; neither was converted to a semantic zero.

## Category Analysis

| Category | n | Control correctness | Candidate correctness | Control grounding | Candidate grounding | Control tags | Candidate tags |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| direct_factual | 8 | correct=4, mostly_correct=2, partial=1, incorrect=1 | incorrect=8 | grounded=6, partially_grounded=2 | unsupported=8 | hallucination=2, partial_answer=2 | over_refusal=8 |
| procedural | 6 | mostly_correct=3, partial=2, incorrect=1 | partial=1, incorrect=5 | grounded=3, partially_grounded=3 | grounded=1, unsupported=5 | hallucination=1, unsupported_claim=2, partial_answer=5, retrieval_failure=1 | partial_answer=1, retrieval_failure=1, over_refusal=5 |
| multi_condition | 6 | correct=2, partial=3, incorrect=1 | correct=1, incorrect=5 | grounded=3, partially_grounded=2, unsupported=1 | grounded=1, unsupported=5 | hallucination=2, unsupported_claim=1, partial_answer=4 | over_refusal=5 |
| multi_chunk | 6 | partial=6 | incorrect=5, not_applicable=1 | grounded=4, partially_grounded=2 | partially_grounded=2, unsupported=3, not_applicable=1 | hallucination=2, partial_answer=6, retrieval_failure=2 | retrieval_failure=2, over_refusal=5, request_error=1 |
| policy_eligibility | 5 | correct=3, partial=1, not_applicable=1 | incorrect=5 | grounded=3, partially_grounded=1, not_applicable=1 | unsupported=5 | hallucination=1, request_error=1 | over_refusal=5 |
| summarization | 3 | partial=3 | incorrect=3 | grounded=3 | partially_grounded=2, unsupported=1 | partial_answer=3, retrieval_failure=2 | retrieval_failure=2, over_refusal=3 |
| ambiguous | 3 | partial=2, incorrect=1 | partial=1, incorrect=2 | grounded=2, partially_grounded=1 | grounded=1, partially_grounded=1, unsupported=1 | hallucination=1, partial_answer=3, over_refusal=1, ambiguity_issue=3 | partial_answer=1, retrieval_failure=1, over_refusal=3, ambiguity_issue=3 |
| unanswerable | 3 | correct=3 | correct=3 | grounded=3 | grounded=3 | none | none |

No category winner is defined. Candidate `incorrect`/`unsupported`/`over_refusal` concentrations appeared across direct factual, procedural, multi-condition, multi-chunk, policy-eligibility, and summarization categories. Both configurations were `correct`, `grounded`, and `appropriate` on all three unanswerable questions.

## ExpectedPointCoverage Diagnostic

| Question | Control points | Control correctness | Candidate points | Candidate correctness |
| --- | ---: | --- | ---: | --- |
| Q002 | 3/3 | correct | 0/3 | incorrect |
| Q003 | 2/2 | correct | 0/2 | incorrect |
| Q004 | 0/1 | mostly_correct | 0/1 | incorrect |
| Q005 | 1/1 | partial | 0/1 | incorrect |
| Q007 | 1/1 | correct | 0/1 | incorrect |
| Q009 | 2/3 | correct | 0/3 | incorrect |
| Q027 | 1/1 | partial | 0/1 | incorrect |
| Q028 | 1/1 | partial | 0/1 | incorrect |
| Q030 | 0/2 | partial | 0/2 | incorrect |
| Q034 | 1/2 | correct | 0/2 | incorrect |
| Q035 | 1/1 | partial | 0/1 | incorrect |

All 11 candidate zero-match cases were frozen as `incorrect`. Within this eligible subset, zero lexical matches therefore align with frozen qualitative failure rather than acceptable paraphrase. This does not prove why each answer failed or make the lexical metric equivalent to correctness. Control Q004 and Q030 also had zero lexical matches but were labeled `mostly_correct` and `partial`, demonstrating that a zero matcher result can coexist with non-incorrect qualitative outcomes.

## Latency and Operational Trade-offs

| Latency | Control | Candidate | Difference |
| --- | --- | --- | ---: |
| Retrieval | n=40; mean 130.034 ms; median 120.960 ms; p95 208.240 ms | n=40; mean 159.662 ms; median 138.070 ms; p95 210.020 ms | +29.628 ms mean |
| Generation | n=39; mean 14178.546 ms; median 11854.850 ms; p95 25421.003 ms | n=39; mean 19918.148 ms; median 18906.447 ms; p95 23100.631 ms | +5739.601 ms mean |

Candidate mean generation latency was 40.481% higher. Operational reliability was tied at 39/40, with failures on different questions. Latency is one independent trade-off and is not used alone for selection.

## Relationship to Historical E9

Historical E9 remains separate context: Hit@5=0.9730, Recall@5=0.8509, CompleteEvidence@5=0.7297, ExpectedPointCoverage=10/18 (0.5455); correctness was 15 correct, 7 mostly correct, 14 partial, and 4 incorrect; grounding was 26 grounded, 11 partially grounded, 2 unsupported, and 1 not applicable.

The fresh control differed from historical E9 despite the same generation configuration. Runtime-default sampling and the changed post-recovery run environment reinforce why fresh E10 control?not historical E9?is the paired causal reference.

## Engineering Interpretation

**Observed facts.** Retrieval identities and aggregate retrieval controls matched. Candidate correctness was lower on 29 comparable questions and never higher; grounding was lower on 30 and higher on 1. Candidate over-refusal rose to 34 responses. Operational success tied, while candidate mean generation latency was 5739.601 ms higher.

**Interpretation.** In this paired run, the larger generator frequently avoided answering even though the shared retrieval context was unchanged. The separate qualitative and lexical results both disfavor that behavior for this setup. The control's main weaknesses were incompleteness and some hallucinated/unsupported content, so retaining it does not imply those issues are solved.

**Limitations.** There was one run per configuration, runtime-default sampling, different question-level operational failures, no significance analysis, and one local corpus/hardware/runtime. Qualitative results are dual-reviewer/adjudicated labels rather than project-owner manual annotation of every answer. ExpectedPointCoverage is conservative lexical matching.

## Engineering Decision

The **preferred generation configuration for the tested DDRAG setup** is `qwen2.5:1.5b`. It had materially stronger frozen correctness and grounding distributions, far less over-refusal, nonzero deterministic expected-point coverage, equal operational reliability, and lower mean generation latency. This is a scoped engineering choice for the tested CSU handbook setup, not a claim that it is universally better.

## Reproducibility / Artifact Hashes

Frozen inputs:

- Final annotations: `e1daa07a5000f0ae48feace27a2173a11675b13098576bcb52b1aeb41c21a3e1`
- Field-source provenance: `7a147cdd5166bcba13f231d80b2b56f44aa7b0f69c7a58a74666892ce318f586`
- Merge manifest: `44d0ee457e878ba88649468ccb3373932069ef29524fa5d491793c3fb9f423f4`
- Hidden mapping: `4cd74098d41ff67a10337e791f958e097e69a83b48bd91e3256cba38bf08bf94`
- Deterministic summary: `fe6db34103a89c8126854084e6e5c9418a2ebc4f57a67d0b191315f0cf20a561`
- Control raw results: `ab3a4fa45c30c3ed147521b6684b0137e687688e0da42994273a35094d39d3d9`
- Candidate raw results: `b2c7f6c544d048dc37c86e721c8e5a5eefd763df3012136e48624138e4cfecce`
