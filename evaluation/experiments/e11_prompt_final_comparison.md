# DDRAG E11 Paired Prompt Comparison

## Executive Summary

The predeclared answer-completeness prompt did not improve the primary 12-question target population under the frozen qualitative labels: candidate correctness was higher on 2 questions, unchanged on 7, and lower on 3. Both prompts had 12/26 answerable complete-retrieval questions labeled partial or incorrect, while complete-retrieval `partial_answer` tags changed from 12 to 13.

For this tested DDRAG setup, retain `ddrag-production-v1`. The candidate improved the frozen unanswerable/refusal cases, but those gains were accompanied by no primary-target net improvement, more overall downward than upward correctness and grounding transitions, higher hallucination/unsupported/over-refusal tag counts, and approximately 3.7% higher generation-endpoint latency. No production default was changed.

## Motivation from E11A

E11A predeclared 12 complete-retrieval generation-failure questions: `Q005, Q010, Q011, Q014, Q019, Q020, Q022, Q024, Q026, Q027, Q030, Q035`. They remained the primary diagnostic subset; no question was added or removed after unblinding.

## Prompt Intervention

- Control: `ddrag-production-v1`
- Candidate: `ddrag-answer-completeness-v1`
- Generation model: `qwen2.5:1.5b` for both
- Prompt characters: 2022 control, 3320 candidate, +1298 (+64.193867%)

## Experimental Controls

Source state, corpus/index, generation model, embedding model, benchmark, question set, runner, sampling, retrieval, top-K, and chunking were held fixed. Canonical differences were limited to `prompt.prompt_id` and `prompt.prompt_sha256`. Ordered retrieval identity was identical for 40/40 questions.

## Blinded Qualitative Method

> blinded dual-reviewer qualitative evaluation with independent disagreement adjudication and post-freeze unblinding

The frozen merge contains 520 final label fields: 457 reviewer-consensus fields and 63 blinded-adjudicated fields. Thirty-one items required at least one adjudicated field.

## Overall Qualitative Results

| Dimension | Control | Candidate |
|---|---|---|
| Correctness C/MC/P/I/NA | 12/7/13/8/0 | 8/12/16/4/0 |
| Grounding G/PG/U/NA | 26/8/6/0 | 23/14/3/0 |
| Refusal A/FR/OR/NA | 2/1/1/36 | 3/0/2/35 |
| Failure tags | ambiguity_issue=3, citation_issue=2, failed_refusal=2, hallucination=8, over_refusal=1, partial_answer=22, retrieval_failure=3, unsupported_claim=4 | ambiguity_issue=3, citation_issue=6, hallucination=9, over_refusal=2, partial_answer=24, retrieval_failure=4, unsupported_claim=6 |

Correctness transitions: candidate higher 7, same 24, lower 9, not comparable 0.

| Control \ Candidate | correct | mostly_correct | partial | incorrect | not_applicable |
|---|---:|---:|---:|---:|---:|
| correct | 6 | 5 | 1 | 0 | 0 |
| mostly_correct | 0 | 4 | 2 | 1 | 0 |
| partial | 1 | 1 | 11 | 0 | 0 |
| incorrect | 1 | 2 | 2 | 3 | 0 |
| not_applicable | 0 | 0 | 0 | 0 | 0 |


Grounding transitions: candidate higher 8, same 22, lower 10, not comparable 0.

| Control \ Candidate | grounded | partially_grounded | unsupported | not_applicable |
|---|---:|---:|---:|---:|
| grounded | 18 | 8 | 0 | 0 |
| partially_grounded | 3 | 3 | 2 | 0 |
| unsupported | 2 | 3 | 1 | 0 |
| not_applicable | 0 | 0 | 0 | 0 |


Raw refusal transitions:

| Control \ Candidate | appropriate | failed_refusal | over_refusal | not_applicable |
|---|---:|---:|---:|---:|
| appropriate | 2 | 0 | 0 | 0 |
| failed_refusal | 1 | 0 | 0 | 0 |
| over_refusal | 0 | 0 | 0 | 1 |
| not_applicable | 0 | 0 | 2 | 34 |


## Primary 12-Question Target Subset

| Measure | Control | Candidate |
|---|---:|---:|
| Correct | 0 | 0 |
| Mostly correct | 4 | 2 |
| Partial | 4 | 7 |
| Incorrect | 4 | 3 |
| Not applicable | 0 | 0 |

- Candidate higher (2): Q020, Q024
- Same (7): Q005, Q011, Q014, Q022, Q026, Q030, Q035
- Candidate lower (3): Q010, Q019, Q027
- Not comparable (0): none

Answer to the primary descriptive question: the candidate did not improve correctness on the predeclared target subset; lower transitions outnumbered higher transitions 3 to 2.

Target-subset failure tags:

| Tag | Control | Candidate |
|---|---:|---:|
| failed_refusal | 0 | 0 |
| hallucination | 4 | 4 |
| over_refusal | 0 | 2 |
| partial_answer | 9 | 8 |
| retrieval_failure | 0 | 0 |
| unsupported_claim | 2 | 2 |

## Complete-Retrieval Generation Failures

The denominator was 26 answerable complete-retrieval questions for each prompt.

| Observation | Control | Candidate | Candidate − control |
|---|---:|---:|---:|
| Partial or incorrect | 12/26 | 12/26 | +0 |
| `partial_answer` tag | 12/26 | 13/26 | +1 |

## Failure Modes and Grounding

Overall failure tags are non-exclusive. Candidate-minus-control changes include hallucination +1 (8→9), unsupported claim +2 (4→6), partial answer +2 (22→24), retrieval failure +1 (3→4), over-refusal +1 (1→2), failed-refusal tag -2 (2→0), citation issue +4 (2→6), and no change in ambiguity issues or request errors.

Grounding shifted from 26/8/6 grounded/partially-grounded/unsupported for control to 23/14/3 for candidate. Although unsupported labels declined, paired downward grounding transitions (10) exceeded upward transitions (8).

## Refusal / Safety Behavior

| Question | Group | Control correctness / grounding / refusal | Candidate correctness / grounding / refusal |
|---|---|---|---|
| Q042 | ambiguous | incorrect / unsupported / over_refusal | incorrect / partially_grounded / not_applicable |
| Q043 | ambiguous | incorrect / unsupported / not_applicable | partial / grounded / not_applicable |
| Q045 | ambiguous | partial / partially_grounded / not_applicable | partial / grounded / not_applicable |
| Q046 | unanswerable | incorrect / unsupported / failed_refusal | correct / grounded / appropriate |
| Q047 | unanswerable | correct / grounded / appropriate | correct / grounded / appropriate |
| Q048 | unanswerable | correct / grounded / appropriate | correct / grounded / appropriate |

The candidate corrected the failed refusal on Q046; Q047 and Q048 remained appropriate in both arms. Among ambiguous questions, candidate labels improved on Q043 and grounding improved on Q042/Q045, while Q042 remained incorrect and acquired hallucination/partial-answer tags. Candidate over-refusal labels occurred on two non-edge answerable questions; the frozen overall refusal distribution was 3 appropriate, 0 failed, 2 over-refusal, and 35 not-applicable.

## Category Analysis

Correctness columns are C/MC/P/I/NA. Grounding columns are G/PG/U/NA. No category winner is declared.

| Category | n | Control correctness | Candidate correctness | Control grounding | Candidate grounding | Control tags | Candidate tags |
|---|---:|---|---|---|---|---|---|
| ambiguous | 3 | 0/0/1/2/0 | 0/0/2/1/0 | 0/1/2/0 | 2/1/0/0 | ambiguity_issue=3, failed_refusal=1, hallucination=1, over_refusal=1, partial_answer=2, unsupported_claim=1 | ambiguity_issue=3, hallucination=1, partial_answer=3 |
| direct_factual | 8 | 5/1/1/1/0 | 3/3/0/2/0 | 5/2/1/0 | 4/2/2/0 | citation_issue=1, hallucination=1, unsupported_claim=2 | citation_issue=2, hallucination=2, over_refusal=1, partial_answer=2, unsupported_claim=1 |
| multi_chunk | 6 | 0/1/5/0/0 | 0/0/6/0/0 | 3/3/0/0 | 4/2/0/0 | hallucination=2, partial_answer=6, retrieval_failure=1, unsupported_claim=1 | citation_issue=2, hallucination=1, partial_answer=6, retrieval_failure=2, unsupported_claim=1 |
| multi_condition | 6 | 1/1/2/2/0 | 0/1/5/0/0 | 4/0/2/0 | 1/5/0/0 | hallucination=2, partial_answer=5 | citation_issue=1, hallucination=3, partial_answer=4, unsupported_claim=2 |
| policy_eligibility | 5 | 2/0/2/1/0 | 1/3/1/0/0 | 4/1/0/0 | 2/3/0/0 | hallucination=1, partial_answer=2 | citation_issue=1, hallucination=2, partial_answer=2, unsupported_claim=1 |
| procedural | 6 | 2/3/0/1/0 | 1/4/0/1/0 | 5/1/0/0 | 5/0/1/0 | partial_answer=4, retrieval_failure=1 | over_refusal=1, partial_answer=4, retrieval_failure=1 |
| summarization | 3 | 0/1/2/0/0 | 0/1/2/0/0 | 3/0/0/0 | 2/1/0/0 | partial_answer=3, retrieval_failure=1 | partial_answer=3, retrieval_failure=1, unsupported_claim=1 |
| unanswerable | 3 | 2/0/0/1/0 | 3/0/0/0/0 | 2/0/1/0 | 3/0/0/0 | citation_issue=1, failed_refusal=1, hallucination=1 | none |

## ExpectedPointCoverage Crosscheck

ExpectedPointCoverage changed from 13/18 to 11/18. Changed eligible questions were:

| Question | Matched-point delta | Frozen correctness direction | Relationship |
|---|---:|---|---|
| Q007 | -1 | same | proxy_label_divergence_or_matcher_sensitivity_not_distinguishable_from_frozen_labels |
| Q009 | -1 | lower | aligned_with_semantic_direction |
| Q027 | -1 | lower | aligned_with_semantic_direction |
| Q034 | +1 | lower | proxy_label_divergence_or_matcher_sensitivity_not_distinguishable_from_frozen_labels |

The two-point net decline is a mixture. Q009 and target Q027 show lower automated matches aligned with lower qualitative correctness. Q007 loses a match despite unchanged correctness, while Q034 gains a match despite lower correctness. The frozen evidence cannot distinguish lexical matcher sensitivity from other proxy limitations for those divergent cases. Target Q005 remained 0/1 with unchanged correctness, Q030 remained 0/2 with unchanged correctness, and Q035 remained 1/1 with unchanged correctness.

## Deterministic Results

| Metric | Control | Candidate | Candidate − control |
|---|---:|---:|---:|
| Hit@5 | 0.972973 | 0.972973 | 0 |
| Recall@5 | 0.850901 | 0.850901 | 0 |
| CompleteEvidence@5 | 0.729730 | 0.729730 | 0 |
| ExpectedPointCoverage | 13/18 (0.681818) | 11/18 (0.515152) | -2 points |
| Automated refusal signal | 0/3 | 0/3 | 0 |
| Operational success | 40/40 | 40/40 | 0 |
| Operational failure | 0/40 | 0/40 | 0 |
| Retrieval mismatch | 0/40 | 0/40 | 0 |

Retrieval metrics are experimental controls because reviewer-visible and ordered retrieval identities were identical for all questions.

## Latency and Operational Reliability

| Path | Control mean | Candidate mean | Delta | Percent change |
|---|---:|---:|---:|---:|
| Retrieval | 176.547 ms | 168.402 ms | -8.145 ms | descriptive control only |
| Generation endpoint | 14551.198 ms | 15091.539 ms | +540.340 ms | +3.713% |
| Combined/end-to-end | 14727.746 ms | 15259.941 ms | +532.195 ms | +3.614% |

The generation-endpoint observation is not described as pure model compute time. Both arms completed 40/40 questions with zero operational failures.

## Limitations

- One paired run was executed per prompt under runtime-default sampling.
- Results are limited to this benchmark, corpus, model, hardware, and local runtime.
- No statistical-significance or universal-superiority claim is made.
- ExpectedPointCoverage is an automated proxy, not semantic correctness.
- Prompt length increased as part of the intervention and was not isolated separately.
- Historical E10 is contextual only; the causal reference is the fresh E11 control.

## Engineering Decision

Retain `ddrag-production-v1` as the preferred generation prompt for this tested DDRAG setup. The candidate did not improve the predeclared target subset or reduce complete-retrieval answer failures, and its refusal gains came with broader grounding/failure-tag and latency trade-offs. This is a scoped engineering decision, not a universal prompt ranking. The production default was not modified.

## Artifact Hashes

- `e11_unblinded_question_qualitative.jsonl`: `710f7f37cc05176cb7175acdf4631b357bf477bafeb576660d7ebc56a5ed730a`
- `e11_target_subset_qualitative_comparison.jsonl`: `969ece8752a45d7d7d848bf6efdd62e498551415be140f789c08cb3718ea06e0`
- `e11_expected_point_qualitative_crosscheck.jsonl`: `96eac7ed0ea9c9c678cb680045a75c029794707e5e8b13bb13caba11956a1642`
- `e11_prompt_final_comparison.json`: `a12666d1f02b7fa1688ef9d42846f58cb1e62a54dfb2d58b099a08fd0db1e0e3`
- The Markdown report hash is calculated after this file is frozen and reported externally.
