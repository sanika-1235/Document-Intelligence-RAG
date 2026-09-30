# Evaluation Benchmark Metadata

DDRAG's official evaluation uses a fixed benchmark derived from the Caraga State University Student Handbook corpus. The public repository provides benchmark identity and aggregate composition only; it does not redistribute the handbook-derived questions, reference answers, gold evidence text, quoted anchors, or review notes.

## Identity

| Field | Value |
| --- | --- |
| Benchmark ID | `csu_handbook` |
| Version | `1.0.0` |
| Question count | 40 |
| SHA-256 | `f2d5e90b252f288572f617d56e27f1a003e18b1171421881c636793e32137ae0` |
| Answerable | 34 |
| Ambiguous | 3 |
| Unanswerable | 3 |

Category composition:

| Category | Count |
| --- | ---: |
| `direct_factual` | 8 |
| `procedural` | 6 |
| `multi_condition` | 6 |
| `multi_chunk` | 6 |
| `policy_eligibility` | 5 |
| `summarization` | 3 |
| `ambiguous` | 3 |
| `unanswerable` | 3 |

## Publication boundary

Each benchmark record contains a stable question ID, benchmark ID/version, review status, category, question, answerability, expected behavior, reference answer where applicable, expected points with automation eligibility, and evidence units. Evidence units identify a physical PDF page and optional printed-page label/section, with a text anchor for matching. The [public schema](../schemas.py) defines the contracts without distributing the official records.

The official benchmark artifact is retained locally and is intentionally excluded from the public repository because repository evidence does not establish permission to redistribute its handbook-derived content. The SHA-256 digest identifies the exact frozen artifact used for the published results; it is not a download link.

Benchmark evidence uses stable PDF page, section, and text-anchor identities rather than database chunk IDs. The public evaluation framework remains inspectable and testable with synthetic inputs. Exact reproduction of the official results requires authorized local corpus and benchmark artifacts whose hashes match the published metadata.

See the [corpus policy](../corpus/README.md), [evaluation framework](../README.md), [evaluation article](../reports/ddrag_evaluation_article.md), and [frozen evidence report](../reports/ddrag_evaluation_report.md).
