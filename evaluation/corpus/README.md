# Evaluation Corpus

DDRAG's official evaluation uses one fixed local source document.

| Field | Value |
| --- | --- |
| Corpus ID | `csu_student_handbook_2019` |
| Expected local filename | `CSU-Student-Handbook.pdf` |
| SHA-256 | `ce3c2818ea5ba171a2e75b51d3db14970f57d4d54515eccc0c0a275470be4841` |
| Size | 1,784,539 bytes |
| Physical PDF pages | 93 |

## Public repository policy

The source PDF is not redistributed in this repository and is intentionally ignored by Git. Users seeking exact official-result reproduction must obtain the source document through an authorized, lawful source and place the matching file at `evaluation/corpus/CSU-Student-Handbook.pdf`. This repository does not provide or endorse an unofficial mirror.

The tracked `corpus_manifest.json` records the same identity metadata without containing source-document text. Verify the local file before an official evaluation run:

```powershell
Get-FileHash evaluation/corpus/CSU-Student-Handbook.pdf -Algorithm SHA256
```

The hexadecimal digest must match the documented SHA-256 (letter case does not change the digest). Strict validation also checks the filename, byte size, page count, benchmark anchors, and benchmark identity when the authorized private benchmark is locally available. Matching inputs are necessary to repeat the protocol; they do not guarantee identical model outputs under runtime-default sampling.
