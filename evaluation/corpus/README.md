# Evaluation Corpus

## Purpose

This directory contains local source material used for DDRAG evaluation.

## Expected corpus

The expected local file is `CSU-Student-Handbook.pdf`, identified by:

```text
corpus_id = csu_student_handbook_2019
```

## Public repository policy

The source PDF is intentionally not tracked by Git and is not included in the repository. The tracked `corpus_manifest.json` records the SHA-256 digest that identifies the exact local file used by official evaluation runs. Users who want to reproduce this benchmark must supply the source PDF locally at the expected path.

## Verification

From the repository root, calculate the file's SHA-256 digest in PowerShell:

```powershell
Get-FileHash evaluation/corpus/CSU-Student-Handbook.pdf -Algorithm SHA256
```

The calculated digest, compared case-insensitively, must match the lowercase `sha256` value in `corpus_manifest.json`.
