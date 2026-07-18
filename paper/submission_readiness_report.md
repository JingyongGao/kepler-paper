# Project Kepler Paper v1.2 — Submission Readiness Report

## Decision

**Ready for blinded conference submission.** All requested manuscript,
bibliography, figure, packaging, anonymity, numerical, and frozen-science
checks pass. No training, optimizer step, new experiment, transition
redefinition, numerical change, or new statistical test was performed.

## Deliverables

| Deliverable | Status | Location |
| --- | --- | --- |
| Revised Markdown manuscript | Pass | `paper/manuscript.md` |
| Revised LaTeX manuscript | Pass | `paper/manuscript.tex` |
| Blinded compiled submission PDF | Pass | `paper/output/pdf/kepler_paper_v1_2_submission.pdf` |
| Blinded source package | Pass | `paper/anonymized/` |
| arXiv source package | Pass | `paper/arxiv/` |
| Reviewer response matrix | Pass | `paper/reviewer_response_matrix.md` |
| Numerical claim registry | Pass | `paper/numerical_claim_registry.json` |
| Reference verification | Pass | `paper/reference_verification.json` |
| Claim-boundary audit | Pass | `paper/claim_boundary_check.json` |

## Scientific freeze and numerical integrity

- Frozen source tag:
  `kepler-v3.6-confirm1`
  (`f9fc22d85cd52863087afbb3f924466a63b30ba2`).
- Pre-task paper commit:
  `cd25e702358f74121b7942c0354f8d3d257bd0b5`.
- All eight manifest hashes and all 18 tracked scientific-artifact hashes are
  recorded in `paper/submission_science_baseline.json`.
- Post-task byte comparison: **18/18 unchanged**.
- Checkpoint sidecar verification: **330/330 v3.5** and **264/264 v3.6**
  pass.
- Frozen v3.5 result hash:
  `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.
- Frozen v3.6 result hash:
  `5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0`.
- Numerical registry: **39/39** headline claims rebuild exactly from frozen
  machine-readable artifacts.
- Read-only primary-test recomputation: 1,679,616 assignments, statistic 110,
  upper-tail count 16, one-sided
  `p = 9.525986892242037e-06`.

## Build and presentation checks

- LaTeX and BibTeX build: **pass** for manuscript, blinded package, and arXiv
  package.
- Undefined citations or references: **0**.
- Overfull/underfull box warnings: **0**.
- Visible hyperlink boxes: **0** (`hidelinks` enabled).
- Embedded figure references: **7/7**.
- Unique table references: **8/8**.
- Reference validation: **13 cited = 13 BibTeX = 13 verified**; every cited
  entry has authors, title, venue, year, and DOI/arXiv identifier.
- Figures: seven 1800×1080 PNGs plus seven vector PDFs and seven SVGs; frozen
  source data is retained beside each figure.
- Final PDF: 12 pages, US letter, PDF 1.5.
- Visual QA: all 12 rendered pages inspected; no clipping, overlap, missing
  content, or malformed references observed.

## Blinding and metadata

The blinded TeX, bibliography, README, extracted PDF text, and PDF metadata
were scanned for local paths, mounted-volume paths, machine/user names,
identifying repository hosts, the collaboration author string, and the frozen
Git tag. Forbidden matches: **0**. Scientific seeds and protocol identifiers
remain intact.

## Final package hashes

- Blinded submission PDF:
  `37e91cef9e9fc79b453083eaaeeda455dc9aef714e67857a4ecb9cfe74a4aa1c`.
- arXiv manuscript PDF:
  `64910ffa7bf1f84091b1a0e97cf9f2cf599a7c60e3acbb8a3778f026d601d506`.

## Remaining human actions

No scientific action is required. Before uploading, the authors should only
apply venue-specific style files, page limits, author-declaration fields, and
supplementary-file size limits if the target venue requires them. Such edits
should be followed by the same build, anonymity, numerical-registry, and
frozen-artifact checks.
