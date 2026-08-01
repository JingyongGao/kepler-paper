# Project Kepler ICLR 2027 package

This directory is an ICLR-specific, anonymous paper package. It does not
replace or modify the frozen TMLR submission.

## Build

```bash
uv run python paper/iclr2027/scripts/build_iclr2027_package.py
```

The build compiles the official ICLR 2027 style twice, requires byte-identical
PDF output, checks the nine-page main-text boundary, verifies the required AI
use statement, audits anonymity and font embedding, and creates a deterministic
anonymous code ZIP without checkpoints.

Outputs:

- `paper/output/pdf/kepler_paper_iclr2027_anonymous_draft.pdf`
- `paper/output/iclr2027/kepler_iclr2027_anonymous_code.zip`
- `paper/iclr2027/build_record.json`

The new 1M/5M experiment is labeled post-confirmatory robustness throughout.
It is excluded from the original v3.6 p-value and is not a scaling-law fit.
The optional early-removal continuation is not part of this package because it
was not separately authorized.
