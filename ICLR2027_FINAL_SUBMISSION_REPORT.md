# ICLR 2027 Final Submission Report

Freeze and packaging date: 2026-08-01

## Frozen state

- Requested short commit: `3f3d296`
- Verified full commit: `3f3d296b65c04f4f3c5597450ebe3cf7480315b3`
- Parent commit: `098cf1e1222d8df4c2e3277859e2fc802b1d9ff0`
- Annotated tag: `kepler-iclr2027-scale-robustness1`
- Tag object: `ee30213f71bc80bf57010db8fdb70323b66b74ce`
- Peeled tag target: `3f3d296b65c04f4f3c5597450ebe3cf7480315b3`
- Working branch: `kepler-iclr2027-scale-robustness`
- Remote: confirmed private origin; the branch and annotated tag were pushed and their refs verified.

All pre-existing branches, tags, frozen TMLR material, v1.3 audit material,
checkpoints, manifests, and result files were preserved. The final package work
adds manuscript, audit, registry, and packaging artifacts only.

## Cross-scale manuscript integration

The compact main-text table reports R0 versus R2 at three displayed scales:

| Scale | Status | Paired direction | Verified onset R0 | Verified onset R2 | Mean R2-R0 onset ordinal |
|---|---|---:|---:|---:|---:|
| approximately 1M | post-confirmatory robustness | R2 earlier 4/4 | 0/4 | 4/4 | -5.25 |
| approximately 2M | confirmatory v3.6 | R2 earlier 8/8 | 1/8 | 8/8 | -5.125 |
| approximately 5M | post-confirmatory robustness | R2 earlier 4/4 | 3/4 | 4/4 | -3.75 |

The 2M row is sourced only from
`artifacts/cue_dose_confirmatory_results.json`; the 1M/5M rows are sourced only
from `artifacts/iclr2027_scale_robustness_results.json`. The manuscript labels
the extension post-confirmatory, excludes it from the original v3.6 p-value,
states that rows are not pooled, reports no new combined p-value, and states
that the three selected widths are not a scaling-law fit.

The original v3.6 exact one-sided p-value remains
`0.000009525986892242037`. No inferential p-value was computed for the scale
extension.

## Narrative and source audit

- Main text: 9 pages.
- References begin: page 10.
- Appendix begins: page 11, after references.
- Main text retains the task/cue construction, discovery/confirmation separation, primary confirmatory effect, persistence, compact cross-scale robustness, and limitations.
- Complete cross-scale block/metric tables and all v1.3 Hessian diagnostics are in the appendix.
- The Hessian table is sourced from an exact copy of the frozen v1.3 JSON at tag `kepler-v1.3-transition-audit`; SHA-256 `a38246e51e1b851b7672072a50f03befee514b05c23b3387bfb31d4f663e6af3`.
- `ICLR2027_NUMERICAL_CLAIM_REGISTRY.json` carries the 39 pre-existing frozen claims plus the direct cross-scale and exploratory Hessian source rows.
- Registry generation performed zero optimizer steps and zero new statistical tests.

## ICLR compliance audit

- PASS — current official ICLR 2027 style ZIP re-downloaded from the author-guideline link; SHA-256 `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
- PASS — the four submission style files exactly match the official copies.
- PASS — review-mode double-blind anonymity.
- PASS — required AI-use statement appears on page 10, before references.
- PASS — no TMLR submission identifier, private repository name, private remote, or local path in the final PDF or supplemental ZIP.
- PASS — all 32 PDF font records are embedded; page size is US Letter.
- PASS — two consecutive source-date-fixed builds produced byte-identical PDFs.
- PASS — all numerical claims are indexed to frozen JSON in `ICLR2027_NUMERICAL_CLAIM_REGISTRY.json`.

## Integrity and verification

- All 176 scale-extension checkpoint files were rehashed from disk and matched the recorded SHA-256 values and sidecars.
- All 16 zero-step sampler-switch checks remain passed in the frozen result JSON.
- Full repository tests: 69 passed.
- Clean-extraction supplemental tests: 64 passed, 5 explicitly deselected because the anonymous archive intentionally excludes checkpoint binaries.
- Clean-extraction JSON sidecars: all included sidecars passed.
- Clean-extraction identity/path scan: no matches.

The complete per-checkpoint inventory is
`artifacts/iclr2027_scale_robustness_checkpoint_hashes.sha256`. All manifest,
result, report, PDF, and ZIP hashes are recorded in
`paper/output/iclr2027/iclr2027_final_package_hashes.json`.

## Final artifacts

- PDF: `paper/output/pdf/iclr2027_final_anonymous_submission.pdf`
  - bytes: `381642`
  - SHA-256: `42af0eec0c557332ee2f1acb9481eee01d01da99a30d626bb52bb76dc285304c`
- Supplement: `paper/output/iclr2027/iclr2027_final_anonymous_supplement.zip`
  - bytes: `2332985`
  - SHA-256: `66f3c2cfe6e537b05b57f3874de60095f94adf599eb32c18d6fdf4bb1a449638`

## Stop-condition attestation

No early cue-removal continuation, new model scale, recurrent experiment,
sampled-NLL experiment, Hessian extension, new statistical test, optimizer
step, or checkpoint modification was performed during final packaging. Work
stopped after the final anonymous PDF, supplemental ZIP, hash registry, and
submission audits were produced.
