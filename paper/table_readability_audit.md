# Project Kepler v1.2.1 Table Readability Audit

## Result

**Pass.** Every main-text table is typeset at LaTeX `\small` in the 11 pt
document (approximately 10 pt body text), uses an unscaled `tabular`, and is
legible at 100% zoom. No main-text table uses `\resizebox`.

All values come from the same frozen table builders. The eight existing CSV
table hashes are unchanged from the v1.2 baseline, and every complete table
remains in the submission package and in the Supplement.

## Main-text presentation

| Main table | Narrative-facing fields retained | Detail moved to Supplement |
| --- | --- | --- |
| Representative distribution invariants | R0/R1/R2 condition, achieved cue index, minimum pair probability, maximum marginal error, conditional entropy, joint-information error, full-support flag | All registered v3.5/v3.6 rules, permutations, dose parameters, and provenance fields |
| Exploratory outcomes | C0-C3 onset by 5M/10M, establishment by 10M, mean R at 5M/10M | Geometry controls, complete onset exposure vectors, terminal ordinals, status and provenance |
| Confirmatory outcomes | R0/R1/R2 onset by 5M/10M, establishment by 10M, mean R at 5M/10M | Complete onset vectors, terminal ordinals, status and provenance |
| Primary exact test | Blocks, exact assignments, observed statistic, upper-tail count, one-sided p-value | Rule/permutation strata, inference-status flags and audit fields |
| Paired R differences | Comparison, paired blocks, mean R differences at 5M/10M, mean onset-ordinal difference | Medians and complete provenance fields |
| Cue-removal persistence | Confirmatory R1/R2 positive and retained counts at 5M/6M/10M | Complete exploratory and confirmatory retention records, fractions and sampler audit |

The protocol-integrity and claim-boundary tables no longer interrupt the main
narrative. They appear with the complete registered distribution, outcome,
exact-test, paired-difference, and persistence tables in Supplementary
Section 14.7.

## Mechanical checks

- Compact main-table files: 6.
- Complete supplementary tables retained: 8.
- Main-table body size: approximately 10 pt.
- Main-table caption and bibliography size: approximately 9 pt.
- Main-table scaling commands: 0.
- Main-table overfull or underfull warnings: 0.
- Clipped main tables: 0.
- Existing table CSV hashes changed: 0 of 8.
- Cross-references: normal LaTeX labels and references; 13 unique explicit
  table destinations cover the six main summaries and complete supplementary
  table ranges.
- Visual review: Tables 1-6 were inspected on rendered pages 4-7 at 100% zoom.
