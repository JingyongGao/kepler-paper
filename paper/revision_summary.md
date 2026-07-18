# Project Kepler Paper v1.2 — Revision Summary

## Scope

This submission-hardening revision changes only the manuscript, bibliography,
publication figures, deterministic paper builders, and submission packages. It
adds no experiment, training run, optimizer step, transition redefinition,
post-hoc statistical test, or numerical-result change.

The pre-task paper commit was
`cd25e702358f74121b7942c0354f8d3d257bd0b5`. The scientific source remains
`kepler-v3.6-confirm1`
(`f9fc22d85cd52863087afbb3f924466a63b30ba2`).

## Manuscript hardening

1. Corrected the title and consistently used “higher-order compositional
   representations.”
2. Replaced ambiguous “independent” and “preregistered” language with
   prospectively frozen protocol wording. The Abstract now names exactly what
   was frozen before confirmatory training.
3. Put the registered full-support cue distribution and the operational
   `C_single` definition in the main paper. The text states which quantities
   remain invariant and that cue-bearing parents need not remain jointly
   independent.
4. Recast the mechanism discussion as an experiment-bounded optimization
   interpretation. Gradient coherence is explicitly correlational and is not
   treated as an identified mediator.
5. Clarified that eight paired blocks are the primary inference units, while
   the 24 trajectories are paired condition observations. The within-block
   exchangeability null and inclusion of the observed assignment in the upper
   tail are explicit. Stratified tests are labeled secondary, nominal, and not
   multiplicity-adjusted.
6. Expanded related work across delayed generalization, curriculum and
   self-paced learning, continuation methods, modular-arithmetic mechanisms,
   probe limitations, optimization path dependence, and PID caveats.
7. Added the requested limitations concerning warmup timing, joint-parent
   dependence, scale, field size, noise, input interface, and exact-soft-label
   dependence.
8. Corrected provenance language: the v3.6 confirmatory manifest instantiated
   architecture and optimization settings frozen from v3.5.

## Figures, references, and packages

- Figure 1 now combines the parent-to-center-to-target graph, pure-synergy
  identities, R0/R2 pair-distribution geometry, and the 0–5M cue window. Every
  plotted value is derived from frozen distribution JSON.
- All seven figure families retain machine-readable source data. PNG outputs
  are 1800×1080, and vector PDF/SVG versions are present.
- The manuscript now uses BibTeX with 13 cited entries. Every cited entry has
  verified authors, title, venue, year, and DOI or arXiv identifier. One
  unverified candidate was kept in the human-review TODO and is not cited.
- Separate blinded and arXiv source packages were produced. The blinded source
  and PDF contain no detected local path, machine name, Git username,
  identifying repository URL, collaboration name, or frozen Git tag.
- The formal submission PDF is the blinded 12-page letter-size build:
  `paper/output/pdf/kepler_paper_v1_2_submission.pdf`.

## Verification outcome

- Frozen v3.5 result SHA-256:
  `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.
- Frozen v3.6 result SHA-256:
  `5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0`.
- All 18 tracked scientific artifacts are byte-identical to the freeze record.
- All 330 v3.5 and 264 v3.6 checkpoint sidecar hashes pass.
- Read-only result verification recomputed the distribution invariants, 54
  transition cells, and the exact 1,679,616-assignment test. It recovered
  statistic 110, upper-tail count 16, and one-sided
  `p = 9.525986892242037e-06`.
- All 39 headline numerical claims rebuild exactly from frozen JSON.
- LaTeX, BibTeX, cross-reference, overfull-box, figure-resolution, anonymous
  metadata, and all-page visual checks pass.
- Submission PDF SHA-256:
  `37e91cef9e9fc79b453083eaaeeda455dc9aef714e67857a4ecb9cfe74a4aa1c`.
- Optimizer steps: **0**. New experiments: **0**. New statistical tests:
  **0**.
