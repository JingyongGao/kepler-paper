# Project Kepler Paper Revision v1.1 — Revision Summary

## Scope

This revision is limited to manuscript positioning, interpretation, narrative
structure, and consistency checks. It introduces no experiment, training run,
optimizer step, statistical redefinition, or numerical-result change. All
scientific artifacts remain anchored to `kepler-v3.6-confirm1`
(`f9fc22d85cd52863087afbb3f924466a63b30ba2`).

## Reviewer-driven changes

1. **Introduction and positioning.** The Introduction now centers the question:
   “How does training-distribution geometry affect the accessibility of
   task-sufficient representations when the final task information is fixed?”
   It explicitly distinguishes variable identification from variable
   composition in the parent-exposed interface and rejects scaling-law or
   universal-intelligence framing.

2. **Related Work.** A dedicated section now distinguishes the contribution
   from fixed-distribution grokking, connects diagnostic Fourier probes to
   mechanistic studies of modular arithmetic without assuming a specific
   circuit, contrasts the intervention with standard curriculum learning, and
   gives a bounded PID connection. The contribution is described as
   “distributional bootstrapping of representation formation.”

3. **Mechanistic interpretation.** The Discussion now contains “Possible
   mechanism: symmetry breaking and optimization accessibility.” It connects
   cue-dose ordering, persistence after cue removal, and gradient-coherence
   correlates while explicitly declining causal-mediation and universal
   loss-landscape claims.

4. **Practical implication.** A new subsection states a possible design
   principle for temporary training distributions in high-order compositional
   tasks. It explicitly does not recommend low-order cues for all AI training.

5. **Limitations.** The revised section separately addresses model scale,
   parent-exposed input control, exact soft-label training, finite-field task
   scope, and the absence of a universal mechanism claim. Existing boundaries
   concerning the recurrent task, thermodynamic language, and the original
   scaling fit remain intact.

6. **Narrative density.** Objective details, exhaustive evaluation, probe and
   bootstrap definitions, cue-family root solving, and construction audits
   moved to Supplementary Methods. The main text retains the scientific
   question, controlled construction, exploratory/confirmatory separation,
   exact confirmatory test, persistence result, heterogeneity, and implications.

7. **References and embedding.** The LaTeX manuscript now uses normal
   `\cite`, `\ref`, and labeled figure/table references. All seven frozen
   figures remain embedded in the source; all eight frozen tables are
   cross-referenced. Markdown table-file placeholders were removed from the
   narrative.

## Related-work sources added

- Power et al. (2022), “Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets,” [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).
- Nanda et al. (2023), “Progress Measures for Grokking via Mechanistic
  Interpretability,” [arXiv:2301.05217](https://arxiv.org/abs/2301.05217).
- Bengio et al. (2009), “Curriculum Learning,”
  [ICML paper](https://icml.cc/2009/papers/119.pdf).
- Williams and Beer (2010), “Nonnegative Decomposition of Multivariate
  Information,” [arXiv:1004.2515](https://arxiv.org/abs/1004.2515).

## Consistency audit

- Frozen v3.5 result SHA-256:
  `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.
- Frozen v3.6 result SHA-256:
  `5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0`.
- Read-only result verification recomputed the distribution invariants, 54
  transition codings, and the exact 1,679,616-assignment test. It recovered
  observed statistic 110, upper-tail count 16, and one-sided
  `p = 9.525986892242037e-06`.
- Frozen-artifact verification reported all 18 tracked scientific artifacts
  unchanged and no changes outside the paper-packaging scope.
- LaTeX compiled successfully with seven embedded figures, seven figure labels,
  eight unique table references, and no undefined-reference warnings.
- Optimizer steps during revision: **0**.

The machine-readable result of this audit is
`paper/claim_boundary_check.json`.
