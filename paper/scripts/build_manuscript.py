from __future__ import annotations

from common import (
    PAPER,
    V35_AUDIT,
    V35_RESULTS,
    V36_AUDIT,
    V36_MANIFEST,
    V36_RESULTS,
    atomic_write_text,
    load_json,
    require_frozen_source_hashes,
)


TITLE = (
    "Transient Low-Order Cues Bootstrap High-Order Compositional "
    "Representations in Transformers"
)


def count(summary: dict, key: str) -> tuple[int, int]:
    values = summary[
        "onset_exposure_by_seed"
        if "onset_exposure_by_seed" in summary
        else "onset_exposure_by_block"
    ]
    return int(round(float(summary[key]) * len(values))), len(values)


def ff(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def main() -> None:
    require_frozen_source_hashes()
    v35_audit = load_json(V35_AUDIT)
    v36_audit = load_json(V36_AUDIT)
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    v36_manifest = load_json(V36_MANIFEST)

    p = int(v36_audit["p"])
    eta = float(v36_audit["eta"])
    first_distribution = v36_audit["conditions"][0]
    noise_entropy = first_distribution["entropies_nats"]["H_Y_given_X1_X2"]
    joint_information = first_distribution["mutual_information_nats"]["I_Y_X1_X2"]
    v35_test = v35["ordered_dose_trend"]
    primary = v36["primary_confirmatory_test"]["ordered_onset_test"]
    summaries = v36["primary_confirmatory_test"]["condition_summaries"]
    onset_counts = {
        condition: count(summary, "fraction_onset_verified_by_10M")
        for condition, summary in summaries.items()
    }
    establishment_counts = {
        condition: count(summary, "fraction_established_by_10M")
        for condition, summary in summaries.items()
    }
    paired = v36["primary_confirmatory_test"]["paired_differences"]
    rule0 = v36["by_coefficient_rule"]["rule_0"]["condition_summaries"]
    rule1 = v36["by_coefficient_rule"]["rule_1"]["condition_summaries"]
    r1_rule0 = count(rule0["R1"], "fraction_onset_verified_by_10M")
    r1_rule1 = count(rule1["R1"], "fraction_onset_verified_by_10M")
    r2_rule0 = count(rule0["R2"], "fraction_onset_verified_by_10M")
    r2_rule1 = count(rule1["R2"], "fraction_onset_verified_by_10M")
    v35_summaries = v35["condition_summaries"]
    c0_onset = count(v35_summaries["C0"], "fraction_onset_verified_by_10M")
    c1_onset = count(v35_summaries["C1"], "fraction_onset_verified_by_10M")
    c2_onset = count(v35_summaries["C2"], "fraction_onset_verified_by_10M")
    c3_onset = count(v35_summaries["C3"], "fraction_onset_verified_by_10M")
    geometry = v35["matched_singleton_index_geometry"]["comparisons_against_m64_C2"]
    retained_r1 = summaries["R1"]["post_switch_threshold_retention"]
    retained_r2 = summaries["R2"]["post_switch_threshold_retention"]
    retained_count = int(retained_r1["pre_switch_threshold_positive_blocks"]) + int(
        retained_r2["pre_switch_threshold_positive_blocks"]
    )
    registered_run = v36_manifest["runs"][0]
    optimizer = v36_manifest["frozen_protocol"]["optimizer"]
    learning_rate_exponent = -4
    learning_rate_mantissa = optimizer["learning_rate"] / (
        10 ** learning_rate_exponent
    )

    md = f"""# {TITLE}

## Abstract

How a model acquires a high-order predictive representation can depend on
training-distribution structure even when the target conditional, irreducible
noise, marginal entropy, and total predictable information are fixed. We
studied a controlled finite-field task in which two uniformly distributed
parents determine a modular center state and a fixed additive-noise channel
determines the target. Under uniform full support, neither parent alone carries
predictive information, although the pair jointly carries
{ff(joint_information)} nats. We constructed strictly positive full-support
training distributions containing transient low-order cues while preserving
uniform marginals and the exact target conditional. An exploratory v3.5
experiment identified earlier predictive-state formation with increasing
registered cue dose. We then froze the model, optimizer, objective, evaluation,
onset definition, and test before an independent v3.6 replication using two new
coefficient rules, two new cue permutations, two initialization seeds, and
three cue doses. Across eight paired blocks, verified onset occurred in
{onset_counts['R0'][0]}/{onset_counts['R0'][1]},
{onset_counts['R1'][0]}/{onset_counts['R1'][1]}, and
{onset_counts['R2'][0]}/{onset_counts['R2'][1]} blocks for R0, R1, and R2.
The preregistered one-sided exact blocked Page-style test enumerated all
{primary['exact_permutation_count']:,} assignments and gave
`p = {primary['exact_one_sided_p']:.16g}`. All {retained_count} nonzero-cue
cells positive at cue removal retained the representation threshold at 6M and
10M exposures. These results show that temporary cue-bearing full-support
distributions causally changed representation-onset timing or probability
within the frozen parent-exposed task and model family. They do not identify
singleton mutual information alone as the causal variable and do not establish
a universal Transformer property.

## Introduction

The central question is: **How does training-distribution geometry affect the
accessibility of task-sufficient representations when the final task
information is fixed?** A representation may be statistically available yet
difficult to discover because useful optimization directions are weak, cancel
across examples, or become accessible only after other features organize.
This motivates a controlled intervention that holds the target conditional
and its information budget fixed while changing the geometry of the early
training distribution.

Project Kepler isolates this question in a finite field. Earlier recurrent
experiments motivated a parent-exposed mechanism assay, but the present paper
does not claim that the recurrent production task has been solved. Instead, it
asks a narrower causal question: can a temporary low-order cue in a
full-support sampling distribution alter when a Transformer forms the
high-order predictive center state? The parent-exposed design intentionally
separates two problems: (1) identifying the relevant variables, which the
interface controls by presenting both parents directly, and (2) composing
those discovered variables into a predictive representation, which remains
the learning problem.

The experimental sequence was explicitly separated into discovery and
confirmation. Version 3.5 was exploratory and selected the cue doses and
frozen outcome definitions. Version 3.6 used new rules, new cue permutations,
new initialization seeds, and no v3.5 checkpoints. The v3.5 data are not
included in the v3.6 primary p-value. The aim is neither a new scaling law nor
a universal intelligence principle; it is a bounded causal test within one
controlled task and model family.

![Figure 1. Controlled task and cue schedule.](figures/fig01_task_cue_schematic.png)

*Figure 1.* The parent pair determines a modular center; fixed additive noise
determines the target. Cue-bearing sampling is used only from 0--5M exposures,
after which training returns to uniform full support. All figure values are
read from frozen JSON.

## Related Work

### Grokking and delayed generalization

Modular-arithmetic grokking studies established that small neural networks can
generalize long after achieving near-perfect training performance under a
fixed training distribution ([Power et al., 2022](https://arxiv.org/abs/2201.02177)).
This paper does not offer a universal explanation of grokking. It asks a
different question: whether the geometry of the *early* training distribution,
with the final conditional and information budget held fixed, changes the
accessibility or timing of a delayed representational transition.

### Mechanistic interpretability of modular arithmetic

Mechanistic analyses of modular addition have recovered Fourier-like internal
representations and circuits implementing trigonometric structure
([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)). Here, fixed-Fourier
probes are diagnostics alongside linear and fixed-capacity MLP probes. We do
not assume that a specific Fourier circuit must be learned, and the causal
result concerns the timing or probability of task-sufficient representation
formation rather than identification of a unique circuit.

### Curriculum learning

Classic curriculum learning changes example order or introduces examples and
concepts by increasing difficulty
([Bengio et al., 2009](https://icml.cc/2009/papers/119.pdf)). The present
intervention instead keeps the target conditional and Bayes information budget
fixed and changes only the geometry of the full-support training distribution
during an early window. We describe this narrower intervention as
**distributional bootstrapping of representation formation**.

### Partial Information Decomposition

Partial Information Decomposition separates redundant, unique, and synergistic
contributions in multivariate information
([Williams and Beer, 2010](https://arxiv.org/abs/1004.2515)). The full-support
base task is a pure-synergy construction in the limited operational sense
needed here: singleton information is zero while joint information is
nonzero. We do not attempt a full PID decomposition of the cue distributions
or learned representations.

## Controlled finite-field task

The field size was `p = {p}`. For coefficients `a1,a2 != 0`, the predictive
center was

`z = a1*x1 + a2*x2 mod p`.

The target followed `y = z + epsilon mod p`, with
`P(epsilon=0)=1-eta`, `P(epsilon=r)=eta/(p-1)` for nonzero `r`, and
`eta = {eta:g}`. Consequently,
`H(Y|X1,X2) = H(nu_eta) = {ff(noise_entropy)} nats`, and the target
conditional was unchanged across cue conditions. The primary error metric was

`R = E_h[KL(q_h || p_theta)] / (log(p) - H(nu_eta))`.

A uniform predictor has `R=1` and the analytic oracle has `R=0`. Exact
soft-label training, exhaustive evaluation, probe definitions, and bootstrap
details are given in Supplementary Methods.

## Pure-synergy information structure

With uniformly sampled independent parents and nonzero modular coefficients,
`X1`, `X2`, `Z`, and `Y` are uniform. Each individual parent is independent of
the target: `I(Y;X1)=I(Y;X2)=0`. Jointly, however, the pair determines the
noisy center and carries
`I(Y;X1,X2)=log(p)-H(nu_eta)={ff(joint_information)} nats`.

This pure-synergy baseline makes the construction unusually clean. The
model must compose both parents to predict the center, and no marginal
frequency cue points to the answer. It also makes clear why the registered
cue index must not be described as a unique-information percentage: it is a
scalar property of the constructed sampling family, not a decomposition of
all predictive information.

## Full-support cue-family construction

The cue family reweighted ordered parent pairs through a frozen permutation
while keeping every pair probability strictly positive. Exhaustive audits
confirmed uniform `X1`, `X2`, `Z`, and `Y`, invariant `H(Y|X1,X2)`, invariant
`I(Y;X1,X2)`, and exact uniform full support at zero cue. In v3.6 the maximum
marginal-uniformity error was
`{max(max(item['uniformity_max_absolute_error'].values()) for item in v36_audit['conditions']):.3e}`
and the minimum pair probability was
`{min(item['minimum_pair_probability'] for item in v36_audit['conditions']):.3e}`.

The cue manipulation therefore changed the training distribution without
changing the target conditional or removing held-out pair support. It did not
isolate singleton mutual information from all other distributional geometry,
so the causal object is the registered cue-bearing distribution, not the
scalar index alone. Table 1 reports the frozen distribution audits; the
construction and deterministic root-solving protocol are specified in
Supplementary Methods.

## Exploratory discovery experiment

Version 3.5 used the rule `(45,36)`, five paired initialization seeds, and four
ordered `m=64` cue doses plus two matched-index geometry controls. Each run used
the same approximately 2M-parameter parent-exposed Transformer, exact
soft-label objective, and 10M-exposure schedule. Cue-bearing conditions used
their registered sampler through 5M and uniform full support thereafter.

Verified onset occurred in {c0_onset[0]}/{c0_onset[1]} C0,
{c1_onset[0]}/{c1_onset[1]} C1, {c2_onset[0]}/{c2_onset[1]} C2, and
{c3_onset[0]}/{c3_onset[1]} C3 seeds. The exploratory exact Page-style test
gave `p = {v35_test['exact_one_sided_p']:.16g}` over
`{v35_test['exact_permutation_count']:,}` assignments. These data established
the frozen dose levels and transition definitions; they are discovery evidence
and were not reused as confirmatory observations.

![Figure 2. v3.5 exploratory trajectories.](figures/fig02_v35_dose_trajectories.png)

*Figure 2.* Seed-level v3.5 R trajectories and frozen onset ordinals. Thin
lines show seeds. Bands show the mean of registered row-bootstrap interval
endpoints and are not a new group-level confidence interval.

Table 2 reports the exploratory onset and establishment outcomes.

## Independent confirmatory replication

Version 3.6 crossed two independently generated rules,
`({v36_audit['rules'][0]['coefficients'][0]},
{v36_audit['rules'][0]['coefficients'][1]})` and
`({v36_audit['rules'][1]['coefficients'][0]},
{v36_audit['rules'][1]['coefficients'][1]})`, with two new cue permutations
and two new initialization seeds. Within each of the eight blocks, R0, R1, and
R2 shared initial weights and every non-condition RNG/configuration field.
The targets were `C_single = 0`, `0.016`, and `0.065092`. All 24 trajectories
were stopped at 10M.

The onset counts were {onset_counts['R0'][0]}/{onset_counts['R0'][1]},
{onset_counts['R1'][0]}/{onset_counts['R1'][1]}, and
{onset_counts['R2'][0]}/{onset_counts['R2'][1]}; establishment counts by 10M
were {establishment_counts['R0'][0]}/{establishment_counts['R0'][1]},
{establishment_counts['R1'][0]}/{establishment_counts['R1'][1]}, and
{establishment_counts['R2'][0]}/{establishment_counts['R2'][1]}.
The primary Page-style statistic was `{primary['observed_statistic']}`.
Exactly {primary['upper_tail_count']} of
{primary['exact_permutation_count']:,} within-block assignments were at least
as large, yielding one-sided
`p = {primary['exact_one_sided_p']:.16g}`. The result crossed the frozen 0.05
confirmatory threshold.

![Figure 3. v3.6 paired onset ordinals.](figures/fig03_v36_paired_onset.png)

*Figure 3.* Paired terminal-coded onset ordinals for all eight new blocks.
Lines preserve rule, permutation, and initialization pairing.

![Figure 4. Exact confirmatory null.](figures/fig04_v36_exact_null.png)

*Figure 4.* Exact null distribution reconstructed from the frozen v3.6 block
ranks. The primary test contains v3.6 only.

Tables 3 and 4 report the confirmatory outcomes and the primary and stratified
exact tests. The transition and terminal-coding rules are specified in
Supplementary Methods.

Mean paired R differences at 5M pre-switch and 10M were
`{ff(paired['R1_vs_R0']['R_5M_pre_delta']['mean'])}` and
`{ff(paired['R1_vs_R0']['R_10M_delta']['mean'])}` for R1-R0, and
`{ff(paired['R2_vs_R1']['R_5M_pre_delta']['mean'])}` and
`{ff(paired['R2_vs_R1']['R_10M_delta']['mean'])}` for R2-R1. The adjacent
ordering was not strict in every block; the inference rests on the frozen
blocked test rather than visual curve separation.

Table 5 reports all frozen paired R differences.

## Representation persistence after cue removal

At exactly 5M, nonzero-cue runs were evaluated, only the sampler was switched
to uniform full support, and a zero-optimizer-step evaluation verified
identical model output and unchanged model, optimizer, scheduler, exposure,
and RNG state. R0 remained uniform throughout.

Five R1 and eight R2 cells were threshold-positive before the switch. All
{retained_count} remained positive at both 6M and 10M. Thus continued exposure
to the cue-bearing distribution was not required once the predictive
representation had formed. This supports learning-path preconditioning rather
than a permanently required shortcut.

![Figure 5. Persistence after cue removal.](figures/fig05_v36_persistence.png)

*Figure 5.* Individual v3.6 threshold-positive cells at 5M, 6M, and 10M.
Whiskers are the registered first-parent-row bootstrap intervals.

Table 6 reports the cue-removal persistence audit.

## Rule and cue-geometry heterogeneity

The confirmatory ordering was significant within each coefficient rule and
within each cue permutation, but the mid-dose response was heterogeneous.
R1 onset occurred in {r1_rule0[0]}/{r1_rule0[1]} rule-0 blocks and
{r1_rule1[0]}/{r1_rule1[1]} rule-1 blocks. R2 onset occurred in
{r2_rule0[0]}/{r2_rule0[1]} and {r2_rule1[0]}/{r2_rule1[1]}, respectively.
The high dose was therefore robust across the tested rules, whereas the
mid-dose probability or timing depended more strongly on the coefficient rule.

![Figure 6. Rule-specific confirmatory outcomes.](figures/fig06_v36_rule_heterogeneity.png)

*Figure 6.* R1 shows rule-dependent heterogeneity; R2 forms in every tested
block. This is a bounded statement about two preregistered new rules.

The v3.5 matched-index controls further showed that equal `C_single` values
were not interchangeable. Relative to `m=64`, the concentrated `m=1`
condition changed onset ordinal by a paired mean of
`{ff(geometry['geometry_concentrated']['onset_terminal_ordinal_delta']['mean'])}`,
whereas the diffuse `m=128` condition changed it by
`{ff(geometry['geometry_diffuse']['onset_terminal_ordinal_delta']['mean'])}`.
This geometry sensitivity is why the manuscript does not attribute the effect
to singleton mutual information alone.

![Figure 7. v3.5 matched-index geometry.](figures/fig07_v35_geometry_comparison.png)

*Figure 7.* Exploratory matched-`C_single` conditions differed in onset and
intermediate R despite sharing the registered scalar index.

## Discussion

The central result is a causal property of a controlled training intervention:
temporary cue-bearing full-support distributions changed whether or when the
frozen Transformer formed a high-order modular center representation. The cue
did not change the target conditional, marginal target entropy, irreducible
noise, total pairwise predictable information, model architecture, optimizer,
or objective. Pairing initial weights and non-condition randomness made onset
timing the prespecified comparison.

The persistence result narrows the mechanism. Because threshold-positive
models retained the representation after returning to uniform full support,
the cue was not merely a predictive feature required at inference or a
permanent sampling crutch. Instead, the early distribution placed optimization
on a learning path from which the high-order representation remained useful
under the pure-synergy distribution.

At the same time, the rule and geometry results reject an overly scalar
interpretation. The registered cue index ordered selected distributions, but
equal index values could have different effects, and the R1 response varied
between coefficient rules. Distribution geometry, gradient organization, and
model state may all contribute. The present experiments identify the
registered distribution intervention as causal; they do not identify a unique
information-theoretic mediator.

### Possible mechanism: symmetry breaking and optimization accessibility

The evidence suggests a possible optimization-accessibility account:
pure synergy creates a symmetry under which singleton-aligned gradients vanish
in expectation. The parents are individually uninformative, and the symmetric
full-support average cancels their singleton-aligned gradient directions.
Transient cue distributions introduce lower-order statistical asymmetries
while preserving the target conditional and Bayes information budget. Those
asymmetries may redirect early optimization toward a state in which the two
parents can be composed; once that task-sufficient representation forms, the
cue is no longer required.

Three observations are consistent with this account: onset was ordered by
registered cue dose, threshold-positive representations persisted after cue
removal, and the frozen gradient audits showed changes in gradient coherence
around observed transitions. This is mechanistic interpretation, not a causal
mediation result. **Gradient statistics are correlates of transition dynamics,
not identified causal mediators.** The data do not establish a universal
loss-landscape mechanism or a unique circuit.

### Practical implication: distributional bootstrapping

The result suggests a possible design principle: temporary training
distributions can be used to improve access to otherwise hard-to-reach
compositional representations when the final task distribution contains
high-order dependencies. This is a hypothesis for bounded, auditable
interventions—not a claim that all AI training should use low-order cues.

## Limitations

First, model scale is limited to one approximately 2M-parameter Transformer
family; whether the effect strengthens, weakens, or changes form with scale
remains open. Second, the parent-exposed interface intentionally controls
variable identification, so recurrent context discovery and variable
selection remain future work. Third, exact soft-label training isolates
representation formation from sampled-label variance; extension to
sampled-token training remains future work. Fourth, finite-field modular
arithmetic is a controlled model organism rather than a proxy for the full
range of natural-data learning. Fifth, the experiments identify a distribution
intervention but do not establish a universal loss-landscape mechanism.

Additional design limits remain. The study uses one field size (`p={p}`), one
noise level (`eta={eta:g}`), a finite set of cue constructions, and eight
confirmatory blocks. The checkpoint grid limits onset timing resolution, and
terminal coding cannot verify a first pass at 10M. Row-bootstrap intervals
quantify evaluation variation over parent rows, not between-rule population
uncertainty. The exact test is valid for the frozen randomization structure but
does not turn the tested rules into a population sample of all modular tasks.
The result is not a universal Transformer property, is not a thermodynamic
phase transition, does not show that the recurrent production task is solved,
and is excluded from the original scaling fit.

## Reproducibility statement

The final v3.6 commit is tagged `kepler-v3.6-confirm1`. Every manuscript number,
figure, and table is generated from frozen machine-readable JSON whose hashes
are recorded in `artifact_index.json` and
`../artifacts/v3_6_confirmatory_freeze_record.json`. The lightweight verifier
recomputes distribution invariants, onset and establishment coding, the exact
{primary['exact_permutation_count']:,}-assignment primary test, and all paper
outputs without loading a training loop or taking an optimizer step.

Tables 7 and 8 summarize the protocol-integrity audit and claim boundaries.

## Data and code availability

Source code, frozen manifests, frozen result JSON, figure source data, table
exports, and verification scripts are contained in this repository. Raw model
checkpoints are referenced by SHA-256 sidecars and are not required for the
lightweight result-verification path. Public archival identifiers can be added
without changing the frozen scientific artifacts.

## Supplementary methods

### Task objective, evaluation, and probes

Exact soft-label cross-entropy was used throughout the parent-exposed
experiments: at each observed pair, the target was the analytic noisy
conditional rather than one sampled label. Evaluation enumerated all
`{p**2:,}` ordered parent pairs. Every checkpoint recorded exact conditional
KL, normalized R, center accuracy, center probability, predictive entropy, and
a 2,000-replicate first-parent-row bootstrap interval. Selected checkpoints
included layer-wise linear, fixed-Fourier, and fixed-capacity MLP probes.

### Cue-family construction and audit

The frozen cue permutations and coefficients reweighted the complete pair
space while preserving strictly positive probability for every pair. Lambda
was solved by deterministic root finding for each preregistered target, not
selected from model outcomes. Exhaustive enumeration verified marginal
uniformity, conditional invariance, joint information, and full support.

### Model and optimization

All paper experiments used the frozen v3.5 parent-exposed Transformer with
manifest SHA-256 `{v36['manifest']['sha256']}`. The run manifest records
{registered_run['actual_core_parameters']:,} core parameters, width
{registered_run['width']}, batch size {registered_run['batch_size']:,}, exact
soft-label cross-entropy, AdamW learning rate
`{optimizer['learning_rate']}`, betas
`({optimizer['betas'][0]},{optimizer['betas'][1]})`, weight decay
`{optimizer['weight_decay']}`, gradient clipping at
`{optimizer['grad_clip_norm']}`, {optimizer['warmup_tokens'] / 1e6:g}M warmup
exposures, and an
absolute-exposure cosine schedule. No optimizer or scheduler state was reset at
cue removal.

### Transition coding

Probe checkpoints were 2M, 3M, 4M, 5M, 6M, 7.5M, and 10M. Onset required both
the R and layer-6 linear-probe criteria at one checkpoint and their retention
at the next. Establishment was a single-checkpoint conjunction. Coding was
implemented before confirmatory outcomes were observed and was not changed.

### Exact test

Within each v3.6 block, terminal onset ordinals were converted to midranks with
earlier onset receiving higher rank. Ranks were represented as doubled
integers, preserving ties exactly. The weighted Page-style statistic used
registered condition order R0, R1, R2. All six label assignments per block
were combined across eight blocks, giving `6^8 =
{primary['exact_permutation_count']:,}` exact assignments. The one-sided p-value
was the fraction with statistic at least as large as observed.

### Artifact integrity

The confirmatory freeze contains every Git tag, every manifest hash, v3.5 and
v3.6 result hashes, the complete environment lock, test definition and result,
and raw artifact counts. The audit validated 330 v3.5 and 264 v3.6 checkpoint
sidecars. The manuscript revision changes only files under `paper/`; prior
manifests, results, checkpoints, the confirmatory freeze record, and statistical
definitions remain unchanged.

## References

1. Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. (2022).
   “Grokking: Generalization Beyond Overfitting on Small Algorithmic
   Datasets.” [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).
2. Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023).
   “Progress Measures for Grokking via Mechanistic Interpretability.”
   [arXiv:2301.05217](https://arxiv.org/abs/2301.05217).
3. Bengio, Y., Louradour, J., Collobert, R., and Weston, J. (2009).
   “Curriculum Learning.” *Proceedings of ICML 2009*.
   [Primary paper](https://icml.cc/2009/papers/119.pdf).
4. Williams, P. L., and Beer, R. D. (2010). “Nonnegative Decomposition of
   Multivariate Information.” [arXiv:1004.2515](https://arxiv.org/abs/1004.2515).
"""

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\title{{{TITLE}}}
\author{{Project Kepler Collaboration}}
\date{{}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We study whether training-distribution geometry can alter formation of a
high-order predictive representation while the target conditional,
irreducible noise, marginals, and total predictable information remain fixed.
In a finite-field parent-exposed task, two uniform parents determine a modular
center and a fixed additive-noise channel determines the target. Version 3.5
provided exploratory evidence that transient full-support low-order cues
advanced predictive-state formation. An independent v3.6 replication used two
new rules, two new cue permutations, two initialization seeds, and three cue
doses. Verified onset occurred in {onset_counts['R0'][0]}/8,
{onset_counts['R1'][0]}/8, and {onset_counts['R2'][0]}/8 blocks. The frozen
one-sided exact blocked Page-style test enumerated
{primary['exact_permutation_count']:,} assignments and gave
$p={primary['exact_one_sided_p']:.8g}$. All {retained_count} nonzero-cue cells
positive at cue removal retained the representation threshold at 6M and 10M.
Temporary cue-bearing distributions therefore causally changed onset timing or
probability within the frozen task and model family. The experiment does not
identify singleton mutual information alone as causal and does not establish a
universal Transformer property.
\end{{abstract}}

\section{{Introduction}}
The central question is: \emph{{How does training-distribution geometry affect
the accessibility of task-sufficient representations when the final task
information is fixed?}} A representation may be statistically available yet
difficult to discover because useful optimization directions are weak, cancel
across examples, or become accessible only after other features organize. We
therefore intervene on early training-distribution geometry while holding the
target conditional and information budget fixed.

Project Kepler uses a finite-field mechanism assay motivated by, but distinct
from, its recurrent production task. The parent-exposed design intentionally
separates (1) identifying relevant variables, which the interface controls by
presenting both parents directly, from (2) composing discovered variables into
a predictive representation, which remains the learning problem. The present
paper does not claim that the recurrent task has been solved.

Version 3.5 is exploratory discovery; version 3.6 is an independent
confirmatory replication. The discovery data are excluded from the primary
confirmatory $p$-value. The aim is neither a new scaling law nor a universal
intelligence principle; it is a bounded causal test within one controlled task
and model family. Figure~\ref{{fig:task}} summarizes the construction and cue
schedule.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig01_task_cue_schematic.pdf}}
\caption{{Controlled finite-field task and transient cue schedule. All values
are generated from frozen JSON.}}
\label{{fig:task}}
\end{{figure}}

\section{{Related Work}}
\paragraph{{Grokking and delayed generalization.}}
Modular-arithmetic grokking studies established that small neural networks can
generalize long after achieving near-perfect training performance under a
fixed training distribution \cite{{power2022grokking}}. This paper does not
offer a universal explanation of grokking. It instead asks whether the
geometry of the \emph{{early}} training distribution, with the final
conditional and information budget held fixed, changes the accessibility or
timing of a delayed representational transition.

\paragraph{{Mechanistic interpretability of modular arithmetic.}}
Mechanistic analyses of modular addition have recovered Fourier-like internal
representations and circuits implementing trigonometric structure
\cite{{nanda2023progress}}. Here, fixed-Fourier probes are diagnostics
alongside linear and fixed-capacity MLP probes. We do not assume that a
specific Fourier circuit must be learned, and the causal result concerns the
timing or probability of task-sufficient representation formation rather than
identification of a unique circuit.

\paragraph{{Curriculum learning.}}
Classic curriculum learning changes example order or introduces examples and
concepts by increasing difficulty \cite{{bengio2009curriculum}}. The present
intervention instead keeps the target conditional and Bayes information
budget fixed and changes only the geometry of the full-support training
distribution during an early window. We call this narrower intervention
\emph{{distributional bootstrapping of representation formation}}.

\paragraph{{Partial Information Decomposition.}}
Partial Information Decomposition separates redundant, unique, and
synergistic contributions in multivariate information
\cite{{williams2010pid}}. The full-support base task is a pure-synergy
construction in the limited operational sense needed here: singleton
information is zero while joint information is nonzero. We do not attempt a
full PID decomposition of the cue distributions or learned representations.

\section{{Controlled finite-field task}}
The field size is $p={p}$. The predictive center is
$z=a_1x_1+a_2x_2 \bmod p$, and $y=z+\epsilon \bmod p$ with
$P(\epsilon=0)=1-\eta$, $P(\epsilon=r)=\eta/(p-1)$ for $r\ne0$, and
$\eta={eta:g}$. Thus $H(Y\mid X_1,X_2)={ff(noise_entropy)}$ nats, and the
target conditional is unchanged across conditions. The primary metric is
\[
R=\frac{{\mathbb{{E}}_h D_{{\rm KL}}(q_h\Vert p_\theta)}}
{{\log p-H(\nu_\eta)}}.
\]
A uniform predictor has $R=1$ and the oracle has $R=0$. Objective,
evaluation, bootstrap, and probe details appear in Supplementary Methods.

\section{{Pure-synergy information structure}}
Under uniform full support, $X_1,X_2,Z,Y$ are uniform,
$I(Y;X_1)=I(Y;X_2)=0$, and
$I(Y;X_1,X_2)={ff(joint_information)}$ nats. The scalar cue index orders
selected constructed distributions; it is not a unique-information percentage.

\section{{Full-support cue-family construction}}
The cue family reweights ordered parent pairs through frozen permutations while
retaining strictly positive support. Exhaustive enumeration verifies uniform
marginals, invariant $H(Y\mid X_1,X_2)$, invariant joint information, and exact
uniform full support at zero cue. The intervention is the registered
distribution, not singleton mutual information in isolation.
Table~\ref{{tab:table01-distribution-invariants}} reports the frozen audits;
construction details appear in Supplementary Methods.

\input{{tables/table01_distribution_invariants.tex}}

\section{{Exploratory discovery experiment}}
Version 3.5 used rule $(45,36)$, five paired seeds, four ordered $m=64$ doses,
and two matched-index geometry controls. Verified onset occurred in
{c0_onset[0]}/5 C0, {c1_onset[0]}/5 C1, {c2_onset[0]}/5 C2, and
{c3_onset[0]}/5 C3 seeds. The exploratory exact test gave
$p={v35_test['exact_one_sided_p']:.8g}$. These observations selected frozen
confirmatory conditions and definitions; they are not confirmatory data.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig02_v35_dose_trajectories.pdf}}
\caption{{Exploratory v3.5 R trajectories and onset outcomes. Bands are means
of registered row-bootstrap endpoints, not new group-level intervals.}}
\label{{fig:v35-trajectories}}
\end{{figure}}

\input{{tables/table02_v35_exploratory_outcomes.tex}}
Figure~\ref{{fig:v35-trajectories}} and
Table~\ref{{tab:table02-v35-exploratory-outcomes}} report the discovery
trajectories and frozen outcomes.

\section{{Independent confirmatory replication}}
Version 3.6 crossed two new rules with two new cue permutations and two new
initialization seeds, yielding eight paired blocks and 24 trajectories. R0,
R1, and R2 shared initial weights and all non-condition random fields within
each block. Verified onset occurred in {onset_counts['R0'][0]}/8,
{onset_counts['R1'][0]}/8, and {onset_counts['R2'][0]}/8 blocks. The observed
Page-style statistic was {primary['observed_statistic']}; exactly
{primary['upper_tail_count']} of {primary['exact_permutation_count']:,}
assignments were as large, giving one-sided
$p={primary['exact_one_sided_p']:.8g}$.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig03_v36_paired_onset.pdf}}
\caption{{Paired terminal-coded onset ordinals for all eight confirmatory
blocks.}}
\label{{fig:v36-onset}}
\end{{figure}}

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig04_v36_exact_null.pdf}}
\caption{{Exact v3.6 permutation null. Version 3.5 is excluded.}}
\label{{fig:exact-null}}
\end{{figure}}

\input{{tables/table03_v36_confirmatory_outcomes.tex}}
\input{{tables/table04_exact_tests.tex}}
\input{{tables/table05_paired_R_differences.tex}}
Figures~\ref{{fig:v36-onset}} and \ref{{fig:exact-null}} show the paired
onset ordinals and exact null. Tables~\ref{{tab:table03-v36-confirmatory-outcomes}},
\ref{{tab:table04-exact-tests}}, and
\ref{{tab:table05-paired-R-differences}} report confirmatory outcomes, exact
tests, and paired differences. The transition-coding rule is specified in
Supplementary Methods.

\section{{Representation persistence after cue removal}}
At 5M only the sampler changed for nonzero-cue runs. Zero-step checks verified
unchanged outputs and model, optimizer, scheduler, exposure, and RNG state.
All {retained_count} threshold-positive R1/R2 cells remained positive at 6M
and 10M, supporting learning-path preconditioning rather than a permanently
required shortcut.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig05_v36_persistence.pdf}}
\caption{{Persistence after return to uniform full support. Whiskers are
registered first-parent-row bootstrap intervals.}}
\label{{fig:persistence}}
\end{{figure}}

\input{{tables/table06_cue_removal_persistence.tex}}
Figure~\ref{{fig:persistence}} and
Table~\ref{{tab:table06-cue-removal-persistence}} report the persistence
result and zero-step switch audit.

\section{{Rule and cue-geometry heterogeneity}}
R1 onset occurred in {r1_rule0[0]}/4 rule-0 and {r1_rule1[0]}/4 rule-1
blocks, while R2 onset occurred in {r2_rule0[0]}/4 and {r2_rule1[0]}/4.
The ordered effect spans both rules but the mid-dose response is
rule-dependent. In exploratory matched-index controls, concentrated and
diffuse geometries differed from $m=64$, so equal cue-index values are not
interchangeable.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig06_v36_rule_heterogeneity.pdf}}
\caption{{Rule-specific R1 heterogeneity and R2 robustness.}}
\label{{fig:rule-heterogeneity}}
\end{{figure}}

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig07_v35_geometry_comparison.pdf}}
\caption{{Exploratory matched-index geometry comparison.}}
\label{{fig:geometry}}
\end{{figure}}
Figures~\ref{{fig:rule-heterogeneity}} and \ref{{fig:geometry}} separate the
confirmatory rule comparison from the exploratory geometry comparison.

\section{{Discussion}}
Temporary cue-bearing full-support distributions changed whether or when the
frozen Transformer formed the modular predictive center. The target
conditional, marginals, irreducible noise, total pairwise predictable
information, architecture, optimizer, and objective were held fixed.
Persistence after cue removal indicates that the early distribution altered
the learning path rather than supplying a permanently required feature.
Heterogeneity across rules and geometries rejects an explanation based on the
scalar cue index alone.

\subsection{{Possible mechanism: symmetry breaking and optimization accessibility}}
The evidence suggests a possible optimization-accessibility account: pure
synergy creates a symmetry under which singleton-aligned gradients vanish in
expectation. The parents are individually uninformative, and the symmetric
full-support average cancels their singleton-aligned gradient directions.
Transient cue distributions introduce lower-order statistical
asymmetries while preserving the target conditional and Bayes information
budget. These asymmetries may redirect early optimization toward a state in
which the two parents can be composed; after the task-sufficient
representation forms, the cue is no longer required.

Three observations are consistent with this account: onset was ordered by
registered cue dose, threshold-positive representations persisted after cue
removal, and the frozen gradient audits showed changes in gradient coherence
around observed transitions. This is mechanistic interpretation, not a causal
mediation result. \emph{{Gradient statistics are correlates of transition
dynamics, not identified causal mediators.}} The data do not establish a
universal loss-landscape mechanism or a unique circuit.

\subsection{{Practical implication: distributional bootstrapping}}
The result suggests a possible design principle: temporary training
distributions can be used to improve access to otherwise hard-to-reach
compositional representations when the final task distribution contains
high-order dependencies. This is a hypothesis for bounded, auditable
interventions, not a claim that all AI training should use low-order cues.

\section{{Limitations}}
First, model scale is limited to one approximately 2M-parameter Transformer
family; scale dependence remains open. Second, the parent-exposed interface
intentionally controls variable identification; recurrent context discovery
and variable selection remain future work. Third, exact soft-label training
isolates representation formation from sampled-label variance; sampled-token
training remains future work. Fourth, finite-field modular arithmetic is a
controlled model organism rather than a proxy for the full range of
natural-data learning. Fifth, the experiments identify a distribution
intervention but do not establish a universal loss-landscape mechanism.

The study also uses one field size, one noise level, a finite checkpoint grid,
and eight confirmatory blocks. The exact test is valid for the frozen
randomization structure but does not make the tested rules a population sample
of all modular tasks. The result is not a universal Transformer claim, not a
thermodynamic phase transition, and not evidence that the recurrent production
task is solved. These results are excluded from the original scaling fit.

\section{{Reproducibility statement}}
The final result is tagged \texttt{{kepler-v3.6-confirm1}}. Frozen hashes,
environment, test definitions, counts, and provenance are indexed in the
package. The lightweight verifier recomputes distribution audits, transition
coding, all {primary['exact_permutation_count']:,} exact assignments, figures,
and tables without optimizer steps.

\input{{tables/table07_protocol_integrity.tex}}
\input{{tables/table08_claim_boundaries.tex}}
Tables~\ref{{tab:table07-protocol-integrity}} and
\ref{{tab:table08-claim-boundaries}} summarize protocol integrity and claim
boundaries.

\section{{Data and code availability}}
Code, manifests, result JSON, source data, and verification scripts are
included in the repository. Checkpoints are referenced by SHA-256 sidecars and
are unnecessary for lightweight verification.

\section{{Supplementary methods}}
\subsection{{Task objective, evaluation, and probes}}
Exact soft-label cross-entropy was used throughout the parent-exposed
experiments: at each observed pair, the target was the analytic noisy
conditional rather than one sampled label. Evaluation enumerated all
{p**2:,} ordered parent pairs. Every checkpoint recorded exact conditional KL,
normalized $R$, center accuracy, center probability, predictive entropy, and a
2,000-replicate first-parent-row bootstrap interval. Selected checkpoints
included layer-wise linear, fixed-Fourier, and fixed-capacity MLP probes.

\subsection{{Cue-family construction and audit}}
Frozen cue permutations and coefficients reweighted the complete pair space
while preserving strictly positive probability for every pair. Lambda was
solved by deterministic root finding for each preregistered target, not
selected from model outcomes. Exhaustive enumeration verified marginal
uniformity, conditional invariance, joint information, and full support.

\subsection{{Model and optimization}}
The manifest records {registered_run['actual_core_parameters']:,} core
parameters, width {registered_run['width']}, batch size
{registered_run['batch_size']:,}, exact soft-label cross-entropy, AdamW
learning rate ${learning_rate_mantissa:g}\times10^{{{learning_rate_exponent}}}$,
betas
$({optimizer['betas'][0]},{optimizer['betas'][1]})$, weight decay
{optimizer['weight_decay']}, clipping at {optimizer['grad_clip_norm']},
{optimizer['warmup_tokens'] / 1e6:g}M warmup exposures, and an
absolute-exposure cosine schedule.

\subsection{{Transition coding}}
Probe checkpoints were 2M, 3M, 4M, 5M, 6M, 7.5M, and 10M. Onset required
$R<0.9$ and layer-6 linear accuracy above 0.5, retained at the next checkpoint.
Establishment required $R<0.1$, center accuracy above 0.9, and layer-6 linear
accuracy above 0.9.

\subsection{{Exact test}}
Within-block terminal onset ordinals were converted to doubled midranks, with
earlier onset ranked higher. All six label assignments were combined across
eight blocks, giving $6^8={primary['exact_permutation_count']:,}$ assignments.

\subsection{{Artifact integrity}}
The freeze records every tag, manifest and result hash, the environment lock,
test definition and result, and artifact counts. This manuscript revision
changes only the \texttt{{paper/}} tree; no prior scientific artifact changes.

\begin{{thebibliography}}{{9}}
\bibitem{{power2022grokking}}
A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra.
\newblock Grokking: Generalization beyond overfitting on small algorithmic
datasets.
\newblock \emph{{arXiv:2201.02177}}, 2022.
\newblock \url{{https://arxiv.org/abs/2201.02177}}.

\bibitem{{nanda2023progress}}
N. Nanda, L. Chan, T. Lieberum, J. Smith, and J. Steinhardt.
\newblock Progress measures for grokking via mechanistic interpretability.
\newblock \emph{{arXiv:2301.05217}}, 2023.
\newblock \url{{https://arxiv.org/abs/2301.05217}}.

\bibitem{{bengio2009curriculum}}
Y. Bengio, J. Louradour, R. Collobert, and J. Weston.
\newblock Curriculum learning.
\newblock In \emph{{Proceedings of the 26th International Conference on
Machine Learning}}, 2009.
\newblock \url{{https://icml.cc/2009/papers/119.pdf}}.

\bibitem{{williams2010pid}}
P. L. Williams and R. D. Beer.
\newblock Nonnegative decomposition of multivariate information.
\newblock \emph{{arXiv:1004.2515}}, 2010.
\newblock \url{{https://arxiv.org/abs/1004.2515}}.
\end{{thebibliography}}

\end{{document}}
"""

    atomic_write_text(PAPER / "manuscript.md", md)
    atomic_write_text(PAPER / "manuscript.tex", tex)
    print(PAPER / "manuscript.md")
    print(PAPER / "manuscript.tex")


if __name__ == "__main__":
    main()
