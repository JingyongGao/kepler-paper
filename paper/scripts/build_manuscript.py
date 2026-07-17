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

Scaling laws summarize average changes in loss with model and data size, but
they do not by themselves explain when a model forms a task-sufficient internal
state. A representation may be statistically available yet difficult to
discover because useful gradient directions are weak, cancel across examples,
or emerge only after other features have organized. This motivates controlled
experiments in which the target conditional distribution and its information
budget are held fixed while the geometry of the training distribution is
changed.

Project Kepler isolates this question in a finite field. Earlier recurrent
experiments motivated a parent-exposed mechanism assay, but the present paper
does not claim that the recurrent production task has been solved. Instead, it
asks a narrower causal question: can a temporary low-order cue in a
full-support sampling distribution alter when a Transformer forms the
high-order predictive center state?

The experimental sequence was explicitly separated into discovery and
confirmation. Version 3.5 was exploratory and selected the cue doses and
frozen outcome definitions. Version 3.6 used new rules, new cue permutations,
new initialization seeds, and no v3.5 checkpoints. The v3.5 data are not
included in the v3.6 primary p-value.

![Figure 1. Controlled task and cue schedule.](figures/fig01_task_cue_schematic.png)

*Figure 1.* The parent pair determines a modular center; fixed additive noise
determines the target. Cue-bearing sampling is used only from 0--5M exposures,
after which training returns to uniform full support. All figure values are
read from frozen JSON.

## Controlled finite-field task

The field size was `p = {p}`. For coefficients `a1,a2 != 0`, the predictive
center was

`z = a1*x1 + a2*x2 mod p`.

The target followed `y = z + epsilon mod p`, with
`P(epsilon=0)=1-eta`, `P(epsilon=r)=eta/(p-1)` for nonzero `r`, and
`eta = {eta:g}`. Consequently,
`H(Y|X1,X2) = H(nu_eta) = {ff(noise_entropy)} nats` and the target conditional
was unchanged across cue conditions. Exact soft-label cross-entropy was used
throughout the parent-exposed experiments, so the objective at each observed
pair was the cross-entropy against the analytic noisy conditional rather than a
single sampled label.

Evaluation enumerated all `{p**2:,}` ordered parent pairs. The primary error
metric was

`R = E_h[KL(q_h || p_theta)] / (log(p) - H(nu_eta))`.

A uniform predictor has `R=1` and the analytic oracle has `R=0`. Every
checkpoint also recorded center accuracy, probability assigned to the center,
predictive entropy, and a 2,000-replicate first-parent-row bootstrap interval.
Selected checkpoints included layer-wise linear, fixed-Fourier, and
fixed-capacity MLP probes.

## Pure-synergy information structure

With uniformly sampled independent parents and nonzero modular coefficients,
`X1`, `X2`, `Z`, and `Y` are uniform. Each individual parent is independent of
the target: `I(Y;X1)=I(Y;X2)=0`. Jointly, however, the pair determines the
noisy center and carries
`I(Y;X1,X2)=log(p)-H(nu_eta)={ff(joint_information)} nats`.

This pure-synergy baseline makes the discovery problem unusually clean. The
model must compose both parents to predict the center, and no marginal
frequency cue points to the answer. It also makes clear why the registered
cue index must not be described as a unique-information percentage: it is a
scalar property of the constructed sampling family, not a decomposition of
all predictive information.

## Full-support cue-family construction

The cue family reweighted ordered parent pairs through a frozen permutation and
cue coefficient while keeping every pair probability strictly positive.
Lambda was solved by deterministic root finding for each registered target.
Exhaustive audits confirmed uniform `X1`, `X2`, `Z`, and `Y`, invariant
`H(Y|X1,X2)`, invariant `I(Y;X1,X2)`, and exact uniform full support at zero
cue. In v3.6 the maximum marginal-uniformity error was
`{max(max(item['uniformity_max_absolute_error'].values()) for item in v36_audit['conditions']):.3e}`
and the minimum pair probability was
`{min(item['minimum_pair_probability'] for item in v36_audit['conditions']):.3e}`.

The cue manipulation therefore changed the training distribution without
changing the target conditional or removing held-out pair support. It did not
isolate singleton mutual information from all other distributional geometry,
so the causal object is the registered cue-bearing distribution, not the scalar
index alone.

[Table 1: Distribution invariants](tables/table01_distribution_invariants.md)

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

[Table 2: v3.5 exploratory outcomes](tables/table02_v35_exploratory_outcomes.md)

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

Onset was frozen as the first probe checkpoint with `R<0.9` and layer-6
linear-probe accuracy above 0.5, with both retained at the next registered
probe checkpoint. No onset received terminal ordinal 8; a first pass at 10M
could not initiate onset because no later checkpoint existed. Establishment
required `R<0.1`, center accuracy above 0.9, and layer-6 linear accuracy above
0.9 at one checkpoint.

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

[Table 3: v3.6 confirmatory outcomes](tables/table03_v36_confirmatory_outcomes.md)

[Table 4: Primary and stratified exact tests](tables/table04_exact_tests.md)

Mean paired R differences at 5M pre-switch and 10M were
`{ff(paired['R1_vs_R0']['R_5M_pre_delta']['mean'])}` and
`{ff(paired['R1_vs_R0']['R_10M_delta']['mean'])}` for R1-R0, and
`{ff(paired['R2_vs_R1']['R_5M_pre_delta']['mean'])}` and
`{ff(paired['R2_vs_R1']['R_10M_delta']['mean'])}` for R2-R1. The adjacent
ordering was not strict in every block; the inference rests on the frozen
blocked test rather than visual curve separation.

[Table 5: Paired R differences](tables/table05_paired_R_differences.md)

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

[Table 6: Cue-removal persistence](tables/table06_cue_removal_persistence.md)

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

## Limitations

The experiments use one field size (`p={p}`), one noise level
(`eta={eta:g}`), one approximately 2M-parameter parent-exposed Transformer
family, exact soft-label training, and a finite set of cue constructions. The
checkpoint grid limits onset timing resolution, and terminal coding cannot
verify a first pass at 10M. Row-bootstrap intervals quantify evaluation
variation over parent rows, not between-rule population uncertainty.

The confirmatory sample consists of eight paired blocks. The exact test is
valid for the frozen randomization structure but does not turn the tested rules
into a population sample of all modular tasks. The result is not a universal
Transformer property, is not a thermodynamic phase transition, and does not
show that the recurrent production task is solved. These mechanism experiments
are excluded from the original scaling fit.

## Reproducibility statement

The final v3.6 commit is tagged `kepler-v3.6-confirm1`. Every manuscript number,
figure, and table is generated from frozen machine-readable JSON whose hashes
are recorded in `artifact_index.json` and
`../artifacts/v3_6_confirmatory_freeze_record.json`. The lightweight verifier
recomputes distribution invariants, onset and establishment coding, the exact
{primary['exact_permutation_count']:,}-assignment primary test, and all paper
outputs without loading a training loop or taking an optimizer step.

[Table 7: Protocol and integrity audit](tables/table07_protocol_integrity.md)

[Table 8: Claim boundaries](tables/table08_claim_boundaries.md)

## Data and code availability

Source code, frozen manifests, frozen result JSON, figure source data, table
exports, and verification scripts are contained in this repository. Raw model
checkpoints are referenced by SHA-256 sidecars and are not required for the
lightweight result-verification path. Public archival identifiers can be added
without changing the frozen scientific artifacts.

## Supplementary methods

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
sidecars. Paper packaging changes only files under `paper/` plus the new
confirmatory freeze record; prior manifests, results, checkpoints, and
statistical definitions remain unchanged.
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
Scaling curves do not specify when a model forms a task-sufficient internal
state. We therefore manipulate training-distribution geometry while holding
the target conditional and information budget fixed. Project Kepler uses a
finite-field mechanism assay motivated by, but distinct from, its recurrent
production task. The present paper does not claim that the recurrent task has
been solved. Version 3.5 is exploratory discovery; version 3.6 is an independent
confirmatory replication. The discovery data are excluded from the primary
confirmatory $p$-value.

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig01_task_cue_schematic.pdf}}
\caption{{Controlled finite-field task and transient cue schedule. All values
are generated from frozen JSON.}}
\label{{fig:task}}
\end{{figure}}

\section{{Controlled finite-field task}}
The field size is $p={p}$. The predictive center is
$z=a_1x_1+a_2x_2 \bmod p$, and $y=z+\epsilon \bmod p$ with
$P(\epsilon=0)=1-\eta$, $P(\epsilon=r)=\eta/(p-1)$ for $r\ne0$, and
$\eta={eta:g}$. Thus $H(Y\mid X_1,X_2)={ff(noise_entropy)}$ nats. Exact
soft-label cross-entropy is used. Evaluation enumerates all {p**2:,} ordered
pairs and reports
\[
R=\frac{{\mathbb{{E}}_h D_{{\rm KL}}(q_h\Vert p_\theta)}}
{{\log p-H(\nu_\eta)}}.
\]
A uniform predictor has $R=1$ and the oracle has $R=0$.

\section{{Pure-synergy information structure}}
Under uniform full support, $X_1,X_2,Z,Y$ are uniform,
$I(Y;X_1)=I(Y;X_2)=0$, and
$I(Y;X_1,X_2)={ff(joint_information)}$ nats. The scalar cue index orders
selected constructed distributions; it is not a unique-information percentage.

\section{{Full-support cue-family construction}}
The cue family reweights ordered parent pairs through frozen permutations while
retaining strictly positive support. Deterministic root solving targets each
registered cue index. Exhaustive enumeration verifies uniform marginals,
invariant $H(Y\mid X_1,X_2)$, invariant joint information, and exact uniform
full support at zero cue. The intervention is the registered distribution, not
singleton mutual information in isolation.

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
\end{{figure}}

\input{{tables/table02_v35_exploratory_outcomes.tex}}

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
\end{{figure}}

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig04_v36_exact_null.pdf}}
\caption{{Exact v3.6 permutation null. Version 3.5 is excluded.}}
\end{{figure}}

\input{{tables/table03_v36_confirmatory_outcomes.tex}}
\input{{tables/table04_exact_tests.tex}}
\input{{tables/table05_paired_R_differences.tex}}

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
\end{{figure}}

\input{{tables/table06_cue_removal_persistence.tex}}

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
\end{{figure}}

\begin{{figure}}[tb]
\centering
\includegraphics[width=\linewidth]{{figures/fig07_v35_geometry_comparison.pdf}}
\caption{{Exploratory matched-index geometry comparison.}}
\end{{figure}}

\section{{Discussion}}
Temporary cue-bearing full-support distributions changed whether or when the
frozen Transformer formed the modular predictive center. The target
conditional, marginals, irreducible noise, total pairwise predictable
information, architecture, optimizer, and objective were held fixed.
Persistence after cue removal indicates that the early distribution altered
the learning path rather than supplying a permanently required feature.
Heterogeneity across rules and geometries rejects an explanation based on the
scalar cue index alone.

\section{{Limitations}}
The experiment uses one field size, one noise level, one approximately 2M
parent-exposed architecture, exact soft labels, a finite checkpoint grid, and
eight confirmatory blocks. It is not a universal Transformer claim, not a
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

\section{{Data and code availability}}
Code, manifests, result JSON, source data, and verification scripts are
included in the repository. Checkpoints are referenced by SHA-256 sidecars and
are unnecessary for lightweight verification.

\section{{Supplementary methods}}
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
test definition and result, and artifact counts. Packaging changes only the
\texttt{{paper/}} tree and the new freeze record; no prior scientific artifact
changes.

\end{{document}}
"""

    atomic_write_text(PAPER / "manuscript.md", md)
    atomic_write_text(PAPER / "manuscript.tex", tex)
    print(PAPER / "manuscript.md")
    print(PAPER / "manuscript.tex")


if __name__ == "__main__":
    main()
