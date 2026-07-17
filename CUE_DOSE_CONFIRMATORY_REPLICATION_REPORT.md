# Project Kepler v3.6 — Confirmatory Cross-Rule Cue-Dose Replication

## Executive result

The independent confirmatory replication is complete. All 24 registered
trajectories reached the 10M pair-exposure stop, all 24 zero-step switch
checks passed, and no unregistered gradient-SNR audit or prohibited
experiment was run.

The preregistered one-sided blocked Page-style test enumerated all
`(3!)^8 = 1,679,616` within-block assignments. The observed statistic was
`110.0`; 16 assignments were in the upper tail, giving
`p = 9.525986892242037e-06`. This is below the frozen `0.05` threshold, so
the replication is statistically confirmatory.

The registered onset order was broadly reproduced across both new
coefficient rules and both new cue permutations. R0 had verified onset in
1/8 blocks by 10M, R1 in 6/8, and R2 in 8/8. Mean terminal-coded onset
ordinals were `7.75`, `4.75`, and `2.625`, respectively. R2 was earlier
than R0 in every block; R1 was earlier than R0 in six blocks and tied it in
two; R2 was earlier than R1 in six blocks and tied it in two.

The mid-dose R1 response was heterogeneous by rule: all four `rule_0`
blocks formed a verified representation, while only two of four `rule_1`
blocks did so by 10M. Nevertheless, the ordered test was significant
within each rule and within each permutation. The result therefore
confirms a cross-rule and cross-permutation change in representation-onset
timing or probability, while retaining evidence of rule-dependent
mid-dose heterogeneity.

All 13 nonzero-cue cells that were threshold-positive at the 5M
pre-switch evaluation remained positive at both 6M and 10M after cue
removal. This is evidence for learning-path preconditioning rather than a
permanently required predictive shortcut.

## Freeze and protocol integrity

- The v3.5 result is frozen at commit
  `8788de3ffedd8ce120691c48c0d863796edd17ca` and tag
  `kepler-v3.5-cuedose1`.
- v3.6 was run on branch `kepler-v3.6-confirmatory-replication`.
- The freeze record is
  [`artifacts/v3_6_freeze_record.json`](artifacts/v3_6_freeze_record.json),
  SHA-256
  `12c31219bda06f181600250cda5e980091a4f12e645d954d646004a09beb7397`.
- The confirmatory manifest is
  [`artifacts/cue_dose_confirmatory_manifest.json`](artifacts/cue_dose_confirmatory_manifest.json),
  SHA-256
  `04ca572e9367ab4216ec4ff1f0528e63f6a3f7b690ae960dc214fd0e4bfadb39`.
- The frozen v3.5 result SHA-256 remains
  `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.
- All seven previously registered manifest hashes were rechecked and are
  unchanged. No previous manifest or result was modified.
- v3.5 observations were not combined into the v3.6 primary p-value.

The two model initialization seeds were `1992599467` and `59227503`.
Within each rule × permutation × initialization block, R0/R1/R2 used the
same initial weights and the same non-condition RNG/configuration fields.
The architecture, exact soft-label objective, AdamW settings, scheduler,
`p=257`, `eta=0.1`, and 10M exposure ceiling remained frozen.

## Independent distribution audit

The exhaustive distribution audit is
[`artifacts/confirmatory_distribution_audit.json`](artifacts/confirmatory_distribution_audit.json),
SHA-256
`f67d261bc6e97fcb9fb7f25e12d7dfdae29fe0567acc930677507d5bf454c3c9`.

The SHA-256-generated rules were:

| Rule | Coefficients `(a1,a2)` | Cue coefficient `c` |
|---|---:|---:|
| `rule_0` | `(205,5)` | `111` |
| `rule_1` | `(123,209)` | `145` |

The rule-generation seed was
`b27ac3dd3035423188193159d1b1e2ad954adef65f3843022a63b8ff7fefff2e`.
The two cue-permutation seeds were
`1f31b4ee7abe61520cbfd1d12ee54908e4cdcc5d6153fdfbcbae64d65b57f642`
and
`25805e63757592cc4a5f60fe9e4b40fb8b100113b6ee6b99b0d41f8e6fcdc262`.
Both rules, both permutations, and all cue coefficients are distinct as
required; neither the v3.5 `(45,36)` rule nor its cue permutation was
reused.

Each rule/permutation/condition cell was enumerated independently before
training. The solved distribution summaries were:

| Condition | Target / achieved `C_single` | Solved lambda | Minimum pair probability |
|---|---:|---:|---:|
| R0 | `0 / 7.61e-16` | `0` | `1.5140e-5` |
| R1 | `0.016 / 0.0159999999999990` | `0.18295450567904434` | `1.2370e-5` |
| R2 | `0.065092 / [0.06509200000000008, 0.06509200000000027]` | `[0.3805503208347219, 0.38055032083472196]` | `9.3786e-6` |

The last-bit R2 lambda difference is the deterministic result of solving
each block independently, not a reused constant. All 12 distributions
had strictly positive support. R0 matched exact uniform full support.
`X1`, `X2`, `Z`, and `Y` were uniform to maximum absolute error
`9.11e-18`; `H(Y|X1,X2)` and `I(Y;X1,X2)` matched their frozen values to
the registered tolerance, with maximum joint-information absolute error
`1.07e-14`.

## Frozen transition outcomes

Onset was decoded without modification: full-support `R < 0.9` and
layer-6 linear-probe accuracy `> 0.5`, both retained at the next
registered probe checkpoint. A first pass at 10M cannot establish onset
because no later checkpoint exists. Establishment was independently
defined by `R < 0.1`, center accuracy `> 0.9`, and layer-6 linear accuracy
`> 0.9`.

| Condition | Verified onset by 5M | Verified onset by 10M | Onset exposure by block (M) | Established by 5M / 10M | Mean R at 5M pre / 10M |
|---|---:|---:|---|---:|---:|
| R0 | 0/8 | 1/8 | `—, —, 7.5, —, —, —, —, —` | 0/8 / 2/8 | `1.000239 / 0.432743` |
| R1 | 5/8 | 6/8 | `4, 4, 5, 4, 7.5, —, —, 4` | 4/8 / 6/8 | `0.303076 / 0.110766` |
| R2 | 8/8 | 8/8 | `3, 3, 4, 4, 4, 3, 4, 4` | 6/8 / 8/8 | `0.043710 / 0.005181` |

The two R0 establishment events at 10M do not imply two verified onset
events: establishment is a single-checkpoint criterion, whereas onset
must be retained at a subsequent registered probe checkpoint.

Terminal-coded onset ordinals were:

| Block | R0 | R1 | R2 |
|---|---:|---:|---:|
| `rule_0/permutation_0/init_0` | 8 | 3 | 2 |
| `rule_0/permutation_0/init_1` | 8 | 3 | 2 |
| `rule_0/permutation_1/init_0` | 6 | 4 | 3 |
| `rule_0/permutation_1/init_1` | 8 | 3 | 3 |
| `rule_1/permutation_0/init_0` | 8 | 6 | 3 |
| `rule_1/permutation_0/init_1` | 8 | 8 | 2 |
| `rule_1/permutation_1/init_0` | 8 | 8 | 3 |
| `rule_1/permutation_1/init_1` | 8 | 3 | 3 |

## Primary and stratified tests

| Analysis | Blocks | Exact assignments | Observed statistic | One-sided p |
|---|---:|---:|---:|---:|
| Primary, all new blocks | 8 | 1,679,616 | 110.0 | `9.525986892242037e-06` |
| `rule_0` | 4 | 1,296 | 55.5 | `0.0015432098765432098` |
| `rule_1` | 4 | 1,296 | 54.5 | `0.006172839506172839` |
| `permutation_0` | 4 | 1,296 | 55.5 | `0.0015432098765432098` |
| `permutation_1` | 4 | 1,296 | 54.5 | `0.006172839506172839` |

The primary result is not confined to one coefficient rule or one cue
permutation. The stratified results do not erase heterogeneity: R1
verified onset occurred in 4/4 `rule_0` blocks but only 2/4 `rule_1`
blocks. R2 occurred in 4/4 blocks under each rule and each permutation.
Thus cue geometry did not dominate generality in this replication, while
the new coefficient rule influenced the mid-dose response.

## Paired R differences

Differences are right condition minus left condition; negative values
favor the higher registered dose.

| Comparison | Mean paired delta R at 5M pre | Mean paired delta R at 10M | Mean onset-ordinal delta |
|---|---:|---:|---:|
| R1 − R0 | `-0.697162` | `-0.321977` | `-3.000` |
| R2 − R1 | `-0.259366` | `-0.105585` | `-2.125` |
| R2 − R0 | `-0.956528` | `-0.427562` | `-5.125` |

The adjacent R2 − R1 difference was not negative in every individual
block: the 5M R difference was positive in one block, the 10M difference
was positive in one block, and two blocks tied in terminal onset ordinal.
The confirmatory conclusion comes from the frozen blocked ordering test,
not from visual separation or a claim of strict pairwise ordering in
every trajectory.

## Cue removal and persistence

At exactly 5M, each trajectory was evaluated and a second evaluation was
performed with zero optimizer steps. For R1/R2, only the sampler was
changed to uniform full support; R0 remained on uniform full support
throughout. All 24 switch records confirm:

- unchanged model and optimizer state;
- identical full-support model-output hashes;
- preserved NumPy, PyTorch CPU/MPS, and Python RNG state;
- no model, scheduler, optimizer-moment, AdamW-step, or exposure-counter
  reset.

Five R1 and all eight R2 cells were threshold-positive before the switch.
All 13 remained threshold-positive at both 6M and 10M. The R1 and R2 mean
R values continued to fall after cue removal. This persistence supports
the registered learning-path-preconditioning interpretation; the
cue-bearing sampler was not permanently required after the predictive
state had formed.

## Evaluation, probes, and artifact audit

The machine-readable result is
[`artifacts/cue_dose_confirmatory_results.json`](artifacts/cue_dose_confirmatory_results.json),
SHA-256
`5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0`.
It contains every checkpoint's exact conditional KL, normalized R,
2,000-replicate first-parent-row bootstrap interval, center accuracy,
mean center probability, predictive entropy, checkpoint diagnostics,
probe metrics, and artifact hashes.

The final integrity audit found:

- 24/24 completed trajectories and 8/8 complete paired blocks;
- 264/264 checkpoint SHA-256 sidecars valid;
- 288/288 exact full-support evaluations finite and complete;
- 168/168 selected-checkpoint probe files complete for layers 0, 3, and
  6, with linear, fixed-Fourier, and fixed-capacity MLP probes;
- 24/24 zero-step switch-invariance records passed;
- zero gradient-SNR audit files, consistent with the absence of a
  prespecified numerical anomaly;
- maximum exposure exactly 10M.

One trajectory,
`confirm_rule_1_permutation_0_init_1_R2`, was operationally interrupted
at 1,917,504 exposures. Its 1M checkpoint sidecar was validated, and the
trajectory resumed from that checkpoint with model, AdamW, scheduler,
RNG, and exposure state restored. The pre-interruption training log is
preserved as `train.interrupted_at_1917504.jsonl`; no observed result was
used to change configuration or run order.

## Interpretation and limits

The ordered replication across new rules and permutations confirms, for
this frozen parent-exposed task and model family, that the registered
cue-bearing full-support distributions change predictive-state
representation-onset timing or probability beyond the original v3.5
rule and permutation.

The result does **not** identify singleton mutual information alone as
the causal variable. Changing the registered cue distribution also
changes distribution geometry and other joint properties. The result is
not a universal Transformer capability or impossibility claim, and it is
excluded from the original production scaling fit.

Stopped as registered. No recurrent bridge, production trajectory,
architecture/optimizer/scheduler change, field-size change, model-size
increase, checkpoint extension beyond 10M, or v3.5 checkpoint reuse was
performed.
