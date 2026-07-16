# Project Kepler v3.4 — Full-Support Transfer Report

## Gate status and protocol

All no-training audits and all three preregistered transfer trajectories are
complete.  Each trajectory loaded its paired frozen G2 checkpoint at exactly
5M total exposure, preserved model and AdamW state, the absolute-exposure
schedule, NumPy and Torch RNG state, and replaced only the sampler with uniform
full support.  The runs stopped at the registered 20M total-exposure limit.

The transfer manifest is `artifacts/full_support_transfer_manifest.json`,
SHA-256 `5e158ea537ad267da5bc8bb262552ba9e28502e876a7edb3cde2e0bc0b3e1cbe`.
All runs are exploratory, diagnostic-only, and excluded from confirmatory
scaling fits.  No prior manifest was modified and no control was retrained.

At the first transfer checkpoint, every optimizer parameter has Adam step 1347,
continuing from step 1224 in the source checkpoint; the scheduler exposure
continues from 5.0M to 5.5M and the post-transfer exposure counter is exactly
500,000.  This rules out an optimizer, scheduler, or absolute-counter reset in
the primary runs.

## Artifact integrity

There are exactly 15 transfer checkpoints, 15 exact-evaluation JSON files, 15
probe JSON files, and 15 SHA-256 sidecars.  Every sidecar verifies.  All metrics
are finite and every evaluation contains the full pair space, original G2
training support, original G2 held-out support, row-bootstrap intervals, three
probe layers, per-x1/x2/z arrays and tails, and checkpoint gradient/update
diagnostics.

The following are SHA-256 hashes over the textual concatenation of sorted
`shasum -a 256` output lines:

- transfer evaluation set:
  `487a0bf4c9fe7d57108bbe8434a2f9d3f854ae0b46d4372955ca4dd460f58d17`;
- transfer probe set:
  `83414e7a2246e5ecda52f90af6dd6c971434cb5b851f2f551ac977e3c0d4981b`;
- checkpoint-sidecar set:
  `588153fb31468ce2f07e28e83e394356a9f1ed8d50833f5bbcc621b1d05a1fc0`.

Detailed artifacts are under
`artifacts/full_support_transfer_runs/full_support_transfer_s<seed>/`.

## Exact transfer metrics

`R_train` and `R_held` below always refer to the original frozen G2 partitions;
after transfer, both are intentionally sampled by the full-support sampler.
The confidence interval is a 2,000-replicate bootstrap over first-parent rows.

| seed | total exposure | exact KL | full R [95% CI] | R_train | R_held | center acc. | mean p(z) | entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.5M | 0.088452 | 0.018943 [0.018347, 0.019582] | 0.018860 | 0.019275 | 0.998486 | 0.819425 | 1.289395 |
| 0 | 6M | 0.049599 | 0.010622 [0.010312, 0.010949] | 0.010564 | 0.010853 | 0.999470 | 0.863672 | 1.069131 |
| 0 | 7.5M | 0.014560 | 0.003118 [0.003114, 0.003123] | 0.003117 | 0.003121 | 1.000000 | 0.891395 | 0.930745 |
| 0 | 10M | 0.008365 | 0.001791 [0.001790, 0.001793] | 0.001791 | 0.001792 | 1.000000 | 0.898021 | 0.886952 |
| 0 | 20M | 0.004057 | 0.000869 [0.000869, 0.000869] | 0.000869 | 0.000869 | 1.000000 | 0.898950 | 0.885064 |
| 1 | 5.5M | 0.078916 | 0.016900 [0.015346, 0.018740] | 0.016742 | 0.017535 | 0.993853 | 0.837942 | 1.213021 |
| 1 | 6M | 0.029946 | 0.006413 [0.005957, 0.007070] | 0.006382 | 0.006538 | 0.999425 | 0.881809 | 0.978808 |
| 1 | 7.5M | 0.014972 | 0.003206 [0.003165, 0.003252] | 0.003200 | 0.003233 | 1.000000 | 0.894172 | 0.907830 |
| 1 | 10M | 0.006071 | 0.001300 [0.001300, 0.001300] | 0.001300 | 0.001299 | 1.000000 | 0.897915 | 0.890298 |
| 1 | 20M | 0.002251 | 0.000482 [0.000482, 0.000482] | 0.000482 | 0.000482 | 1.000000 | 0.899497 | 0.882876 |
| 2 | 5.5M | 0.015727 | 0.003368 [0.003365, 0.003371] | 0.003365 | 0.003381 | 1.000000 | 0.894700 | 0.904258 |
| 2 | 6M | 0.011467 | 0.002456 [0.002453, 0.002459] | 0.002455 | 0.002457 | 1.000000 | 0.898858 | 0.877184 |
| 2 | 7.5M | 0.016447 | 0.003522 [0.003456, 0.003593] | 0.003514 | 0.003556 | 0.999985 | 0.896131 | 0.889239 |
| 2 | 10M | 0.005707 | 0.001222 [0.001222, 0.001222] | 0.001222 | 0.001222 | 1.000000 | 0.898705 | 0.884742 |
| 2 | 20M | 0.002152 | 0.000461 [0.000461, 0.000461] | 0.000461 | 0.000459 | 1.000000 | 0.899693 | 0.881483 |

All three seeds remain far below `R=1` immediately after the switch.  Seed 2
has a small non-monotone change from 6M to 7.5M, then recovers by 10M.  The
effect is two orders of magnitude below the uniform-predictor regime and is not
a loss of the learned rule.

## Predictive-state probes

The table reports full-space layer-6 metrics.  Linear and fixed-capacity MLP
columns are 257-way accuracies; fixed Fourier is variance-normalized R2.

| seed | exposure | linear | fixed Fourier | fixed MLP |
|---:|---:|---:|---:|---:|
| 0 | 5.5M | 0.998668 | 0.335801 | 0.999606 |
| 0 | 6M | 0.999485 | 0.359103 | 0.999727 |
| 0 | 7.5M | 1.000000 | 0.440282 | 1.000000 |
| 0 | 10M | 1.000000 | 0.487173 | 1.000000 |
| 0 | 20M | 1.000000 | 0.523425 | 1.000000 |
| 1 | 5.5M | 0.997502 | 0.344584 | 0.999304 |
| 1 | 6M | 0.999410 | 0.395509 | 0.999652 |
| 1 | 7.5M | 1.000000 | 0.456410 | 1.000000 |
| 1 | 10M | 1.000000 | 0.509563 | 1.000000 |
| 1 | 20M | 1.000000 | 0.516123 | 1.000000 |
| 2 | 5.5M | 1.000000 | 0.433808 | 1.000000 |
| 2 | 6M | 1.000000 | 0.460102 | 1.000000 |
| 2 | 7.5M | 0.999985 | 0.469173 | 0.999985 |
| 2 | 10M | 1.000000 | 0.521288 | 1.000000 |
| 2 | 20M | 1.000000 | 0.520775 | 1.000000 |

The predictive state remains linearly and nonlinearly decodable across the
switch.  The output distribution simultaneously remains calibrated, so this is
not a state-represented-but-unused outcome.

## Per-parent and target tails

These are full-space q99/maximum R values across the 257 x1, x2, or target-z
groups.  Complete worst-10 identities and arrays are in each evaluation JSON.

| seed | exposure | x1 q99/max | x2 q99/max | z q99/max |
|---:|---:|---:|---:|---:|
| 0 | 5.5M | 0.04115 / 0.04645 | 0.03003 / 0.03450 | 0.03406 / 0.04820 |
| 0 | 6M | 0.01860 / 0.02611 | 0.01959 / 0.02166 | 0.02187 / 0.02262 |
| 0 | 7.5M | 0.00321 / 0.00325 | 0.00325 / 0.00331 | 0.00393 / 0.00463 |
| 0 | 10M | 0.00182 / 0.00184 | 0.00182 / 0.00185 | 0.00211 / 0.00215 |
| 0 | 20M | 0.00087 / 0.00087 | 0.00087 / 0.00088 | 0.00104 / 0.00108 |
| 1 | 5.5M | 0.06758 / 0.10477 | 0.06652 / 0.07596 | 0.05655 / 0.12706 |
| 1 | 6M | 0.01343 / 0.07387 | 0.01490 / 0.02461 | 0.01410 / 0.01905 |
| 1 | 7.5M | 0.00461 / 0.00472 | 0.00417 / 0.00471 | 0.00441 / 0.00508 |
| 1 | 10M | 0.00130 / 0.00131 | 0.00130 / 0.00131 | 0.00147 / 0.00148 |
| 1 | 20M | 0.00049 / 0.00049 | 0.00049 / 0.00049 | 0.00064 / 0.00067 |
| 2 | 5.5M | 0.00342 / 0.00346 | 0.00342 / 0.00347 | 0.00404 / 0.00423 |
| 2 | 6M | 0.00253 / 0.00256 | 0.00252 / 0.00256 | 0.00310 / 0.00326 |
| 2 | 7.5M | 0.00593 / 0.00627 | 0.00533 / 0.00674 | 0.00637 / 0.00808 |
| 2 | 10M | 0.00123 / 0.00124 | 0.00123 / 0.00124 | 0.00135 / 0.00137 |
| 2 | 20M | 0.00047 / 0.00047 | 0.00047 / 0.00047 | 0.00059 / 0.00065 |

Seed 1 has a heterogeneous tail at 5.5M despite a low mean, but it contracts
rapidly by 7.5M.  Seed 2's small 7.5M mean-R increase is also visible across
all tail views and resolves by 10M.

## Gradient and update magnitudes

Every checkpoint records all six blocks, the embedding interface, and final
normalization.  The compact table gives the largest group at each checkpoint;
the complete per-group values are embedded in the evaluation JSON.

| seed | exposure | largest gradient group / norm | largest update/parameter group / ratio |
|---:|---:|---|---|
| 0 | 5.5M | embedding / 2.90249 | block 5 / 0.001211 |
| 0 | 6M | block 0 / 1.44438 | block 5 / 0.001220 |
| 0 | 7.5M | block 5 / 0.10629 | embedding / 0.000794 |
| 0 | 10M | block 5 / 0.02410 | block 5 / 0.000462 |
| 0 | 20M | block 5 / 0.01091 | embedding / 0.000844 |
| 1 | 5.5M | block 0 / 3.60756 | block 5 / 0.001151 |
| 1 | 6M | block 0 / 0.64896 | block 5 / 0.001210 |
| 1 | 7.5M | embedding / 0.21470 | block 5 / 0.001194 |
| 1 | 10M | block 5 / 0.01562 | block 5 / 0.000511 |
| 1 | 20M | block 5 / 0.00431 | embedding / 0.000633 |
| 2 | 5.5M | block 0 / 0.05647 | block 5 / 0.000866 |
| 2 | 6M | embedding / 0.07053 | block 5 / 0.000999 |
| 2 | 7.5M | embedding / 0.40336 | block 5 / 0.001714 |
| 2 | 10M | block 5 / 0.00596 | block 5 / 0.000391 |
| 2 | 20M | embedding / 0.00754 | block 3 / 0.000681 |

The larger seed-0/1 early gradients decay with calibration.  Seed 2's 7.5M
gradient/update increase coincides with its temporary R increase.  This is a
checkpoint association and is not interpreted causally.

## Frozen-control comparison

The controls below are the already-existing scratch G1 and restricted-support
G2 trajectories.  No interpolation or retraining was performed.

| seed | trajectory | R at 5M | R at 10M | R at 20M |
|---:|---|---:|---:|---:|
| 0 | G1 scratch full | 1.000263 | 0.540392 | 0.341753 |
| 0 | G2 restricted continuation | 0.039394 | 0.003116 | 0.001134 |
| 0 | transfer to full | 0.039394 source | 0.001791 | 0.000869 |
| 1 | G1 scratch full | 1.000417 | 0.582831 | 0.209549 |
| 1 | G2 restricted continuation | 0.017478 | 0.002476 | 0.000490 |
| 1 | transfer to full | 0.017478 source | 0.001300 | 0.000482 |
| 2 | G1 scratch full | 0.814021 | 0.510010 | 0.612722 |
| 2 | G2 restricted continuation | 0.011948 | 0.002179 | 0.000897 |
| 2 | transfer to full | 0.011948 source | 0.001222 | 0.000461 |

At both matched post-switch endpoints, transfer remains at least as good as the
paired restricted continuation within this exploratory three-seed sample.  The
scratch G1 controls are much less learned at the same absolute exposures.  This
comparison supports a bootstrapping interpretation; it does not isolate which
feature of the restricted mask produced the faster transition.

## Registered interpretation

The primary observed category is **R remains low after switching to full
support**.  In all three seeds, full R is lower at 5.5M than at the 5M source
checkpoint and continues toward zero.  This is evidence that restricted support
bootstrapped a predictive representation that operates and continues to adapt
under the pure-synergy full-support distribution.

The other registered outcomes are resolved as follows:

- **R rises temporarily and later recovers:** there is no immediate post-switch
  rise.  Seed 2 has a smaller 6M-to-7.5M fluctuation followed by recovery, which
  is evidence for a modest local adaptation/optimization cost, not a global
  representation failure.
- **R returns toward 1 and remains there:** not observed.  Performance is not
  dependent on continued access to the frozen restricted-support singleton cue.
- **Seeds diverge:** quantitative paths and early tails differ, but all three
  seeds share the same qualitative low-R transfer outcome.  The present sample
  does not show a seed-level split between successful and failed transfer.

## Distinctions required by the gate

1. **Out-of-support pair generalization** was established before transfer: G2
   predicted the frozen 20% pairs that its sampler never emitted.
2. **Pure-synergy learning under full-support training** is represented by G1
   scratch.  It can begin late and is strongly seed-dependent in this sample.
   The transfer result is different: the models start full-support training only
   after a representation has formed and then retain/improve it.  It does not
   establish that de novo full-support learning is equally likely.
3. **Support-induced symmetry breaking** is supported by the exact information
   audit: the G2 training support has nonzero singleton information
   (`C_single=0.065092`) whereas G1 has exactly zero to numerical tolerance.
   The transfer shows that this cue is not needed to maintain the solution after
   formation.  It does not, by itself, prove that the singleton cue caused the
   original transition.

These conclusions apply only to the frozen parent-exposed architecture,
coefficients, mask, exact-soft-label objective, optimizer state, and three seeds.
They are not universal Transformer capability or impossibility claims.

## Stop condition

The stop condition is satisfied.  No matched recurrent/objective experiment,
production trajectory, new support mask, or run beyond 20M was launched.
