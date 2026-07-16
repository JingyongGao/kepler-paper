# Project Kepler v3.3 — Compositional Generalization Report

## Gate status

The checkpoint-gradient audit and all six preregistered G1/G2 trajectories are
complete.  Every trajectory stopped at 40M pair exposures.  No production,
remaining micro-pilot, M2 extension, field-size sweep, Fourier initialization,
curriculum, recurrent parent-marker, bilinear/gated-MLP, or optimizer-change
experiment was launched.

The production manifest remains suspended and unmodified.  The frozen mechanism
commit is tagged `kepler-v3.3-mechanism1`; this work is on
`kepler-v3.3-compositional-generalization`.

## Frozen inputs and audit artifacts

- The compositional manifest is
  `artifacts/compositional_generalization_manifest.json`, SHA-256
  `ed4409d18c74d35614e731e90742f586620230b2bfd64b2e7e2fe5294c8d7d72`.
- The parent-pair split is `artifacts/parent_pair_split.json`, SHA-256
  `273fea214893ffd818737a1c9faa75b8d47a5bc3cafb184796868580d757cab2`.
  Its preregistered seed is
  `ba2f84572c3b0c5d3f27b0bf5d8c8e92505cc0a5b18c9274b0ef013c3d90c02f`;
  the 66,049-byte assignment-mask SHA-256 is
  `f16c708e5b49f1aab8a0f8e4ccaadcc8da506ca22d8ac210a583a12d5a1371c9`.
- The split contains 52,839 training pairs and 13,210 held-out pairs.  Every
  `x1`, `x2`, and target `z` occurs in both partitions.  Per-value count ranges
  are train `x1=188–222`, `x2=189–223`, `z=184–223`, and held-out
  `x1=35–69`, `x2=34–68`, `z=34–73`.
- The construction-level restricted-sampler audit drew 1,000,000 pairs with
  zero held-out leakage.  Checkpoint exposure counters independently record
  zero leakage for every G2 checkpoint.
- The no-training gradient artifact is
  `artifacts/checkpoint_gradient_dynamics_audit.json`, SHA-256
  `4a45b5cd539b25de966e814ef22d4287aed05af1f8ada1979a02c9d3f4183cfb`.
  Definitions and the complete M1–M4 interpretation are in
  `CHECKPOINT_GRADIENT_DYNAMICS_REPORT.md`.

The gradient audit covers all 19 registered M1–M4 checkpoints using 32 fixed,
independent exact-soft-label batches and no optimizer steps.  M2 and M4 remain
near `R=1` with low group-gradient SNR and chance probes.  M3 remains near
`R=1` through 5M, while its partially learned 10M checkpoint has sharply larger
embedding, Q/K/V, MLP, and normalization SNR and improved state probes.  These
are associations at checkpoints, not evidence that the gradient statistics
caused either learning or failure.

## Execution and artifact integrity

All six runs have the seven registered checkpoints and evaluations at 0.5M,
1M, 2M, 5M, 10M, 20M, and 40M: 42 checkpoints, 42 exact-evaluation JSON files,
42 probe JSON files, and 42 checkpoint SHA-256 sidecars.  All sidecars verify.
All evaluation values are finite and every file contains the four registered
partitions, 2,000-replicate first-parent-row bootstrap intervals, the three
probe layers, and the `x1`, `x2`, target-`z`, and pair-frequency breakdowns.

For a compact integrity handle, the following hashes are SHA-256 over the
textual concatenation of sorted `shasum -a 256` output lines:

- exact-evaluation JSON set: `217a601824488fc7bdffc542a52e608f9ca10b1644562549ca0afbae8bbeabe3`;
- probe JSON set: `32423a0965b66159c1be5f5e237ce02f4a02563fa69a351df037fe37e0750fcb`;
- checkpoint-sidecar set: `0be35cb2a263acbf6d1c5d74cb76491ca9d2447bc4557bed861d124b01d75ab1`.

Detailed files are under `artifacts/compositional_runs/<run_id>/`.  Each
`checkpoint_<D>.composition.eval.json` reports exact KL, normalized `R`, the
row-bootstrap interval, center accuracy and probability, entropy, all
stratified results, leakage, and `G`.  Its matching
`checkpoint_<D>.composition.probes.json` reports linear, fixed-Fourier, and
fixed-capacity MLP metrics for every registered layer and partition.

## Checkpoint results

`R_train` is the actual sampling support: all 66,049 pairs for G1 and the frozen
52,839-pair support for G2.  The G1 `R_held` column is only a labeled subset of
its exposed full support, so it is a code-matched reference and not an
out-of-support generalization estimate.  `G` is reported without clipping only
when `R_train < 0.95`.

| run | exposure | R_train | R_held | R_full | G |
|---|---:|---:|---:|---:|---:|
| G1 s0 | 0.5M | 1.000636 | 1.000631 | 1.000636 | — |
| G1 s0 | 1M | 1.000244 | 1.000199 | 1.000244 | — |
| G1 s0 | 2M | 1.000358 | 1.000324 | 1.000358 | — |
| G1 s0 | 5M | 1.000263 | 1.000300 | 1.000263 | — |
| G1 s0 | 10M | 0.540392 | 0.540701 | 0.540392 | 0.999328 |
| G1 s0 | 20M | 0.341753 | 0.341452 | 0.341753 | 1.000457 |
| G1 s0 | 40M | 0.040596 | 0.040039 | 0.040596 | 1.000581 |
| G1 s1 | 0.5M | 1.000539 | 1.000576 | 1.000539 | — |
| G1 s1 | 1M | 1.000321 | 1.000297 | 1.000321 | — |
| G1 s1 | 2M | 1.000366 | 1.000526 | 1.000366 | — |
| G1 s1 | 5M | 1.000417 | 1.000363 | 1.000417 | — |
| G1 s1 | 10M | 0.582831 | 0.583010 | 0.582831 | 0.999572 |
| G1 s1 | 20M | 0.209549 | 0.209676 | 0.209549 | 0.999839 |
| G1 s1 | 40M | 0.061681 | 0.061504 | 0.061681 | 1.000189 |
| G1 s2 | 0.5M | 1.000616 | 1.000545 | 1.000616 | — |
| G1 s2 | 1M | 1.000358 | 1.000294 | 1.000358 | — |
| G1 s2 | 2M | 1.000355 | 1.000469 | 1.000355 | — |
| G1 s2 | 5M | 0.814021 | 0.813606 | 0.814021 | 1.002230 |
| G1 s2 | 10M | 0.510010 | 0.510908 | 0.510010 | 0.998167 |
| G1 s2 | 20M | 0.612722 | 0.614871 | 0.612722 | 0.994450 |
| G1 s2 | 40M | 0.314210 | 0.314119 | 0.314210 | 1.000134 |
| G2 s0 | 0.5M | 1.000309 | 1.002436 | 1.000734 | — |
| G2 s0 | 1M | 0.999451 | 1.005884 | 1.000738 | — |
| G2 s0 | 2M | 0.988474 | 1.033251 | 0.997430 | — |
| G2 s0 | 5M | 0.038866 | 0.041509 | 0.039394 | 0.997250 |
| G2 s0 | 10M | 0.003109 | 0.003143 | 0.003116 | 0.999965 |
| G2 s0 | 20M | 0.001131 | 0.001145 | 0.001134 | 0.999986 |
| G2 s0 | 40M | 0.001898 | 0.001938 | 0.001906 | 0.999960 |
| G2 s1 | 0.5M | 1.000293 | 1.002239 | 1.000682 | — |
| G2 s1 | 1M | 0.999574 | 1.006147 | 1.000889 | — |
| G2 s1 | 2M | 0.987356 | 1.034685 | 0.996822 | — |
| G2 s1 | 5M | 0.017221 | 0.018509 | 0.017478 | 0.998689 |
| G2 s1 | 10M | 0.002469 | 0.002503 | 0.002476 | 0.999967 |
| G2 s1 | 20M | 0.000490 | 0.000491 | 0.000490 | 0.999999 |
| G2 s1 | 40M | 0.000424 | 0.000424 | 0.000424 | 1.000000 |
| G2 s2 | 0.5M | 1.000319 | 1.002522 | 1.000760 | — |
| G2 s2 | 1M | 0.999316 | 1.008216 | 1.001096 | — |
| G2 s2 | 2M | 0.985598 | 1.036363 | 0.995752 | — |
| G2 s2 | 5M | 0.011858 | 0.012306 | 0.011948 | 0.999547 |
| G2 s2 | 10M | 0.002171 | 0.002209 | 0.002179 | 0.999962 |
| G2 s2 | 20M | 0.000897 | 0.000898 | 0.000897 | 0.999999 |
| G2 s2 | 40M | 0.000406 | 0.000402 | 0.000405 | 1.000004 |

The table is metric reporting, not a visual-curve pass condition.

## Endpoint calibration and state probes

At 40M, the G2 train/held-out exact-KL and calibration metrics are:

| seed | split | exact KL (nats) | R [row-bootstrap 95% CI] | center acc. | mean p(z) | entropy (nats) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | train | 0.008864 | 0.001898 [0.001646, 0.002186] | 0.999489 | 0.891993 | 0.929544 |
| 0 | held | 0.009051 | 0.001938 [0.001643, 0.002289] | 0.999092 | 0.891918 | 0.929274 |
| 1 | train | 0.001978 | 0.000424 [0.000423, 0.000425] | 1.000000 | 0.899586 | 0.882443 |
| 1 | held | 0.001980 | 0.000424 [0.000420, 0.000427] | 1.000000 | 0.899595 | 0.882378 |
| 2 | train | 0.001895 | 0.000406 [0.000405, 0.000407] | 1.000000 | 0.899664 | 0.881898 |
| 2 | held | 0.001877 | 0.000402 [0.000398, 0.000406] | 1.000000 | 0.899669 | 0.881862 |

The selected layer-6 held-out probes show the state-formation transition.  The
fixed-Fourier column is variance-normalized R2; the other probe columns are
257-way accuracies.

| seed | exposure | R_train / R_held | linear | fixed Fourier | fixed MLP |
|---:|---:|---:|---:|---:|---:|
| 0 | 2M | 0.988474 / 1.033251 | 0.003407 | -0.011191 | 0.004164 |
| 0 | 5M | 0.038866 / 0.041509 | 0.999243 | 0.306745 | 0.999924 |
| 0 | 10M | 0.003109 / 0.003143 | 1.000000 | 0.473448 | 1.000000 |
| 0 | 40M | 0.001898 / 0.001938 | 0.999849 | 0.517344 | 0.999773 |
| 1 | 2M | 0.987356 / 1.034685 | 0.003861 | -0.010997 | 0.003558 |
| 1 | 5M | 0.017221 / 0.018509 | 1.000000 | 0.325085 | 1.000000 |
| 1 | 10M | 0.002469 / 0.002503 | 1.000000 | 0.487990 | 1.000000 |
| 1 | 40M | 0.000424 / 0.000424 | 1.000000 | 0.535140 | 1.000000 |
| 2 | 2M | 0.985598 / 1.036363 | 0.004239 | -0.011053 | 0.003709 |
| 2 | 5M | 0.011858 / 0.012306 | 0.999622 | 0.354518 | 1.000000 |
| 2 | 10M | 0.002171 / 0.002209 | 1.000000 | 0.498389 | 1.000000 |
| 2 | 40M | 0.000406 / 0.000402 | 1.000000 | 0.540459 | 1.000000 |

Thus, at 2M neither the output distribution nor the registered probes show a
learned state.  By 5M, all three G2 seeds have near-perfect linear/MLP state
decoding on never-trained pairs and low held-out `R`; the represented state is
also used by the output distribution.  The fixed Fourier metric improves more
gradually and does not need to saturate for the categorical state to be decoded.

## Stratified checks

At 40M, G2 training-support low/middle/high frequency-bin `R` values are,
respectively, `0.001892/0.001906/0.001890` (s0),
`0.000423/0.000423/0.000425` (s1), and
`0.000403/0.000407/0.000407` (s2).  All G2 held-out pairs are in the zero-frequency
bin, with `R=0.001938`, `0.000424`, and `0.000402`.  There is no persistent
frequency-associated held-out penalty after learning.

The 40M held-out per-value `R` ranges are:

| seed | by x1 | by x2 | by target z |
|---:|---:|---:|---:|
| 0 | 0.000723–0.027646 | 0.000721–0.019927 | 0.000200–0.027504 |
| 1 | 0.000328–0.000503 | 0.000333–0.000513 | 0.000006–0.000577 |
| 2 | 0.000316–0.000482 | 0.000311–0.000481 | 0.000002–0.000575 |

Seed 0 retains a small but real heterogeneous tail despite its low mean `R`;
the row-bootstrap interval and complete 257-entry arrays are retained rather
than hiding that heterogeneity behind the mean.

## Late-transition reproducibility

All three G1 seeds remain at the uniform-predictor baseline through 2M and all
have departed from it by 10M.  Seeds 0 and 1 cross the registered `R<0.95`
indicator between 5M and 10M.  Seed 2 crosses between 2M and 5M, but learns more
slowly thereafter and is non-monotone (`R=0.510` at 10M, `0.613` at 20M,
`0.314` at 40M).  Therefore the qualitative late transition reproduces across
seeds, while its timing, speed, monotonicity, and 40M endpoint do not.

The G2 restricted-support trajectories all cross between 2M and 5M and converge
more rapidly than their paired G1 references.  This is an exploratory paired
result under the frozen protocol, not authorization to change production
settings based on the effect direction.

## Registered interpretation

The observed G2 pattern is **train and held-out improve similarly**.  At 5M,
the three unclipped `G` values are 0.997250, 0.998689, and 0.999547; at 40M they
are 0.999960, 1.000000, and 1.000004.  Held-out pairs were never emitted by the
training sampler, yet their exact conditional KL, center accuracy, calibration,
and probes track the training support.  Under the preregistered interpretation,
this is evidence for compositional rule generalization for this parent-exposed,
exact-soft-label, p=257 task.

The alternatives are not supported at the learned checkpoints:

- **Train improves while held-out remains near R=1:** not observed, so the
  full-support M3 result is not explained here by pair memorization or local
  interpolation alone.
- **Both remain near R=1:** true only during the early 0.5M–2M transient, not
  after 5M; restricted support did not prevent learning from starting.
- **Held-out improves but less than train:** a small early support gap is visible
  at 2M while both are still near baseline, but there is no persistent gap once
  `R_train < 0.95` and `G` becomes reportable.

This is not a universal Transformer capability result, and it is not an
impossibility result for recurrent inputs.  It applies to the frozen exploratory
parent-exposed architecture, rule, exact-soft-label objective, split, optimizer,
and three initialization seeds only.  Production remains suspended and no
production setting is changed by this report.
