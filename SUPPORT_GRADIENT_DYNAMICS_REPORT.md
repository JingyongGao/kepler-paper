# Project Kepler v3.4 — Paired Support Gradient-Dynamics Audit

## Protocol

This is a no-training audit of the three registered G1/G2 initialization pairs
at initialization and 0.5M, 2M, 5M, 10M, and 40M checkpoints.  No optimizer
step was taken.  Every statistic uses 32 fixed batches of 512 exact-soft-label
parent pairs.  Two audit streams are retained per model:

- support-specific: independent full-support batches for G1 and frozen G2
  training-support batches for G2;
- common full support: one additional full-support set shared by the paired
  models and reused across checkpoints.

The complete global and per-block statistics, all semantic groups, behavior and
probe summaries, and G2-minus-G1 differences are in
`artifacts/support_gradient_dynamics_audit.json`, SHA-256
`08dec52ff22973a7ae1fdd74793548cc460ee3423dadad15d861e3bd8c3f0c41`.

Metrics use the initialization-audit definitions:

- mean-gradient norm: `||mean(g)||`;
- variance: `mean(||g_i - mean(g)||^2)`;
- SNR: `||mean(g)||^2 / variance`;
- pairwise cosine: the mean over all unordered pairs of unit batch gradients.

The tied embedding is separated into its input contribution and direct output
contribution; fused QKV weights are split into Q, K, and V.  MLP and
normalization groups and every Transformer block are reported separately.

## Initialization control

For every seed, reconstructing G1 and G2 from the registered seed gives exactly
the same initial model.  On each shared full-support audit set, the maximum
absolute G1/G2 difference over loss and all recorded gradient metrics is 0
(tolerance `1e-10`).  Support-specific initialization differences therefore
measure the two different fixed support samples, not different initial weights.

## Common-full-support global dynamics

These are `global.total` statistics on the shared pure-synergy audit set.

| seed | exposure | arm | mean-grad norm | variance | SNR | pairwise cosine |
|---:|---:|---|---:|---:|---:|---:|
| 0 | init | G1/G2 | 0.468154 | 2.158429 | 0.101540 | 0.061605 |
| 0 | 0.5M | G1 | 0.151421 | 0.496651 | 0.046166 | 0.013283 |
| 0 | 0.5M | G2 | 0.162029 | 0.541010 | 0.048527 | 0.015493 |
| 0 | 2M | G1 | 0.108920 | 0.263343 | 0.045050 | 0.012086 |
| 0 | 2M | G2 | 0.518371 | 3.109894 | 0.086405 | 0.048299 |
| 0 | 5M | G1 | 0.103993 | 0.234351 | 0.046147 | 0.013237 |
| 0 | 5M | G2 | 7.376621 | 7.160171 | 7.599614 | 0.885231 |
| 0 | 10M | G1 | 90.386688 | 54.128742 | 150.931900 | 0.993991 |
| 0 | 10M | G2 | 0.414548 | 0.194532 | 0.883403 | 0.552683 |
| 0 | 40M | G1 | 55.768623 | 368.471436 | 8.440653 | 0.905826 |
| 0 | 40M | G2 | 1.010751 | 1.320776 | 0.773497 | 0.481522 |
| 1 | init | G1/G2 | 0.488920 | 2.680815 | 0.089168 | 0.051345 |
| 1 | 0.5M | G1 | 0.160121 | 0.522280 | 0.049090 | 0.016116 |
| 1 | 0.5M | G2 | 0.155964 | 0.572425 | 0.042494 | 0.009849 |
| 1 | 2M | G1 | 0.101530 | 0.270325 | 0.038133 | 0.005551 |
| 1 | 2M | G2 | 0.659680 | 3.805760 | 0.114347 | 0.071192 |
| 1 | 5M | G1 | 0.100504 | 0.243498 | 0.041483 | 0.008911 |
| 1 | 5M | G2 | 3.263104 | 1.732024 | 6.147634 | 0.867772 |
| 1 | 10M | G1 | 59.890640 | 30.239271 | 118.616905 | 0.993452 |
| 1 | 10M | G2 | 0.263475 | 0.100936 | 0.687750 | 0.483457 |
| 1 | 40M | G1 | 82.565979 | 429.561737 | 15.869991 | 0.949006 |
| 1 | 40M | G2 | 0.008303 | 0.000006 | 10.707458 | 0.913179 |
| 2 | init | G1/G2 | 0.467804 | 2.311827 | 0.094661 | 0.056707 |
| 2 | 0.5M | G1 | 0.141852 | 0.516326 | 0.038972 | 0.006512 |
| 2 | 0.5M | G2 | 0.155446 | 0.578036 | 0.041803 | 0.009032 |
| 2 | 2M | G1 | 0.098481 | 0.273257 | 0.035492 | 0.003154 |
| 2 | 2M | G2 | 0.546326 | 4.347480 | 0.068654 | 0.033883 |
| 2 | 5M | G1 | 12.535653 | 7.380011 | 21.293005 | 0.967286 |
| 2 | 5M | G2 | 4.571953 | 4.396435 | 4.754478 | 0.836003 |
| 2 | 10M | G1 | 161.632477 | 186.605637 | 140.001450 | 0.994417 |
| 2 | 10M | G2 | 0.163823 | 0.031632 | 0.848451 | 0.506059 |
| 2 | 40M | G1 | 379.322968 | 949.568298 | 151.527710 | 0.994382 |
| 2 | 40M | G2 | 0.003218 | 0.000001 | 6.927545 | 0.871355 |

At 2M, all models still have near-baseline R and chance probes, but G2 already
has a larger mean-gradient norm and SNR than paired G1 on the common
full-support set.  This is an association before the registered representation
transition, not proof that it causes the transition.

## Semantic groups at 5M

The table reports support-specific SNR.  At this checkpoint the G2 full-space R
values are 0.039394, 0.017478, and 0.011948, with selected layer-6 linear probes
at 0.9996 or above.  G1 seeds 0 and 1 remain near R=1 with chance probes; G1
seed 2 has begun its slower transition (`R=0.814`, linear probe `0.0537`).

| seed | arm | embedding | direct output | Q | K | V | MLP | norm |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | G1 | 0.0189 | 0.0391 | 0.0359 | 0.0266 | 0.0201 | 0.0468 | 0.0528 |
| 0 | G2 | 15.2306 | 0.7000 | 4.3489 | 7.7217 | 5.1886 | 5.6981 | 17.5515 |
| 1 | G1 | 0.0523 | 0.0406 | 0.0474 | 0.0556 | 0.0422 | 0.0459 | 0.0862 |
| 1 | G2 | 13.0830 | 0.8663 | 3.8294 | 3.1576 | 4.5294 | 5.6873 | 12.5334 |
| 2 | G1 | 19.0677 | 0.2864 | 8.9508 | 14.6278 | 12.9723 | 20.6467 | 28.2928 |
| 2 | G2 | 10.5746 | 0.2521 | 2.3758 | 4.8989 | 3.1271 | 3.2207 | 9.3842 |

The direct output slice rises much less than the embedding, Q/K/V, MLP, and
normalization groups.  The common-full-support total SNR is also high for all
three G2 models at 5M (7.600, 6.148, and 4.754), so the coherent checkpoint
gradient is not confined to the restricted-support audit distribution.

## Paired differences

Every JSON cell includes the G2-minus-G1 difference for mean-gradient norm,
variance, SNR, pairwise cosine, mean batch norm, and RMS batch norm.  For
`global.total`, the common-full-support SNR differences at 2M are +0.0414,
+0.0762, and +0.0332.  At 5M they are +7.5535, +6.1062, and -16.5385: the
negative seed-2 value reflects that its paired G1 has already entered a high-SNR
partial transition.  At 10M and 40M, G1 often has much larger residual error
gradients while G2 is close to its calibrated optimum, so a negative difference
does not mean that the G2 representation disappeared.

## Relation to R and probes

- Initialization and 0.5M: low SNR, R near 1, and chance state probes coexist.
- 2M: G2 support-specific and common-full mean-gradient signals have increased,
  while R and probes remain close to baseline.
- 5M: all G2 seeds have low R and near-perfect linear/MLP decoding on held-out
  pairs, together with coherent internal-group gradients on both audit streams.
- 10M–40M: G2 remains learned.  Its absolute mean gradients generally shrink as
  exact KL approaches zero; ratio-valued SNR can remain moderate or high when
  both signal and variance are tiny.
- G1 transitions later and with seed-dependent timing, matching the previously
  reported behavioral and probe heterogeneity.

These checkpoint relationships are not causal estimates.  Restricted-support
singleton information may change the population gradient, but a learned
representation also changes that gradient, and finite fixed audit sets add
sampling variation.  The audit therefore does not claim that an SNR difference
caused either support-induced symmetry breaking or subsequent generalization.
