# Project Kepler v3.5 — Full-Support Cue-Dose Causal Gate

## Executive result

The registered gate is complete: all 30 exploratory seed-condition cells reached the 10M exposure stop, all 30 zero-step sampler-switch checks passed, and no prohibited experiment was launched.

Within the fixed `m=64` family, increasing the registered cue dose was associated with earlier predictive-state formation. C0 produced no verified onset in 5/5 seeds by 10M, C1 produced onset in 5/5 seeds at 5–7.5M, and C2/C3 produced onset in 5/5 seeds at 3–4M. The preregistered exact blocked Page-style permutation test gave one-sided `p = 1.6075102880658438e-05` (128 upper-tail assignments among `24^5 = 7,962,624`). This is evidence that the cue-bearing full-support training distribution changes the probability or timing of representation formation in this paired design.

The matched-index geometry controls were not interchangeable. Relative to `m=64`, the concentrated `m=1` condition delayed onset by a paired mean of one probe-checkpoint ordinal (seed deltas `0, 2, 1, 1, 1`), whereas the diffuse `m=128` condition had heterogeneous deltas (`1, 0, 0, -1, 1`) and larger 5M variability. Frequency concentration or spectral geometry therefore matters beyond the scalar cue index. The results do **not** identify singleton mutual information alone as the causal variable.

All 25 nonzero-cue cells retained the representation threshold after switching to uniform full support. The models did not require continued cue-bearing sampling: every pre-switch threshold-positive cell remained positive at 6M and 10M, and all 25 cells reached establishment by 10M.

## Scope, freeze, and preregistration integrity

- The annotated tag `kepler-v3.4-transfer1` resolves to frozen commit `1a81d8ea7057729cfb4275f1867f03687e0651b5`.
- Work was performed on branch `kepler-v3.5-cue-dose`.
- The freeze record is [`artifacts/v3_5_freeze_record.json`](artifacts/v3_5_freeze_record.json).
- No previous manifest was modified. The cue-dose manifest is [`artifacts/cue_dose_manifest.json`](artifacts/cue_dose_manifest.json), SHA-256 `cfae2e4656c11cf37be387147e76b2f4f9585805320935893f6b27056d42957b`.
- The attempted C0 reuse check failed before performance results were consulted because the newly required 3M/4M checkpoint boundaries changed batch partitioning and pair/noise RNG interleaving. The preregistered fallback was therefore applied: all five C0 controls were trained anew. The failure record is [`artifacts/cue_c0_reuse_failure.json`](artifacts/cue_c0_reuse_failure.json).
- Every run is marked exploratory, diagnostic-only, and excluded from confirmatory scaling fits.

Frozen earlier-manifest hashes are recorded without changing their files:

| Manifest | SHA-256 |
|---|---|
| Production | `4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10` |
| Pilot | `3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0` |
| Learnability | `95a21c5a89af958ae986b3d3284c9e610bd37c16a0d3e68dc2d01bd83c9f5af4` |
| Mechanism isolation | `d854fa799a361287111c2f3ad996558992f72741a6560883517caf0979070b5a` |
| Compositional generalization | `ed4409d18c74d35614e731e90742f586620230b2bfd64b2e7e2fe5294c8d7d72` |
| Full-support transfer | `5e158ea537ad267da5bc8bb262552ba9e28502e876a7edb3cde2e0bc0b3e1cbe` |

## Exact distribution audit

The full audit is in [`artifacts/cue_distribution_audit.json`](artifacts/cue_distribution_audit.json), SHA-256 `e6ce056320307dddd21691e475e7c604f7d02f880f89821f8c89545803fddc0a`; the derivation is in [`CUE_DISTRIBUTION_AUDIT_REPORT.md`](CUE_DISTRIBUTION_AUDIT_REPORT.md).

The registered cue coefficient is `c=1`. Every `B_m` is a prefix of the one permutation registered by SHA-256 seed `e6cccb69e7dba372d1a5db459b15625cc06c614ea8ba59db637d2d843ee282d9`. Lambdas were obtained by 80-step deterministic bisection over `[0, nextafter(1,0)]`; rounded values were not hardcoded.

| Condition | m | Target / achieved cue index | Solved lambda | Min / max pair probability | Pair entropy (nats) | Expected minimum pair exposure at 5M |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 64 | 0 / `7.61e-16` | `0` | `1.5140e-5 / 1.5140e-5` | 11.098152 | 75.70 |
| C1 | 64 | 0.016 / `0.0159999999999990` | `0.18295450567904434` | `1.2370e-5 / 2.3493e-5` | 11.052352 | 61.85 |
| C2 | 64 | 0.065092 / `0.0650920000000006` | `0.3805503208347219` | `9.3786e-6 / 3.2515e-5` | 10.912161 | 46.89 |
| C3 | 64 | 0.13 / `0.1299999999999998` | `0.5461664584309741` | `6.8712e-6 / 4.0077e-5` | 10.725731 | 34.36 |
| Concentrated | 1 | 0.065092 / `0.0650919999999999` | `0.0788263575575168` | `1.3947e-5 / 3.2066e-4` | 10.921890 | 69.73 |
| Diffuse | 128 | 0.065092 / `0.0650920000000001` | `0.5948892782779377` | `6.1335e-6 / 2.4217e-5` | 10.907692 | 30.67 |

Exhaustive enumeration found maximum marginal-uniformity error `1.91e-17`, maximum singleton-information asymmetry `8.88e-16`, and maximum joint-information error `1.60e-14`. All pair probabilities are strictly positive. `X1`, `X2`, `Z`, and `Y` are uniform; `H(Y|X1,X2)` equals the registered noise entropy; `I(Y;X1,X2)=log(257)-H(nu_eta)`; and the lambda-zero pair probabilities match G1 exactly.

## Registered transition outcomes

Onset was applied exactly as frozen: `R < 0.9` and layer-6 linear accuracy `> 0.5`, retained at the next registered probe checkpoint. A state first passing only at 10M cannot count as onset because there is no next checkpoint. No threshold was changed after observing results. Establishment required `R < 0.1`, center accuracy `> 0.9`, and layer-6 linear accuracy `> 0.9`.

| Condition | Onset by 5M | Onset exposure by paired seed (M) | Establishment by 5M | Mean R at 5M pre-switch | Mean R at 10M | Pre-positive retained at 6M / 10M |
|---|---:|---|---:|---:|---:|---:|
| C0 | 0/5 | —, —, —, —, — | 0/5 | 0.973341 | 0.628411 | n/a |
| C1 | 2/5 | 7.5, 5, 6, 6, 5 | 0/5 | 0.428643 | 0.025721 | 2/2 / 2/2 |
| C2 | 5/5 | 4, 3, 3, 4, 3 | 5/5 | 0.013564 | 0.003252 | 5/5 / 5/5 |
| C3 | 5/5 | 4, 3, 3, 4, 4 | 5/5 | 0.007108 | 0.001620 | 5/5 / 5/5 |
| Concentrated, m=1 | 5/5 | 4, 5, 4, 5, 4 | 5/5 | 0.014501 | 0.001528 | 5/5 / 5/5 |
| Diffuse, m=128 | 5/5 | 5, 3, 3, 3, 4 | 4/5 | 0.055923 | 0.007754 | 5/5 / 5/5 |

The complete machine-readable result, including every checkpoint's exact KL, normalized R, row-bootstrap interval, center accuracy/probability, predictive entropy, all selected layer-wise probes, every gradient group, checkpoint hashes, and output hashes, is [`artifacts/cue_dose_results.json`](artifacts/cue_dose_results.json), SHA-256 `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.

## Dose trend and paired effects

Terminal-coded onset ordinals for C0/C1/C2/C3 were:

| Seed | C0 | C1 | C2 | C3 |
|---:|---:|---:|---:|---:|
| 0 | 8 | 6 | 3 | 3 |
| 1 | 8 | 4 | 2 | 2 |
| 2 | 8 | 5 | 2 | 2 |
| 3 | 8 | 5 | 3 | 3 |
| 4 | 8 | 4 | 2 | 3 |

Against paired C0, mean onset-ordinal changes were `-3.2` for C1, `-5.6` for C2, and `-5.4` for C3. Mean paired 5M R changes were `-0.5447`, `-0.9598`, and `-0.9662`, respectively. The ordered test supports a dose trend, while the near tie between C2 and C3 indicates saturation at the available checkpoint resolution rather than a strictly separated response at every adjacent dose.

This supports the registered interpretation that increasing cue dose changes the probability or timing of state formation. Because changing lambda also changes other joint-distribution properties, this is a causal statement about the registered full-support cue distributions—not about the scalar information index in isolation.

## Matched-index geometry

All three geometry conditions target the same cue index `0.065092` but use different `m` and solved lambda values.

- Concentrated minus `m=64` onset ordinals were `0, 2, 1, 1, 1` (mean `+1.0` checkpoint), even though mean 5M R differed by only `+0.00094` and mean 10M R by `-0.00172`.
- Diffuse minus `m=64` onset ordinals were `1, 0, 0, -1, 1` (mean `+0.2`), while mean 5M R was `+0.04236` with marked seed variation and mean 10M R was `+0.00450`.

The concentrated condition's consistent onset delay is strong evidence that cue concentration/spectral geometry affects timing beyond the scalar cue index. The diffuse condition is closer in onset timing but more heterogeneous in intermediate R. Consequently, the dose and geometry results jointly support a causal role for the cue-bearing distribution, but not a claim that singleton mutual information alone is sufficient to specify the effect.

## Cue removal and persistence

At exactly 5M, each run was evaluated, only the sampler was switched to lambda zero, and the model was evaluated again with zero optimizer steps. In all 30 cells:

- model-state hashes were unchanged;
- optimizer-state hashes were unchanged;
- full-support output hashes were identical;
- RNG and absolute exposure state were preserved before continuation.

All pre-switch threshold-positive nonzero-cue cells remained positive at both 6M and 10M. By 10M, all 25 nonzero-cue cells met establishment. Most R values continued to decrease after cue removal; one diffuse seed increased from `0.01327` at 5M to `0.03020` at 10M but remained well inside the registered establishment thresholds. This is evidence that the cue bootstrapped a full-support-compatible predictive representation rather than a solution dependent on continued cue-bearing sampling.

## Gradient and probe correlates

Initialization audits are identical across conditions within each paired seed, as required by the shared initialization and fixed independent full-support audit batches. At 4M, mean global gradient SNR / mean pairwise cosine were:

| Condition | Mean global SNR | Mean pairwise cosine |
|---|---:|---:|
| C0 | 0.0396 | 0.0073 |
| C1 | 66.9249 | 0.9557 |
| C2 | 8.8981 | 0.8928 |
| C3 | 5.2881 | 0.8210 |
| Concentrated | 3.1919 | 0.7104 |
| Diffuse | 14.7019 | 0.9289 |

These statistics co-occur with the behavioral/probe transitions but are not monotone in dose once models occupy different learned states. They are secondary mechanism correlates and are not interpreted as causal mediators. Each of the 150 audits contains mean-gradient norm, variance, SNR, cosine, and embedding, direct-output, Q, K, V, MLP, and normalization groups. The 210 probe reports contain linear, fixed-Fourier, and fixed-capacity MLP metrics for layers 0, 3, and 6.

## Artifact and stop audit

- 30/30 run directories complete.
- 330/330 checkpoints have valid SHA-256 sidecars.
- 360 exact full-support evaluations are finite and use 2,000 first-parent-row bootstrap replicates.
- 210 selected-checkpoint probe reports pass schema validation.
- 150 no-training gradient audits pass schema validation and record zero optimizer steps.
- 30/30 zero-step switch invariance records pass.
- Maximum exposure is 10M pairs.

Stopped as registered. No recurrent bridge, field-size sweep, Fourier initialization, new hard-support mask, optimizer change, production trajectory, or extension beyond 10M was run.

