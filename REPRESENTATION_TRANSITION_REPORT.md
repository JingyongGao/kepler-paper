# Representation Transition Report

## Scope

This post-submission audit uses only the frozen v3.6 confirmatory result JSON and existing probe reports. No model was trained, no optimizer step was executed, and no threshold was changed.

## Registered predictive-state timing

| Condition | Verified onset trajectories | Median verified onset | Establishment observations |
|---|---:|---:|---:|
| R0 | 1/8 | 7500000.0 | 2/8 |
| R1 | 6/8 | 4000000.0 | 6/8 |
| R2 | 8/8 | 4000000.0 | 8/8 |

The frozen trajectories are consistent with cue-associated earlier predictive-state accessibility: all R2 blocks have verified onset, most R1 blocks do, and only one R0 block has a retained onset within 10M exposures.

## Layer and probe dynamics

Layer-wise linear accuracy, fixed-Fourier explained variance, and fixed-capacity MLP accuracy are reported at layers 0, 3, and 6. The first reliable decoding layer is the shallowest measured layer whose linear accuracy exceeds the already registered 0.5 criterion and remains above it at the next registered probe checkpoint. Depth is normalized by six Transformer blocks.

MLP accessibility often rises with or slightly ahead of linear accessibility at the same checkpoint, but this is descriptive and does not identify a causal mediator. The continuous Fourier R2 curves rise across the transition region in learned trajectories.

## Fourier classification constraint

No registered Fourier R2 threshold exists in the frozen manifest or probe method. Therefore a Fourier emergence checkpoint is not classified; the result is recorded as `unclassifiable_no_registered_threshold`. This audit does not introduce a post-hoc threshold, so it cannot formally order Fourier emergence relative to the registered R decrease.

## Answers

1. Cue level is associated with earlier predictive-state onset across the paired R0/R1/R2 trajectories.
2. Whether Fourier structure emerges before R decreases is not formally orderable without a registered Fourier threshold.
3. One R0 trajectory follows a late version of the joint R/probe pathway, and two reach establishment by 10M, but the pattern is not universal across R0.

All claims concern representation accessibility and transition dynamics in these frozen trajectories only.
