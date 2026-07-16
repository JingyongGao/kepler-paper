# Project Kepler v3.3 — Checkpoint Gradient-Dynamics Audit

## Protocol

This is a no-training audit over all 19 registered M1–M4 checkpoints.  No
optimizer step was taken.  Each checkpoint was evaluated on the same 32 fixed,
independent exact-soft-label batches within its task; these batches are disjoint
from training, evaluation, and probe streams.  Recurrent checkpoints used eight
Episodes per batch (512 scored positions), and parent-pair checkpoints used 512
uniform pairs per batch.

For each semantic group and Transformer block the audit reports:

- `norm_of_mean_gradient = ||mean(g)||`;
- `gradient_variance = mean(||g_i-mean(g)||^2)`;
- `gradient_snr = ||mean(g)||^2 / gradient_variance`;
- the mean cosine over all unordered pairs of per-batch gradient vectors;
- mean and RMS batch-gradient norms.

The tied Transformer embedding was decomposed into its direct output-head
gradient and its input-embedding contribution.  Fused QKV parameters were split
into Q, K, and V slices.  M4 has no attention, so Q/K/V are explicitly marked
not applicable.  The complete per-group/per-block values and paired R/probe
summaries are in `artifacts/checkpoint_gradient_dynamics_audit.json` (SHA-256
`4a45b5cd539b25de966e814ef22d4287aed05af1f8ada1979a02c9d3f4183cfb`).

## Checkpoint summary

The compact table shows SNR; the JSON contains mean-gradient norm, variance,
pairwise cosine, and batch-gradient norms for every corresponding cell.

| run | exposure | R | embedding | output | Q | K | V | MLP | norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.5M | 0.997766 | 0.0436 | 0.0443 | 0.0402 | 0.0395 | 0.0474 | 0.0429 | 0.0496 |
| M1 | 1M | 0.961786 | 0.1033 | 0.0910 | 0.0850 | 0.0862 | 0.1271 | 0.0812 | 0.2140 |
| M1 | 2M | 0.487708 | 0.0934 | 0.6538 | 0.1244 | 0.1285 | 0.9511 | 0.6346 | 2.968 |
| M1 | 5M | 0.008324 | 0.8391 | 1.736 | 1.101 | 0.4683 | 1.835 | 2.991 | 3.320 |
| M1 | 10M | 0.004575 | 0.9108 | 2.011 | 0.7711 | 0.5761 | 1.700 | 2.414 | 2.706 |
| M2 | 5M | 1.000612 | 0.1109 | 0.0462 | 0.0317 | 0.0309 | 0.0593 | 0.0545 | 0.0510 |
| M2 | 10M | 1.000339 | 0.0203 | 0.0346 | 0.0314 | 0.0282 | 0.0392 | 0.0386 | 0.0664 |
| M2 | 20M | 1.000155 | 0.0494 | 0.0293 | 0.0444 | 0.0435 | 0.0440 | 0.0317 | 0.0354 |
| M2 | 40M | 1.000008 | 0.0324 | 0.0277 | 0.0353 | 0.0522 | 0.0253 | 0.0261 | 0.0276 |
| M3 | 0.5M | 1.000588 | 0.0500 | 0.0488 | 0.0691 | 0.0704 | 0.0452 | 0.0486 | 0.0495 |
| M3 | 1M | 1.000354 | 0.0603 | 0.0450 | 0.0458 | 0.0446 | 0.0643 | 0.0661 | 0.0596 |
| M3 | 2M | 1.000283 | 0.0573 | 0.0419 | 0.0430 | 0.0489 | 0.0553 | 0.0517 | 0.0502 |
| M3 | 5M | 0.999159 | 0.0843 | 0.0417 | 0.0857 | 0.0680 | 0.0886 | 0.0462 | 0.0566 |
| M3 | 10M | 0.462319 | 90.82 | 0.2384 | 30.58 | 36.32 | 25.19 | 44.69 | 129.8 |
| M4 | 0.5M | 1.000072 | 0.0323 | 0.0354 | n/a | n/a | n/a | 0.0381 | 0.0376 |
| M4 | 1M | 1.000149 | 0.0324 | 0.0374 | n/a | n/a | n/a | 0.0401 | 0.0397 |
| M4 | 2M | 1.000317 | 0.0325 | 0.0428 | n/a | n/a | n/a | 0.0476 | 0.0469 |
| M4 | 5M | 1.000653 | 0.0332 | 0.0493 | n/a | n/a | n/a | 0.0617 | 0.0616 |
| M4 | 10M | 1.000348 | 0.0322 | 0.0453 | n/a | n/a | n/a | 0.0518 | 0.0521 |

## Relation to behavior and probes

- M1 begins near `R=1` with uniformly low SNR.  At 2M, when `R=0.488`, its
  output/V/MLP/norm SNRs have increased; by 5M–10M, `R<0.01` and the selected
  layer-6 linear probe reaches accuracy 1.0.  Coherent exact-soft gradients and
  predictive-state decoding coexist at the learned checkpoints.
- M2 remains at `R≈1` through 40M.  Its semantic-group SNRs remain low and its
  selected layer-6 linear/MLP probes remain at chance.  Short-span recurrence
  did not produce either a decodable state or a coherent population gradient.
- M3 remains at baseline through 5M, then at 10M reaches `R=0.462` with layer-6
  linear/MLP probe accuracies `0.229/0.305`.  At that checkpoint embedding,
  Q/K/V, MLP, and normalization SNRs increase sharply.  The direct tied output
  contribution rises much less (`0.238`), so most gradient coherence is inside
  the composition pathway rather than the direct output-weight slice.
- M4 stays near `R=1`; embedding/output/MLP/norm SNRs remain low at every
  checkpoint.  Q/K/V are structurally absent, not zero-valued measurements.

These are checkpoint associations, not causal estimates.  A learned
representation can itself alter the population-gradient geometry, and the M1
trajectory already showed that low initialization SNR does not establish
unlearnability.  The audit therefore does not claim that an SNR increase caused
learning, or that low SNR alone caused failure.
