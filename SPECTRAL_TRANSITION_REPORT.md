# Spectral Transition Report

## Scope and method

This is CPU-only post-processing of six frozen checkpoints from the paired block `rule_0_permutation_1_init_0`. Every checkpoint uses the same 512 fixed full-support parent pairs, split into four evaluation batches, and the same exact conditional soft cross-entropy objective. Optimizer state is not loaded and no optimizer step is executed.

Maximum and top-k eigenvalues are Lanczos Ritz approximations; trace is a fixed-seed Hutchinson estimate. Effective spectral dimension is computed from the reported largest k Ritz values.

## Spectral signatures associated with transition checkpoints

| Condition | Role | Exposure | lambda_max | Trace estimate | d_eff top-10 |
|---|---|---:|---:|---:|---:|
| R0 | pre_transition | 6,000,000 | 143.058 | 1418.77 | 6.414 |
| R0 | transition | 7,500,000 | 6337.52 | 64422.7 | 5.828 |
| R0 | late | 10,000,000 | 2596.69 | 22164.1 | 6.354 |
| R2 | pre_transition | 3,000,000 | 3155.68 | 75057.1 | 8.486 |
| R2 | transition | 4,000,000 | 9993.67 | 194199 | 8.187 |
| R2 | post_transition | 5,000,000 | 10634.2 | 194973 | 8.146 |

These quantities are spectral signatures associated with the selected transition checkpoints. They do not prove a mechanism, do not establish gradients as causal mediators, and do not show that the cue directly changes the loss landscape.

The calculation is limited to one representative paired block and one fixed evaluation sample; it is not a universal Transformer claim.
