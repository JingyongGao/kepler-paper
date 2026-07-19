# Early Cue-Removal Report

## Result

The frozen 3M and 4M R2 checkpoints permit an immediate no-step sampler-switch boundary description, but no true continuation trained from either early switch exists. Their continuation fields are therefore marked `unavailable`; later checkpoints are not repurposed because they were trained with cue-bearing sampling until 5M.

At 5M, all 8 R2 blocks have observed pre/post boundary evaluations and true frozen continuation checkpoints at 6M, 7.5M, and 10M. Model outputs are identical across all 8 no-step boundary comparisons. The registered onset state is retained in 8/8 blocks at 6M and 8/8 at 10M.

## Availability table

| Removal time | Immediate frozen state | True continuation | Interpretation |
|---|---|---|---|
| 3M | Observed | Unavailable | No persistence claim |
| 4M | Observed | Unavailable | No persistence claim |
| 5M | Observed pre/post | Observed through 10M | Probe and R retention can be reported |

The missing 3M/4M continuations are not imputed or invented.
