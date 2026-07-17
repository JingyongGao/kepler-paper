# Cue-removal persistence

Retention is evaluated only among cells positive at the 5M pre-switch checkpoint.

| Version | Status | Condition | Positive at 5M pre | Retained at 6M | Retained at 10M | Fraction at 6M | Fraction at 10M | Sampler after 5M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3.5 | exploratory | C0 | 0 | — | — | — | — | uniform_full |
| v3.5 | exploratory | C1 | 2 | 2 | 2 | 1 | 1 | uniform_full |
| v3.5 | exploratory | C2 | 5 | 5 | 5 | 1 | 1 | uniform_full |
| v3.5 | exploratory | C3 | 5 | 5 | 5 | 1 | 1 | uniform_full |
| v3.5 | exploratory | geometry_concentrated | 5 | 5 | 5 | 1 | 1 | uniform_full |
| v3.5 | exploratory | geometry_diffuse | 5 | 5 | 5 | 1 | 1 | uniform_full |
| v3.6 | confirmatory | R0 | 0 | — | — | — | — | uniform_full |
| v3.6 | confirmatory | R1 | 5 | 5 | 5 | 1 | 1 | uniform_full |
| v3.6 | confirmatory | R2 | 8 | 8 | 8 | 1 | 1 | uniform_full |
