# ICLR 2027 scale-robustness pilot report

Decision: **GO**

The four pilot cells are preregistered members of the 16-cell extension. The remaining twelve trajectories were not started during this gate. Cue-effect direction and magnitude were not used for Go/No-Go.

| Scale | Engineering | Learnability/identifiability | Decision |
|---|---:|---:|---:|
| 1m | pass | pass | GO |
| 5m | pass | pass | GO |

## Pilot cells

| Run | Median pairs/s | Peak MPS driver bytes | Onset | Establishment |
|---|---:|---:|---:|---:|
| `scale_1m_rule_0_init_0_R0` | 29267.2 | 1289781248 | None | None |
| `scale_1m_rule_0_init_0_R2` | 27041.8 | 1286029312 | 4000000 | 5000000 |
| `scale_5m_rule_0_init_0_R0` | 11428.7 | 2361868288 | None | None |
| `scale_5m_rule_0_init_0_R2` | 11421.3 | 2361868288 | 3000000 | 4000000 |

The gate requires no favorable R2 effect. If any scale is not learnable, not time-resolved, or unreliable, the protocol stops without architecture or optimizer changes.

Machine-readable gate: `artifacts/iclr2027_scale_robustness_pilot_gate.json`
