# Project Kepler v3.2.2 — Learnability Audit Report

Date: 2026-07-16 (Asia/Shanghai)

## Executive conclusion

Runs A, B, and C completed at their registered limits and all training has
stopped. No remaining micro-pilot or production trajectory was launched; no run
was extended beyond its manifest.

The audit rejects a global metric, indexing, checkpoint, or optimizer failure.
The single-parent positive control learns the center state and calibrates the
noise distribution. In contrast, both the 10M sampled-label `(k=2,d=4)` run and
the 2M oracle-soft-label `(k=2,d=4)` run remain uniform, and their linear,
Fourier, and fixed-capacity MLP probes stay at chance across residual layers.

The supported diagnosis is therefore: **the two-parent modular sufficient state
did not form in the tested models and budgets**. The results do not support
“state formed but output ignored,” “noise calibration alone failed,” a capacity
threshold between 2M and 10M, or sampled-label variance as the primary cause.
They do not distinguish a representation/inductive-bias bottleneck from a
task-specific optimization barrier beyond the tested settings.

Every audit run is exploratory, diagnostic-only, and excluded from the
confirmatory fit.

## Frozen baseline and integrity

- Baseline commit: `a5d98f98bdcd4ccf891d662c7630fc795aff4563`
- Annotated tag: `kepler-v3.2.2-rc1`
- Production manifest SHA-256:
  `4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10`
- Phase-4 pilot manifest SHA-256:
  `3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0`
- Learnability audit manifest SHA-256:
  `95a21c5a89af958ae986b3d3284c9e610bd37c16a0d3e68dc2d01bd83c9f5af4`

The two existing manifests were hash-checked before and after the audit and were
not modified. Run A's `(k=1,d=1)` topology exists only in the diagnostic
manifest and was not added to the production factorial design.

## No-training and indexing controls

| Control | Result |
|---|---:|
| Uniform logits | `R = 1.0000000000000007` |
| Analytic `q_h` oracle logits | `R = 0.0` |
| `z_t` recomputed from `h_(t-1)` | 0 mismatches / 2,048 positions |
| Deliberate `z_(t-1)` oracle | `R = 1.485713` (rejected) |
| Deliberate `z_(t+1)` oracle | `R = 1.486452` (rejected) |
| Pure metric batching difference | `0.0` |
| Checkpoint batch-size difference | `6.38e-8` |
| Checkpoint reload difference | `0.0` |

The isolated fixed-batch model reduced memorization NLL from `5.585394` to
`0.0000207`, but on 32 independent fresh Episodes it obtained NLL `10.06836`
and `R = 1.97171`. Near-zero memorization loss therefore does not imply rule
generalization.

## Audit manifest and completion status

| Run | Objective | Model/topology | Checkpoints | Status |
|---|---|---|---|---|
| A | sampled-token NLL | 2M, `(k=1,d=1)`, `a=[94]` | 0.5/1/2/5/10M | complete |
| B | sampled-token NLL | 10M, `(k=2,d=4)`, `a=[45,36]` | 5/10/20/40M | complete |
| C | exact oracle-soft CE | 2M, `(k=2,d=4)`, `a=[45,36]` | 0.5/1/2/5/10M | complete |

All objectives used online independent Episodes. Evaluation and probe Episodes
were fixed and independent from both training and each other.

## Checkpoint evaluation metrics

`NLL_train` is sampled-token NLL on the checkpoint-producing training batch,
including for Run C. `NLL_fresh` is sampled-token NLL on fresh fixed Episodes.
Every confidence interval is a 2,000-resample Episode bootstrap over 256 Episodes.

### Run A — single-parent positive control

| D | R (95% CI) | Center top-1 | Mean `p(z_t)` | Predictive H | NLL_train | NLL_fresh |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5M | 0.573924 [0.573062, 0.574903] | 1.0000 | 0.036763 | 5.443629 | 3.493348 | 3.555671 |
| 1M | 0.189782 [0.189294, 0.190281] | 1.0000 | 0.275210 | 4.549265 | 1.906527 | 1.756814 |
| 2M | 0.005481 [0.005449, 0.005516] | 1.0000 | 0.863291 | 1.131151 | 0.996275 | 0.890650 |
| 5M | 0.005481 [0.005454, 0.005509] | 1.0000 | 0.891737 | 0.916205 | 0.901421 | 0.890191 |
| 10M | 0.003463 [0.003451, 0.003475] | 1.0000 | 0.895323 | 0.899332 | 0.934162 | 0.880989 |

Run A first learns the center identity, then calibrates its probability and
entropy toward the oracle values `p(z_t)=0.9` and `H=0.879601` nats.

### Run B — capacity-threshold calibration

| D | R (95% CI) | Center top-1 | Mean `p(z_t)` | Predictive H | NLL_train | NLL_fresh |
|---:|---:|---:|---:|---:|---:|---:|
| 5M | 1.000926 [1.000677, 1.001171] | 0.003723 | 0.003887 | 5.545683 | 5.557244 | 5.553460 |
| 10M | 1.000217 [1.000080, 1.000360] | 0.003235 | 0.003892 | 5.547880 | 5.548684 | 5.550353 |
| 20M | 1.000031 [0.999947, 1.000110] | 0.003235 | 0.003892 | 5.548700 | 5.549179 | 5.549209 |
| 40M | 1.000017 [0.999994, 1.000040] | 0.003601 | 0.003891 | 5.549047 | 5.549675 | 5.549196 |

The center accuracy/probability remain at the uniform baselines `1/257`, and
entropy remains at `log(257)`. No 10M capacity transition occurred by 40M.

### Run C — oracle-soft-label diagnostic

| D | R (95% CI) | Center top-1 | Mean `p(z_t)` | Predictive H | NLL_train | NLL_fresh |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5M | 1.000524 [1.000319, 1.000744] | 0.003906 | 0.003892 | 5.546496 | 5.553215 | 5.551427 |
| 1M | 1.000367 [1.000209, 1.000530] | 0.004028 | 0.003890 | 5.547525 | 5.552191 | 5.551016 |
| 2M | 1.000408 [1.000239, 1.000568] | 0.003418 | 0.003890 | 5.547478 | 5.547988 | 5.550968 |
| 5M | 1.000236 [1.000059, 1.000411] | 0.004028 | 0.003894 | 5.547416 | 5.548816 | 5.550046 |
| 10M | 1.000239 [1.000114, 1.000365] | 0.003418 | 0.003890 | 5.548265 | 5.548944 | 5.550395 |

Removing sampled-label variance did not change the uniform solution.

## Per-block gradient and relative update diagnostics

Vector order is `embedding / block0 / block1 / block2 / block3 / block4 /
block5 / final_norm`. Update values are `||Delta theta|| / ||theta||` for the
checkpoint-producing optimizer step.

### Run A

| D | Per-block gradient norm | Relative update norm |
|---:|---|---|
| 0.5M | `1.38/2.11/1.43/0.98/0.737/0.557/0.474/0.153` | `9.50e-4/3.15e-4/2.73e-4/3.05e-4/3.42e-4/3.86e-4/4.29e-4/2.87e-5` |
| 1M | `0.970/1.28/1.22/1.07/0.923/0.784/0.575/0.206` | `1.49e-3/3.09e-4/2.56e-4/2.96e-4/3.24e-4/3.61e-4/4.46e-4/5.61e-5` |
| 2M | `0.406/0.571/0.466/0.477/0.482/0.474/0.409/0.0191` | `7.54e-4/7.68e-4/7.56e-4/7.59e-4/7.60e-4/7.65e-4/7.58e-4/8.24e-5` |
| 5M | `0.201/0.168/0.114/0.110/0.111/0.116/0.115/0.00982` | `1.56e-3/1.90e-3/1.89e-3/1.87e-3/1.84e-3/1.82e-3/1.79e-3/8.00e-5` |
| 10M | `0.242/0.138/0.0702/0.0686/0.0735/0.0816/0.0841/0.0134` | `1.69e-3/2.01e-3/2.07e-3/2.00e-3/1.95e-3/1.91e-3/1.88e-3/9.32e-5` |

### Run B

| D | Per-block gradient norm | Relative update norm |
|---:|---|---|
| 5M | `0.694/0.00751/0.00958/0.0104/0.0103/0.0106/0.0107/0.0187` | `3.40e-3/2.05e-3/2.08e-3/2.05e-3/2.05e-3/2.04e-3/2.04e-3/1.41e-4` |
| 10M | `0.489/0.00156/0.00286/0.00293/0.00320/0.00348/0.00346/0.0126` | `3.87e-3/2.43e-3/2.00e-3/1.92e-3/1.89e-3/1.86e-3/1.86e-3/1.64e-4` |
| 20M | `0.145/0.00126/0.000618/0.000620/0.000714/0.000859/0.000775/0.00612` | `2.80e-3/1.46e-3/1.70e-3/1.60e-3/1.57e-3/1.55e-3/1.55e-3/2.39e-4` |
| 40M | `0.00648/0.000264/0.000189/0.000147/0.000182/0.000218/0.000200/0.0137` | `2.65e-3/9.51e-4/1.03e-3/9.03e-4/8.95e-4/8.94e-4/9.04e-4/7.22e-4` |

### Run C

| D | Per-block gradient norm | Relative update norm |
|---:|---|---|
| 0.5M | `0.701/0.227/0.210/0.206/0.211/0.204/0.201/0.0226` | `4.12e-4/2.06e-4/2.10e-4/2.12e-4/2.12e-4/2.11e-4/2.12e-4/1.43e-5` |
| 1M | `0.722/0.107/0.124/0.114/0.111/0.112/0.112/0.0212` | `7.92e-4/4.32e-4/4.16e-4/3.89e-4/3.81e-4/3.74e-4/3.75e-4/2.42e-5` |
| 2M | `0.466/0.0210/0.0298/0.0298/0.0310/0.0307/0.0305/0.0117` | `1.43e-3/7.89e-4/7.19e-4/6.70e-4/6.57e-4/6.59e-4/6.50e-4/4.53e-5` |
| 5M | `0.256/0.00319/0.00473/0.00624/0.00632/0.00651/0.00583/0.00760` | `5.08e-3/3.25e-3/2.88e-3/2.52e-3/2.23e-3/2.05e-3/1.92e-3/1.32e-4` |
| 10M | `0.287/0.00130/0.00271/0.00281/0.00346/0.00348/0.00301/0.0103` | `3.57e-3/2.00e-3/1.75e-3/1.67e-3/1.60e-3/1.58e-3/1.56e-3/1.56e-4` |

Parameters are updating in every run. B and C nevertheless converge toward a
uniform predictive function, so zero learning cannot be attributed to frozen
blocks.

## Predictive-state probes

Each cell is `linear accuracy / fixed-Fourier R^2 / fixed-capacity-MLP accuracy`
on a 1,024-example independent test split. Chance classification accuracy is
approximately `0.00389`. Layers are embedding output 0 and residual blocks 3/6.

### Run A

| D | Layer 0 | Layer 3 | Layer 6 |
|---:|---:|---:|---:|
| 0.5M | `0.989/0.110/1.000` | `1.000/0.211/1.000` | `1.000/0.241/1.000` |
| 2M | `1.000/0.232/1.000` | `1.000/0.309/1.000` | `1.000/0.422/1.000` |
| 10M | `1.000/0.273/1.000` | `1.000/0.513/1.000` | `1.000/0.612/1.000` |

### Run B

| D | Layer 0 | Layer 3 | Layer 6 |
|---:|---:|---:|---:|
| 5M | `0.00586/-0.00165/0.00684` | `0.00488/-0.00498/0.00098` | `0.00488/-0.00579/0.00098` |
| 20M | `0.00391/-0.00182/0.00586` | `0.00098/-0.0262/0.00098` | `0.00684/-0.0616/0.00195` |
| 40M | `0.00098/-0.00218/0.00684` | `0.00195/-0.135/0.00195` | `0.00391/-0.338/0.00293` |

### Run C

| D | Layer 0 | Layer 3 | Layer 6 |
|---:|---:|---:|---:|
| 0.5M | `0.00879/-0.00195/0.00391` | `0.00293/-0.00558/0.00391` | `0.00195/-0.00673/0.00391` |
| 2M | `0.00879/-0.00193/0.00391` | `0.00391/-0.00457/0.00391` | `0.00586/-0.00535/0.00684` |
| 10M | `0.00586/-0.00197/0.00293` | `0.00293/-0.00768/0.00293` | `0.00586/-0.0117/0.00391` |

## Interpretation by preregistered failure mode

| Failure mode | Audit conclusion | Evidence |
|---|---|---|
| No state representation formed | **Supported for B and C** | All three probe families remain at chance/non-positive R^2 through the final residual block. |
| State formed but output distribution did not use it | Not supported | No hidden-state probe signal exists to be ignored by the output. |
| Center learned but noise calibration failed | Transiently observed only in early A; not B/C | A has center top-1 1.0 by 0.5M while `p(z)` and entropy calibrate later; B/C never learn the center. |
| Capacity threshold | Not observed from 2M to 10M | B remains uniform at 10M width and 40M tokens. Thresholds outside tested capacity are not assessed. |
| Sampled-label gradient-variance limitation | Not supported as primary cause | C uses exact `q_h` targets and still remains uniform with chance probes. |
| General optimization or indexing failure | Rejected globally | Analytic controls pass, A learns online and calibrates, and fixed-batch optimization succeeds. A task-specific composition/inductive-bias barrier remains possible. |

No conclusion depends on visual curve separation or an ordering among production
topologies. No production setting is changed based on effect direction.

## Engineering statistics

| Run | Steps | Median token/s | Throughput CV | Peak current MPS | Peak driver MPS |
|---|---:|---:|---:|---:|---:|
| A | 4,886 | 27,428 | 4.86% | 52.9 MB | 1.174 GB |
| B | 19,533 | 11,634 | 2.13% | 659.9 MB | 2.305 GB |
| C | 4,886 | 27,648 | 3.65% | 61.1 MB | 1.174 GB |

All checkpoint and evaluation metrics were finite. Checkpoints used the durable
fsync/load/checksum/atomic-replace path. MPS fallback remained disabled.

## Stop-condition compliance

- Runs A, B, and C completed: yes.
- `p=17` run launched: no.
- Remaining Phase-4 micro-pilot launched: no.
- 80-trajectory production matrix launched: no.
- Existing trajectory extended beyond its manifest: no.
- Production or Phase-4 pilot manifest modified: no.
- Global optimizer setting changed: no.

The learnability audit is complete and training has stopped.
