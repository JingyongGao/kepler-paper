# Project Kepler v3.2.2 — Phase-4 engineering micro-pilot report

Date: 2026-07-15 (Asia/Shanghai)

## Outcome

The Phase-4 task interface, independent pilot manifest, environment lock,
durable checkpoint path, Episode bootstrap evaluation, and resume audit are
implemented and operational. The engineering soak passed its intended memory,
throughput, checkpoint, and finite-metric checks.

The learning-range acceptance criterion did **not** pass. The preregistered
2M calibration trajectory remained at `R ~= 1` from 5M through 40M supervised
tokens. This is an identifiability failure for the tested configuration/range,
not evidence for or against a topology effect. The remaining long trajectories
were not launched after the preselected calibration trajectory reached its 40M
endpoint. All pilot data are exploratory and excluded from the primary
confirmatory fit.

## Files changed

- `src/kepler/environment.py`: pre-Torch MPS environment lock and provenance
- `src/kepler/model.py`: optional per-sequence task embedding and migration checks
- `src/kepler/phase4.py`: durable base-checkpoint task attachment
- `src/kepler/pilot.py`: independent 15-trajectory pilot manifest and runner
- `src/kepler/durability.py`: fsync/load/checksum/atomic checkpoint publication
- `src/kepler/training.py`: complete RNG/scheduler state, throughput/memory, Episode KL
- `src/kepler/audit.py`: deterministic, forced-resume, and overfit diagnostics
- `src/kepler/cli.py`: Phase-4 manifest/run/status/audit commands
- `src/kepler/config.py`, `topology.py`, `fisher.py`: optional task configuration,
  paired distinct rule, and registered five-arm mapping
- `tests/test_phase4.py`: task, manifest, parameter-count, environment, and durability tests
- `README.md`: Phase-4 operating instructions

The existing `artifacts/manifest.json` was not regenerated or modified.

## Tests

`uv run pytest`: **20 passed**.

The new tests cover disabled-conditioner equivalence, zero Task-A/Task-B logits,
scalar and batch broadcasting, invariant sequence/lag geometry, unused-row zero
gradient, checkpoint round trip, production-manifest hash stability, unchanged
Phase 2/3 core counts, pilot composition, distinct paired rules, and durable
checkpoint validation/checksum/reload.

## Manifests and hashes

| Manifest | Contents | SHA-256 |
|---|---|---|
| Production | Existing 80 trajectories; unchanged | `4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10` |
| Phase-4 pilot | 10 x 2M long, 4 x 10M long, 1 x 50M soak | `3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0` |

The 50M engineering soak topology was preregistered as `(k=4,d=64)` and its
endpoint fixed at the protocol-permitted lower bound of 1M supervised tokens.

## Trajectory status

| Trajectory | Tier | `(k,d)` | Status | Completed / target |
|---|---:|---:|---|---:|
| `pilot_2m_long_k2_d4_n2m_r0_s0` | 2M | (2,4) | complete | 40M / 40M |
| `pilot_2m_long_k2_d4_n2m_r1_s0` | 2M | (2,4) | pending | 0 / 40M |
| `pilot_2m_long_k4_d4_n2m_r0_s0` | 2M | (4,4) | pending | 0 / 40M |
| `pilot_2m_long_k4_d4_n2m_r1_s0` | 2M | (4,4) | pending | 0 / 40M |
| `pilot_2m_long_k2_d64_n2m_r0_s0` | 2M | (2,64) | pending | 0 / 40M |
| `pilot_2m_long_k2_d64_n2m_r1_s0` | 2M | (2,64) | pending | 0 / 40M |
| `pilot_2m_long_k4_d64_n2m_r0_s0` | 2M | (4,64) | pending | 0 / 40M |
| `pilot_2m_long_k4_d64_n2m_r1_s0` | 2M | (4,64) | pending | 0 / 40M |
| `pilot_2m_long_k3_d16_n2m_r0_s0` | 2M | (3,16) | pending | 0 / 40M |
| `pilot_2m_long_k3_d16_n2m_r1_s0` | 2M | (3,16) | pending | 0 / 40M |
| `pilot_10m_long_k2_d4_n10m_r0_s0` | 10M | (2,4) | pending | 0 / 40M |
| `pilot_10m_long_k4_d4_n10m_r0_s0` | 10M | (4,4) | pending | 0 / 40M |
| `pilot_10m_long_k2_d64_n10m_r0_s0` | 10M | (2,64) | pending | 0 / 40M |
| `pilot_10m_long_k4_d64_n10m_r0_s0` | 10M | (4,64) | pending | 0 / 40M |
| `pilot_50m_soak_k4_d64_n50m_r0_s0` | 50M | (4,64) | complete | 1M / 1M |

No production trajectory was launched.

## Exact conditional-KL results

Every row uses the same 256 fixed independent evaluation Episodes for its
trajectory and a deterministic 2,000-resample Episode bootstrap.

| Trajectory | `D` | `R` | Episode-bootstrap 95% CI |
|---|---:|---:|---:|
| 2M `(2,4)`, rule 0 | 5M | 1.000530 | [1.000352, 1.000706] |
| 2M `(2,4)`, rule 0 | 10M | 1.000146 | [1.000045, 1.000249] |
| 2M `(2,4)`, rule 0 | 20M | 1.000064 | [1.000010, 1.000119] |
| 2M `(2,4)`, rule 0 | 40M | 1.000030 | [0.999985, 1.000076] |
| 50M soak `(4,64)`, rule 0 | 1M | 1.002052 | [1.001618, 1.002485] |

All Episode contributions are retained in the corresponding
`checkpoint_*.eval.json` files. All values are finite. Because every observed
`R` is near 1, the tested `D` range does not provide a non-saturated learning
range.

## Throughput and memory

`torch.mps.synchronize()` brackets every measured optimizer step.

| Run | Steps | Median tokens/s | Mean tokens/s | CV | Peak current MPS | Peak driver MPS |
|---|---:|---:|---:|---:|---:|---:|
| 2M long `(2,4)` | 19,533 | 21,746 | 22,395 | 10.51% | 51.7 MB | 1.174 GB |
| 50M soak `(4,64)` | 489 | 3,074 | 3,057 | 3.81% | 1.292 GB | 3.470 GB |

The reported MPS recommended maximum was 26.801 GB. No cumulative memory growth,
NaN, Inf, or fallback error was observed. These statements apply only to the
tested operator graphs and configurations; they do not establish universal MPS
correctness.

## Checkpoint, attachment, and resume verification

- Forced interruption at 64 tokens and resume to 128 tokens: passed.
- Exact token accounting: passed.
- Scheduler continued at the absolute-token learning rate: passed.
- Online Episode-generator state matched an independently advanced reference: passed.
- Both forced-resume checkpoint checksums: valid.
- Immediate post-reload exact evaluation: finite (`R=1.004954`).
- Phase-4 attachment permitted only missing `task_emb.weight`; Task A and Task B
  rows were zero and base/A/B maximum logit difference was `0.0` before and after reload.
- Phase-4 attachment checksum:
  `4519281fd183facfbc9ea6e52976d084d41a62bb70a8b1ac8a4b2734923f5140`.

The isolated repeated-batch optimizer diagnostic reduced NLL from `5.58539` to
`0.0000207` in 500 updates. It is explicitly excluded from all scientific fits;
it shows that the update graph can optimize, while the online generalization task
did not become learnable in the tested 5M–40M range.

## Environment and deviations

- macOS: 26.5.2 arm64
- Python: 3.13.13
- PyTorch: 2.13.0
- `PYTORCH_MPS_FAST_MATH=0`
- `PYTORCH_ENABLE_MPS_FALLBACK=0`
- Complete config hash:
  `d9db2cfa76d437c11c8383db6c5daaa71f4e79f8ce7d71743c588d0d3ed1dcbf`
- Git commit: unavailable because the workspace is not a Git repository.
- Strict deterministic algorithms: unsupported on this MPS graph because
  `index_put_with_accumulate_mps` has no deterministic implementation. The
  production setting was restored; no `warn_only` mode or silent fallback was used.

## Required next decision

Do not interpret the current curves as a null topology result. Before running the
remaining exploratory matrix, preregister a global, topology-independent
identifiability adjustment. A defensible next calibration is to keep the already
selected `(k=2,d=4)`, rule-0 trajectory and test extended absolute-token endpoints
(for example 80M and 160M) or a globally fixed optimizer/schedule change. That
calibration cohort must remain exploratory and excluded from the primary fit. Only
after an `R` interval enters a non-saturated region should the final pilot range be
frozen and applied uniformly across topologies.
