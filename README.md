# Project Kepler v3.2.2

Project Kepler tests whether dependency span `d` and fan-in `k` alter Transformer
capacity scaling, online absorption, predictive-state formation, and local
functional adaptation when marginal entropy, entropy rate, and predictable
information per token are held fixed.

The implementation treats the specification as a preregistration. The finite
field is fixed at `p=257`, noise at `eta=0.1`, logarithms are natural, topology is
the five-point `(k,d)` matrix, and all scaling-model parameters are represented in
positive coordinates.

## Frozen experimental conventions

- A rule is `x_t = sum_j a_j x_(t-l_j) + epsilon_t (mod 257)` with every
  `a_j` sampled from `F_p*` and anchored lags beginning at 1 and ending at `d`.
- An episode starts with a uniform `d`-state, executes at least `C=64` background
  transitions, exposes only the last 64 as context, and scores the continuation.
- `D` is the number of fresh next-token targets included in the sampled-token NLL.
  Background burn-in and context-only positions do not count toward `D`.
- `N_core` counts the Transformer blocks and final normalization. Token and
  position embeddings are reported separately through total parameter count; the
  language-model head is tied to the token embedding.
- Depth (6), head count (8), context (128), absolute learned positions, MLP ratio,
  dropout, AdamW hyperparameters, and the schedule as a function of absolute `D`
  are identical at every scale. Only width changes.
- Evaluation uses the full exact conditional distribution to calculate KL and
  normalized excess error `R`; soft conditionals never enter the training loss.

## Setup and first checks

```bash
uv sync --extra dev
uv run kepler model-sizes
uv run kepler manifest --output runs/manifest.json
uv run kepler phase0
uv run kepler phase1
uv run pytest
```

Phase 0 reports TV distances rather than testing exact zero. It checks residual
marginals, token stationarity, residual/history independence through a
preregistered four-bin deterministic history partition, and the `h_(t-1) -> z_t`
label alignment. Phase 1 compares FP32 forward loss and the complete first-order
gradient vector between CPU and local MPS. HVP support is explicitly not a gate.
If MPS is absent, Phase 1 returns `status=unavailable` and does not claim system
admission.

## Run one trajectory

Track IDs are emitted by `kepler manifest`. A short smoke run and exact evaluation
look like this:

```bash
uv run kepler train \
  --track-id k2_d4_n2m_r0_s0_primary \
  --max-tokens 65536 \
  --device auto

uv run kepler evaluate \
  runs/k2_d4_n2m_r0_s0_primary/checkpoint_65536.pt
```

For the registered run, omit `--max-tokens`; checkpoints are written at 5, 10,
20, 40, 80, and 160 million scored targets. Checkpoints include optimizer state,
the online NumPy stream state, and Torch RNG state so resumption does not replay
data.

## Probes, scaling, and functional paths

The probe command extracts train/validation/test representations from three
separately seeded streams that are also separate from training and evaluation:

```bash
uv run kepler probe CHECKPOINT.pt --layer 6
```

It fits a 257-class linear probe, a forced real Fourier readout with fixed
frequencies `m=1..128`, and a small MLP. Validation controls early stopping and
test is evaluated only after model selection.

Scaling input is a CSV with columns `track_id,k,d,N_core,D,R`:

```bash
uv run kepler fit-scaling results/evaluation.csv --output results/scaling.json
```

M0, M1, and M2 are compared by leaving out an entire trajectory, including all
of its exposure checkpoints. This prevents adjacent checkpoints from the same
optimization path leaking across train and holdout sets.

`kepler adaptation-protocol` emits the five registered local-path arms. Task B's
label semantics must be supplied by the experiment owner because the scientific
specification names it but does not define its mapping. Result-matched arms stop
at a preregistered target `R`; budget-matched arms stop at the same token budget.
The path utilities require byte-identical baseline parameters and compute every
Jeffreys step on one immutable `P_ref` sample.

## Repository layout

- `src/kepler/topology.py`: topology, rules, stationary burn-in generation
- `src/kepler/model.py`: fixed-geometry, width-only decoder Transformer
- `src/kepler/training.py`: online training, exact checkpoints, evaluation
- `src/kepler/probes.py`: predictive sufficient-state probe array
- `src/kepler/scaling.py`: positive M0/M1/M2 fits and trajectory holdout
- `src/kepler/fisher.py`: paired controls, memory table, path estimator
- `src/kepler/gates.py`: Phase 0 and Phase 1 admission checks
- `tests/`: protocol and numerical regression tests

## Phase-4 task conditioning and micro-pilot

Phase-4 adds a separate optional interface; it does not alter the production
80-trajectory manifest or the Phase 2/3 scaling models. `num_tasks=0` remains the
default. A Phase-4 model uses two zero-initialized per-sequence rows and accepts a
scalar task ID or one ID per batch element:

```bash
uv run kepler attach-task-conditioner BASE.pt PHASE4.pt --device mps
uv run kepler pilot-manifest --output artifacts/pilot_manifest.json
uv run kepler environment-audit
uv run kepler resume-audit --device mps
uv run kepler pilot-status --run-root artifacts/pilot_runs
```

The attachment command permits only `task_emb.weight` to be absent from a base
state, initializes both rows to zero, and checks FP32 base/Task-A/Task-B logits
before publishing a checksummed checkpoint. The fresh Phase-4 optimizer policy is
recorded and is identical across all five local adaptation arms.

The dedicated pilot contains 10 long 2M trajectories, four long 10M corner
trajectories, and one `(k=4,d=64)` 50M engineering soak. All pilot rows are marked
`exploratory_engineering` and excluded from the primary confirmatory fit. The soak
endpoint is preregistered at the allowed lower bound of 1M supervised tokens.

One pilot trajectory can be durably stopped and resumed only at its registered
checkpoints:

```bash
uv run kepler pilot-run \
  --track-id pilot_2m_long_k2_d4_n2m_r0_s0 \
  --run-root artifacts/pilot_runs \
  --stop-after-tokens 5000000

uv run kepler pilot-run \
  --track-id pilot_2m_long_k2_d4_n2m_r0_s0 \
  --run-root artifacts/pilot_runs \
  --resume artifacts/pilot_runs/pilot_2m_long_k2_d4_n2m_r0_s0/checkpoint_5000000.pt \
  --stop-after-tokens 10000000
```

Every durable checkpoint is flushed and fsynced, loaded once for validation,
checksummed, and published with `os.replace`. It includes model, optimizer,
absolute-token scheduler, all available RNG states, online generator state, rule
and seed identities, and config/Git hashes. Exact evaluation retains every
Episode's KL contribution and reports a deterministic 2,000-resample Episode
bootstrap interval for `R`.
