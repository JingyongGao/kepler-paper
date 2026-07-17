# Full Compute Disclosure

## Scientific runs

| Experiment | Status | Trajectories | Maximum per trajectory | Aggregate pair exposures | Core parameters |
|---|---|---:|---:|---:|---:|
| v3.5 cue-dose gate | Exploratory discovery | 30 | 10,000,000 | 300,000,000 | 2,045,568 |
| v3.6 cross-rule replication | Confirmatory | 24 | 10,000,000 | 240,000,000 | 2,045,568 |

The paper therefore summarizes `540,000,000` exact
soft-label parent-pair exposures. Pair exposures are not reported as sampled
token labels.

## Evaluation and stored artifacts

| Experiment | Checkpoints | Exact evaluations | Probe reports | Gradient audits | Zero-step switch checks |
|---|---:|---:|---:|---:|---:|
| v3.5 | 330 | 360 | 210 | 150 | 30 |
| v3.6 | 264 | 288 | 168 | 0 | 24 |

All exact evaluations enumerate 66,049 ordered parent pairs. Registered
bootstrap intervals use 2,000 first-parent-row resamples. Checkpoint sidecars
validated for all 594
checkpoints.

## Hardware and software

- Processor: `Apple M4`
- Physical memory: `34,359,738,368` bytes
- OS: `macOS 26.5.2`
- PyTorch: `2.13.0`
- Accelerator: local Apple MPS
- Model precision and evaluation behavior: as frozen in the manifests and
  environment record

Wall-clock training time and energy use were not measured under a calibrated
protocol and are therefore not estimated. Paper packaging and lightweight
verification execute zero optimizer steps.
