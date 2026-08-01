# ICLR 2027 cross-scale robustness freeze record

Record created: 2026-08-01 (Asia/Shanghai)

## Git and preregistration boundary

- Working branch: `kepler-iclr2027-scale-robustness`
- Parent TMLR state: `57ef195b654066b1c5ce7d18fcb9843978694b8c`
- Frozen TMLR tag: `kepler-paper-tmlr-desk-rejected`
- Preregistration source commit: `16d1d2e7609c0635f357b8160d2d05ee8f4198fa`
- Registered manifest SHA-256: `f6b873e342a8efa7f0a1b1ae09f0b455cc591ae06a31b0c9bb156726e5d23183`

The TMLR branch, the TMLR submission artifacts, all v1.3 audit material, the
v3.6 manifest and result, pre-existing checkpoints, branches, and tags were not
modified. All nine prior manifest/result hashes locked by the new manifest were
recomputed after the extension and matched their registered values.

## Registered scope and execution

- Scales: 1M target (`1,046,400` actual core parameters, width 120) and 5M
  target (`5,039,232` actual core parameters, width 264).
- Design: 2 scales x 2 frozen coefficient rules x 2 paired initialization
  seeds x 2 conditions (`R0`, `R2`) = 16 trajectories in 8 paired blocks.
- Interface and objective: `p=257`, `eta=0.1`, parent-exposed interface, exact
  soft-label cross-entropy, and the frozen width-only/optimizer protocol.
- Sampling: cue-bearing through 5M, then uniform full support from 5M to 10M.
- Pilot: four preregistered cells passed the engineering and learnability gate;
  cue-effect direction was not a Go/No-Go criterion.
- Completion: 16/16 trajectories reached 10M, each with all 12 registered
  evaluations.

This is a **post-confirmatory robustness extension**, is **excluded from the
original v3.6 p-value**, and is **not a scaling-law fit**. No combined
confirmatory p-value was computed, and the original v3.6 statistical result was
not changed.

## Frozen outcome summary

- 1M: R2 advanced verified onset in 4/4 paired blocks; verified onset was
  observed in 0/4 R0 and 4/4 R2 trajectories.
- 5M: R2 advanced verified onset in 4/4 paired blocks; verified onset was
  observed in 3/4 R0 and 4/4 R2 trajectories.
- All 8 paired blocks are reported individually in
  `ICLR2027_SCALE_ROBUSTNESS_REPORT.md`.
- The v3.6 definitions of verified onset, establishment, exact full-support
  R, center accuracy, registered probes, and persistence after cue removal
  were reused without modification.

## Integrity verification

- Registered checkpoint files verified: 176/176.
- All checkpoint SHA-256 sidecars valid: yes.
- All 16 zero-step switch checks passed: yes.
- Each switch preserved model, optimizer, scheduler, RNG, exposure, and pair
  frequency state, with zero intervening optimizer steps.
- The machine-readable result contains the individual SHA-256 value and
  sidecar comparison for every checkpoint.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| `artifacts/iclr2027_scale_robustness_manifest.json` | `f6b873e342a8efa7f0a1b1ae09f0b455cc591ae06a31b0c9bb156726e5d23183` |
| `artifacts/iclr2027_scale_robustness_pilot_gate.json` | `525d2b72b958c3d2d64a0ad89df3c01413c3a101c55bd537f1b3905f877e9e83` |
| `ICLR2027_SCALE_ROBUSTNESS_PILOT_REPORT.md` | `6adb644c9f07616ee19a0dbdad7a3bda4ef4ba9f2f2eedc5a3a36a2e0d6b43d9` |
| `artifacts/iclr2027_scale_robustness_results.json` | `155a3f6eb4fd9b5890ee6968d9075e0fc90b427afa95c8472757ddfad38325fe` |
| `ICLR2027_SCALE_ROBUSTNESS_REPORT.md` | `be2f0dc5ca742b08a268549cbb371c6cea6e3c9e752f1448c6228f4eb5108e17` |

## Scope exclusions and stop boundary

- No recurrent bridge.
- No p sweep.
- No sampled-NLL training objective.
- No new architecture.
- No combined p-value or scaling-law fit.
- The optional 3M/4M/5M early-removal continuation was **not started** and
  remains forbidden without separate authorization.

The cross-scale experiment is closed at this boundary. Subsequent paper-only
work must not alter this manifest, these results, or the recorded checkpoints.
