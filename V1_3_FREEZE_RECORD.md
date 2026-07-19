# Project Kepler v1.3 Freeze Record

## Git provenance

- Analysis branch: `kepler-v1.3-transition-dynamics`
- Parent commit: `a4d053d8dc4698c7e6de74d46d402a91434c047f`
- Frozen submission tag: `kepler-paper-v1.2.1-final`
- Submission tag commit: `a4d053d8dc4698c7e6de74d46d402a91434c047f`
- Source submission reference: `kepler-paper-v1.2.1-submission`
- Freeze tag to be applied to this audit commit: `kepler-v1.3-transition-audit`

The TMLR submission branch and its frozen submission tag were not modified by the
v1.3 audit.

## Analysis scope

The v1.3 work is a post-submission audit of transition dynamics using only frozen
v3.5/v3.6 scientific artifacts and frozen confirmatory checkpoints. It includes:

- layer-wise linear, fixed-Fourier, and fixed-capacity MLP probe post-processing;
- predictive-state onset and establishment summaries under the registered
  definitions;
- CPU Hessian-vector-product spectral post-processing at six representative
  checkpoints;
- no-step 3M, 4M, and 5M cue-removal availability and persistence accounting;
- deterministic SVG, PDF, and PNG figures generated from frozen source JSON.

No Fourier emergence threshold was introduced because the frozen methodology does
not contain a registered Fourier R2 threshold. The 3M and 4M cue-removal
continuations were marked unavailable rather than imputed.

## Frozen artifact verification

- Frozen files checked before and after analysis: `3,235`
- Frozen bytes checked: `19,653,592,685`
- Checkpoint files recorded: `698`
- Missing frozen files: `0`
- Changed frozen files: `0`
- Checkpoint modifications: `0`
- Before/after inventories byte-identical: `true`

| Integrity artifact | SHA-256 |
|---|---|
| `artifacts/v1_3_transition_freeze_record.json` | `3bddf5d489a97db64e29ce19ae328437d731a2e143362730ee15aced35bbe7f1` |
| `hashes/frozen_scientific_artifacts.before.json` | `c8903c4c6fcac220a939bd12383ef9746096e1982bcac440390bfbc8dc3b3154` |
| `hashes/frozen_scientific_artifacts.after.json` | `c8903c4c6fcac220a939bd12383ef9746096e1982bcac440390bfbc8dc3b3154` |
| `hashes/json_hashes.json` | `b572382e4dbbfe7435c5a4934b7f6111b113592695c7ae3baf770bccbc67fe81` |
| `hashes/figure_hashes.json` | `bda3a90ce2d5b4014230c16d07c199430bd03d5b2612295b7cdb8ca4ff176463` |

## Analysis artifact hashes

| Artifact | SHA-256 |
|---|---|
| `artifacts/representation_transition_dynamics.json` | `b7e62aed4eb8220b36accbcacbb3b05724406df3530ef0d388a049ad6b856f98` |
| `artifacts/spectral_transition_dynamics.json` | `a38246e51e1b851b7672072a50f03befee514b05c23b3387bfb31d4f663e6af3` |
| `artifacts/early_cue_removal_analysis.json` | `834f62e00271c9b1ed0229c3ba880ebc7c97906c026ca12dda3235dd5e80dc86` |
| `REPRESENTATION_TRANSITION_REPORT.md` | `db4ad060665680aab5713acd76c4c3c03d00bf522cb825ba29fa9a927bc79fce` |
| `SPECTRAL_TRANSITION_REPORT.md` | `8cf3ef98be4128fb9fbf4405fdba4a54075bb9db00860b23d4183446dd2ff8f7` |
| `EARLY_CUE_REMOVAL_REPORT.md` | `3995b15d38556aade7eb54938d8511e77da5a71ad9e1150ccadf0071357a5d4e` |
| `V1_3_ANALYSIS_AUDIT_REPORT.md` | `1edc907e659365928bc034bd15d66436b665e5303cebaa55c173640b23e3351d` |

Hashes for all nine rendered figure files and their three source JSON files are
recorded in `hashes/figure_hashes.json`; per-figure sidecars are stored under
`hashes/`.

## Explicit freeze statement

This audit executed **no optimizer steps**, made **no checkpoint modification**,
and created **no new experiments**. It did not train a model, change a frozen
manifest, rerun cue-dose experiments, or modify the TMLR submission branch.
