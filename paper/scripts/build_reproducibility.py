from __future__ import annotations

from common import (
    FREEZE_RECORD,
    PAPER,
    ROOT,
    V35_MANIFEST,
    V36_MANIFEST,
    atomic_write_json,
    atomic_write_text,
    load_json,
    require_frozen_source_hashes,
)


def main() -> None:
    require_frozen_source_hashes()
    freeze = load_json(FREEZE_RECORD)
    v35_manifest = load_json(V35_MANIFEST)
    v36_manifest = load_json(V36_MANIFEST)
    environment = freeze["environment_lock"]
    tag_commits = {
        item["name"]: item["commit"] for item in freeze["all_git_tags"]
    }
    manifests = freeze["manifest_sha256"]

    environment_spec = {
        "name": "kepler-paper-v1",
        "scientific_environment": {
            "authoritative_lock": "uv.lock",
            "pyproject": "pyproject.toml",
            "lock_sha256": environment["lock_files"]["uv.lock"],
            "pyproject_sha256": environment["lock_files"]["pyproject.toml"],
            "python": environment["python_version"],
            "platform": environment["platform"],
            "runtime": environment["runtime"],
        },
        "paper_rendering_overlay": {
            "python_requirement": ">=3.12,<3.14",
            "python_packages": ["reportlab==4.4.9"],
            "system_tools": ["poppler pdftoppm 26.05.0"],
            "optional_manuscript_compiler": "tectonic",
        },
        "determinism": {
            "pdf_invariant_mode": True,
            "png_source": "rasterized from invariant PDF at 180 dpi",
            "all_source_data_from_frozen_json": True,
            "network_required_during_build": False,
        },
    }
    atomic_write_json(PAPER / "environment_spec.json", environment_spec)
    atomic_write_text(
        PAPER / "requirements.txt",
        "# Paper-rendering overlay; scientific dependencies remain frozen in ../uv.lock.\n"
        "reportlab==4.4.9\n",
    )
    atomic_write_text(
        PAPER / "environment.yml",
        """name: kepler-paper-v1
channels:
  - conda-forge
dependencies:
  - python=3.13
  - poppler=26.05
  - pip
  - pip:
      - -e ..
      - reportlab==4.4.9
""",
    )
    atomic_write_text(
        PAPER / "ENVIRONMENT.md",
        f"""# Clean Environment Specification

The scientific runtime is frozen by [`../uv.lock`](../uv.lock), SHA-256
`{environment['lock_files']['uv.lock']}`. The project metadata SHA-256 is
`{environment['lock_files']['pyproject.toml']}`.

- Python: `{environment['python_version']}`
- Platform: `{environment['platform']}`
- Processor: `{environment['processor']}`
- PyTorch: `{environment['runtime']['torch']}`
- NumPy: `{environment['runtime']['numpy']}`
- SciPy: `{environment['runtime']['scipy']}`
- MPS available: `{environment['runtime']['mps_available']}`
- MPS fallback: `{environment['mps_environment']['PYTORCH_ENABLE_MPS_FALLBACK']}`
- MPS fast math: `{environment['mps_environment']['PYTORCH_MPS_FAST_MATH']}`

Paper rendering adds only `reportlab==4.4.9` and Poppler `pdftoppm`
`26.05.0`. The overlay is isolated from the scientific lock. Figures use
invariant PDF mode; PNG files are rendered from those PDFs at 180 dpi.
`environment_spec.json` is the machine-readable specification.
""",
    )

    v35_counts = freeze["artifact_counts"]["v3.5_exploratory"]
    v36_counts = freeze["artifact_counts"]["v3.6_confirmatory"]
    v35_exposures = len(v35_manifest["runs"]) * int(
        v35_manifest["runs"][0]["maximum_pair_exposures"]
    )
    v36_exposures = len(v36_manifest["runs"]) * int(
        v36_manifest["runs"][0]["maximum_pair_exposures"]
    )
    atomic_write_text(
        PAPER / "FULL_COMPUTE_DISCLOSURE.md",
        f"""# Full Compute Disclosure

## Scientific runs

| Experiment | Status | Trajectories | Maximum per trajectory | Aggregate pair exposures | Core parameters |
|---|---|---:|---:|---:|---:|
| v3.5 cue-dose gate | Exploratory discovery | {len(v35_manifest['runs'])} | {v35_manifest['runs'][0]['maximum_pair_exposures']:,} | {v35_exposures:,} | {v35_manifest['runs'][0]['actual_core_parameters']:,} |
| v3.6 cross-rule replication | Confirmatory | {len(v36_manifest['runs'])} | {v36_manifest['runs'][0]['maximum_pair_exposures']:,} | {v36_exposures:,} | {v36_manifest['runs'][0]['actual_core_parameters']:,} |

The paper therefore summarizes `{v35_exposures + v36_exposures:,}` exact
soft-label parent-pair exposures. Pair exposures are not reported as sampled
token labels.

## Evaluation and stored artifacts

| Experiment | Checkpoints | Exact evaluations | Probe reports | Gradient audits | Zero-step switch checks |
|---|---:|---:|---:|---:|---:|
| v3.5 | {v35_counts['checkpoints']} | {v35_counts['evaluations']} | {v35_counts['probe_reports']} | {v35_counts['gradient_audit_records']} | {v35_counts['switch_invariance_records']} |
| v3.6 | {v36_counts['checkpoints']} | {v36_counts['evaluations']} | {v36_counts['probe_reports']} | {v36_counts['gradient_audit_records']} | {v36_counts['switch_invariance_records']} |

All exact evaluations enumerate 66,049 ordered parent pairs. Registered
bootstrap intervals use 2,000 first-parent-row resamples. Checkpoint sidecars
validated for all {v35_counts['checkpoints'] + v36_counts['checkpoints']}
checkpoints.

## Hardware and software

- Processor: `{environment['processor']}`
- Physical memory: `{environment['physical_memory_bytes']:,}` bytes
- OS: `{environment['macos']['product']} {environment['macos']['version']}`
- PyTorch: `{environment['runtime']['torch']}`
- Accelerator: local Apple MPS
- Model precision and evaluation behavior: as frozen in the manifests and
  environment record

Wall-clock training time and energy use were not measured under a calibrated
protocol and are therefore not estimated. Paper packaging and lightweight
verification execute zero optimizer steps.
""",
    )

    genealogy = {
        "project": "Project Kepler",
        "kind": "machine_readable_experiment_genealogy",
        "nodes": [
            {
                "id": "v3.2.2-rc1",
                "tag": "kepler-v3.2.2-rc1",
                "commit": tag_commits["kepler-v3.2.2-rc1"],
                "role": "frozen engineering baseline",
                "status": "baseline",
                "manifest": "artifacts/manifest.json",
                "manifest_sha256": manifests["artifacts/manifest.json"],
            },
            {
                "id": "v3.2.2-learnability",
                "tag": "kepler-v3.2.2-audit1",
                "commit": tag_commits["kepler-v3.2.2-audit1"],
                "role": "learnability audit",
                "status": "diagnostic",
                "manifest": "artifacts/learnability_audit_manifest.json",
                "manifest_sha256": manifests[
                    "artifacts/learnability_audit_manifest.json"
                ],
            },
            {
                "id": "v3.3-mechanism",
                "tag": "kepler-v3.3-mechanism1",
                "commit": tag_commits["kepler-v3.3-mechanism1"],
                "role": "mechanism isolation",
                "status": "diagnostic",
                "manifest": "artifacts/mechanism_isolation_manifest.json",
                "manifest_sha256": manifests[
                    "artifacts/mechanism_isolation_manifest.json"
                ],
            },
            {
                "id": "v3.3-composition",
                "tag": "kepler-v3.3-composition1",
                "commit": tag_commits["kepler-v3.3-composition1"],
                "role": "parent-pair compositional generalization",
                "status": "exploratory",
                "manifest": "artifacts/compositional_generalization_manifest.json",
                "manifest_sha256": manifests[
                    "artifacts/compositional_generalization_manifest.json"
                ],
            },
            {
                "id": "v3.4-transfer",
                "tag": "kepler-v3.4-transfer1",
                "commit": tag_commits["kepler-v3.4-transfer1"],
                "role": "full-support transfer",
                "status": "exploratory",
                "manifest": "artifacts/full_support_transfer_manifest.json",
                "manifest_sha256": manifests[
                    "artifacts/full_support_transfer_manifest.json"
                ],
            },
            {
                "id": "v3.5-discovery",
                "tag": "kepler-v3.5-cuedose1",
                "commit": tag_commits["kepler-v3.5-cuedose1"],
                "role": "cue-dose discovery and geometry controls",
                "status": "exploratory_discovery",
                "manifest": "artifacts/cue_dose_manifest.json",
                "manifest_sha256": manifests["artifacts/cue_dose_manifest.json"],
                "result": "artifacts/cue_dose_results.json",
                "result_sha256": freeze["result_sha256"]["v3.5_exploratory"],
            },
            {
                "id": "v3.6-confirmation",
                "tag": "kepler-v3.6-confirm1",
                "commit": freeze["final_v3_6_commit"],
                "role": "independent cross-rule cue-dose replication",
                "status": "confirmatory_replication",
                "manifest": "artifacts/cue_dose_confirmatory_manifest.json",
                "manifest_sha256": manifests[
                    "artifacts/cue_dose_confirmatory_manifest.json"
                ],
                "result": "artifacts/cue_dose_confirmatory_results.json",
                "result_sha256": freeze["result_sha256"]["v3.6_confirmatory"],
            },
            {
                "id": "paper-v1",
                "branch": "kepler-paper-v1",
                "role": "read-only manuscript derivation",
                "status": "derived_no_training",
                "source_tag": "kepler-v3.6-confirm1",
                "optimizer_steps": 0,
            },
        ],
        "edges": [
            {"from": "v3.2.2-rc1", "to": "v3.2.2-learnability"},
            {"from": "v3.2.2-learnability", "to": "v3.3-mechanism"},
            {"from": "v3.3-mechanism", "to": "v3.3-composition"},
            {"from": "v3.3-composition", "to": "v3.4-transfer"},
            {"from": "v3.4-transfer", "to": "v3.5-discovery"},
            {
                "from": "v3.5-discovery",
                "to": "v3.6-confirmation",
                "relationship": "independent replication; discovery data excluded from primary p-value",
            },
            {
                "from": "v3.6-confirmation",
                "to": "paper-v1",
                "relationship": "frozen JSON derivation only",
            },
        ],
    }
    atomic_write_json(PAPER / "experiment_genealogy.json", genealogy)

    atomic_write_text(
        PAPER / "REPRODUCE.md",
        """# Command-Only Reproduction Guide

## Checkout and verify the frozen source

```sh
git checkout kepler-paper-v1
test "$(git rev-parse kepler-v3.6-confirm1)" = "f9fc22d85cd52863087afbb3f924466a63b30ba2"
uv sync --frozen
```

## Install the isolated paper-rendering overlay

```sh
command -v pdftoppm
uv run --with reportlab==4.4.9 python -c "import reportlab; assert reportlab.Version == '4.4.9'"
```

## Lightweight scientific verification without training

```sh
uv run python paper/scripts/verify_results.py
uv run python paper/scripts/verify_frozen_artifacts.py
```

## Rebuild every figure, table, index, and manuscript

```sh
uv run --with reportlab==4.4.9 python paper/scripts/build_all.py
```

## Test and hash verification

```sh
uv run --with reportlab==4.4.9 python paper/scripts/verify_package.py --rebuild --with-tests
```

## Optional LaTeX compilation

```sh
cd paper
tectonic manuscript.tex
cd ..
```
""",
    )
    atomic_write_text(
        PAPER / "README.md",
        """# Project Kepler Manuscript Package

- `manuscript.md` and `manuscript.tex`: synchronized manuscript drafts.
- `ARTIFACT_INDEX.md` and `artifact_index.json`: canonical frozen-source index.
- `figures/`: seven figures in SVG, PDF, and PNG, plus source data.
- `tables/`: eight tables in Markdown, CSV, and LaTeX.
- `REPRODUCE.md`: command-only no-training reproduction path.
- `environment_spec.json`, `requirements.txt`, and `environment.yml`: clean environment specification.
- `FULL_COMPUTE_DISCLOSURE.md`: scientific and rendering compute disclosure.
- `experiment_genealogy.json`: machine-readable experiment lineage.
- `scripts/`: deterministic builders and read-only verifiers.

Version 3.5 is exploratory discovery. Version 3.6 is the independent
confirmatory replication. They remain separate in every statistical output.
""",
    )
    print(PAPER / "experiment_genealogy.json")


if __name__ == "__main__":
    main()
