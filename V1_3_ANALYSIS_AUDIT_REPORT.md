# V1.3 Analysis Audit Report

## Provenance and immutability

- Parent commit: `a4d053d8dc4698c7e6de74d46d402a91434c047f`
- Frozen submission tag: `kepler-paper-v1.2.1-final`
- Analysis branch: `kepler-v1.3-transition-dynamics`
- Frozen files checked: 3235
- Frozen artifacts byte-identical before/after: `true`
- Missing frozen files: 0
- Changed frozen files: 0
- Optimizer steps executed: 0
- Checkpoints modified: 0

## Computation constraints

Representation and cue-removal analyses read frozen JSON/probe outputs. Spectral analysis loads model weights only, runs exact Hessian-vector products on CPU with the math attention backend, and never loads optimizer state. No training, checkpoint publication, manifest rewrite, cue-dose rerun, or statistical-definition change occurred.

## Tests

- Existing suite: `passed (62 tests)` via `PYTHONPATH=src .venv/bin/python -m pytest -q --ignore=tests/test_transition_dynamics.py`
- New analysis tests: `passed (5 tests)` via `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_transition_dynamics.py`

## Reproducibility commands

```bash
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py freeze
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py analyze
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py spectral
/Users/aijing/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_transition_figures.py
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py finalize
```

## Hash manifests

- Analysis JSON hashes: `hashes/json_hashes.json` (3 files)
- Figure/source hashes: `hashes/figure_hashes.json` (12 files)

## Environment

```json
{
  "cpu_count": 10,
  "hessian_device": "cpu",
  "hessian_sdpa_backend": "math",
  "machine": "arm64",
  "mps_available": true,
  "mps_built": true,
  "numpy": "2.5.1",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "processor": "arm",
  "python_executable": "/Volumes/externaldata/projects/intellligence_test/.venv/bin/python",
  "python_version": "3.13.13 (main, May 10 2026, 19:20:37) [Clang 22.1.3 ]",
  "scipy": "1.18.0",
  "torch": "2.13.0",
  "torch_num_threads": 4
}
```

The permitted interpretation is limited to transition dynamics, representation accessibility, and spectral signatures associated with selected checkpoints.
