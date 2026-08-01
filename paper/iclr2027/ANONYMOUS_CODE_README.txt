ANONYMOUS ICLR 2027 CODE AND RESULTS SUPPLEMENT

This archive contains source code, tests, frozen manifests, machine-readable
results, and audit reports. It intentionally omits checkpoint binaries, Git
history, submission correspondence, and author-identifying material.

The v3.6 result is the prospectively frozen confirmatory analysis. The 1M/5M
experiment is a post-confirmatory robustness extension, is excluded from the
v3.6 p-value, and is not a scaling-law fit. No combined p-value is reported.

Run the checkpoint-free lightweight test suite with:

    uv run --extra dev pytest -q \
        --deselect=tests/test_cue_dose_manifest.py::test_c0_reuse_fallback_is_triggered_by_checkpoint_schedule \
        --deselect=tests/test_cue_dose_manifest.py::test_manifest_has_exactly_30_excluded_paired_cells \
        --deselect=tests/test_cue_dose_manifest.py::test_all_five_C0_seeds_are_new_after_fallback \
        --deselect=tests/test_full_support_transfer.py::test_transfer_manifest_has_three_excluded_runs_and_frozen_sources \
        --deselect=tests/test_full_support_transfer.py::test_transfer_run_pairing_and_limits

This executes 64 tests. Five additional repository tests exercise reuse and
continuation logic by opening excluded legacy checkpoint binaries; they pass in
the full audited repository (69/69 total) but are intentionally unavailable in
this checkpoint-free archive. No test is silently skipped.

Checkpoint files are not needed to inspect the frozen result JSON or reproduce
the paper tables. Their SHA-256 verification records are embedded in the
results JSON and repeated in
artifacts/iclr2027_scale_robustness_checkpoint_hashes.sha256. Paths stored in
the frozen repository JSON are made archive-relative in this anonymous copy;
all numerical and hash fields are left unchanged.
