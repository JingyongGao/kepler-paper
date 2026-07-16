from kepler.cue_dose import (
    GRADIENT_AUDIT_CHECKPOINTS,
    PROBE_CHECKPOINTS,
    TRAIN_CHECKPOINTS,
    assert_c0_reuse_equivalence,
    build_cue_dose_runs,
    cue_dose_manifest_payload,
)


def test_c0_reuse_assertions_pass_for_all_three_frozen_seeds():
    result = assert_c0_reuse_equivalence()
    assert result["all_passed"]
    assert all(item["passed"] for item in result["seeds"].values())


def test_manifest_has_exactly_30_excluded_paired_cells():
    manifest = cue_dose_manifest_payload()
    runs = manifest["runs"]
    assert len(runs) == 30
    assert all(manifest["flags"].values())
    assert {run["model_initialization_seed"] for run in runs} == set(range(5))
    assert len({run["condition_id"] for run in runs}) == 6
    assert all(tuple(run["checkpoints"]) == TRAIN_CHECKPOINTS for run in runs)
    assert all(tuple(run["probe_checkpoints"]) == PROBE_CHECKPOINTS for run in runs)
    assert all(
        tuple(run["gradient_audit_checkpoints"]) == GRADIENT_AUDIT_CHECKPOINTS
        for run in runs
    )


def test_only_frozen_C0_seeds_are_reused():
    runs = build_cue_dose_runs()
    reused = [run.run_id for run in runs if run.reused_source_run_id]
    assert reused == ["cue_C0_s0", "cue_C0_s1", "cue_C0_s2"]
