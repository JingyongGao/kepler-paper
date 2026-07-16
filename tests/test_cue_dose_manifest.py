from kepler.cue_dose import (
    GRADIENT_AUDIT_CHECKPOINTS,
    PROBE_CHECKPOINTS,
    TRAIN_CHECKPOINTS,
    assert_c0_reuse_equivalence,
    build_cue_dose_runs,
    cue_dose_manifest_payload,
)


def test_c0_reuse_fallback_is_triggered_by_checkpoint_schedule():
    result = assert_c0_reuse_equivalence()
    assert not result["all_passed"]
    assert result["fallback_triggered"]
    assert all(
        not item["checks"]["required_checkpoint_schedule_replay"]
        for item in result["seeds"].values()
    )


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


def test_all_five_C0_seeds_are_new_after_fallback():
    runs = build_cue_dose_runs()
    reused = [run.run_id for run in runs if run.reused_source_run_id]
    assert reused == []
    controls = [run for run in runs if run.condition_id == "C0"]
    assert len(controls) == 5
    assert all(run.execution_mode == "new_training" for run in controls)
