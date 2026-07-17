from collections import Counter

from kepler.confirmatory import (
    build_confirmatory_runs,
    confirmatory_manifest_payload,
    model_initialization_seeds,
)
from kepler.cue_dose import PROBE_CHECKPOINTS, TRAIN_CHECKPOINTS


def test_confirmatory_manifest_has_exact_frozen_factorial_design():
    manifest = confirmatory_manifest_payload()
    runs = build_confirmatory_runs()
    assert len(runs) == 24
    assert manifest["factorial_design"]["paired_blocks"] == 8
    assert manifest["primary_test"]["exact_assignment_count"] == 1_679_616
    assert manifest["primary_test"]["confirmatory_alpha"] == 0.05
    assert manifest["prior_manifests_modified"] is False
    assert all(
        run.confirmatory_replication
        and run.excluded_from_original_scaling_fit
        and run.maximum_pair_exposures == 10_000_000
        and run.checkpoints == TRAIN_CHECKPOINTS
        and run.probe_checkpoints == PROBE_CHECKPOINTS
        for run in runs
    )
    assert Counter(run.condition_id for run in runs) == {
        "R0": 8,
        "R1": 8,
        "R2": 8,
    }


def test_every_block_is_strictly_paired_except_registered_condition_fields():
    runs = build_confirmatory_runs()
    blocks = {run.block_id for run in runs}
    assert len(blocks) == 8
    for block_id in blocks:
        cells = [run for run in runs if run.block_id == block_id]
        assert [run.condition_id for run in cells] == ["R0", "R1", "R2"]
        for field in (
            "coefficients",
            "cue_coefficient_c",
            "permutation_seed_sha256",
            "B_m",
            "model_initialization_seed",
            "data_seed",
            "evaluation_seed",
            "probe_seed",
            "batch_size",
            "width",
        ):
            assert len({getattr(run, field) for run in cells}) == 1
        assert cells[0].lambda_value == 0.0
        assert 0.0 < cells[1].lambda_value < cells[2].lambda_value < 1.0


def test_model_seeds_are_two_new_distinct_frozen_values():
    seeds = model_initialization_seeds()
    assert len(set(seeds)) == 2
    assert not set(seeds) & set(range(5))
