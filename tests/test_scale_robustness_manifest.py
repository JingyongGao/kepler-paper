from collections import Counter
import json
from pathlib import Path

from kepler.config import DEFAULT_CONFIG
from kepler.scale_robustness import (
    CONDITION_ORDER,
    SCALE_TARGETS,
    build_scale_robustness_runs,
    scale_robustness_manifest_payload,
)


def test_scale_robustness_has_exact_registered_factorial_design():
    manifest = scale_robustness_manifest_payload()
    runs = build_scale_robustness_runs()
    assert len(runs) == 16
    assert manifest["factorial_design"]["trajectories"] == 16
    assert manifest["factorial_design"]["paired_blocks"] == 8
    assert Counter(run.target_core_parameters for run in runs) == {
        1_000_000: 8,
        5_000_000: 8,
    }
    assert Counter(run.condition_id for run in runs) == {"R0": 8, "R2": 8}
    assert sum(run.engineering_learnability_pilot for run in runs) == 4
    assert all(
        run.post_confirmatory_robustness_extension
        and run.excluded_from_original_v3_6_p_value
        and run.not_a_scaling_law_fit
        for run in runs
    )


def test_scale_robustness_is_strictly_paired_within_every_block():
    runs = build_scale_robustness_runs()
    blocks = {run.block_id for run in runs}
    assert len(blocks) == 8
    for block_id in blocks:
        cells = [run for run in runs if run.block_id == block_id]
        assert [run.condition_id for run in cells] == list(CONDITION_ORDER)
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
            "target_core_parameters",
            "width",
        ):
            assert len({getattr(run, field) for run in cells}) == 1
        assert cells[0].lambda_value == 0.0
        assert cells[1].lambda_value > 0.0


def test_scale_widths_and_counts_follow_frozen_width_only_convention():
    runs = build_scale_robustness_runs()
    sizes = {
        run.target_core_parameters: (
            run.width,
            run.actual_core_parameters,
            run.total_parameters,
        )
        for run in runs
    }
    assert sizes == {
        1_000_000: (120, 1_046_400, 1_092_720),
        5_000_000: (264, 5_039_232, 5_141_136),
    }
    assert all(run.width % DEFAULT_CONFIG.model.n_heads == 0 for run in runs)


def test_transition_definitions_are_byte_for_byte_values_from_v36_manifest():
    old = json.loads(
        Path("artifacts/cue_dose_confirmatory_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    new = scale_robustness_manifest_payload()
    assert new["transition_definitions"] == old["transition_definitions"]
    assert new["factorial_design"]["scales_target_core_parameters"] == list(
        SCALE_TARGETS
    )


def test_all_prior_manifests_are_hash_locked_and_unmodified():
    manifest = scale_robustness_manifest_payload()
    assert manifest["prior_manifests_modified"] is False
    assert len(manifest["prior_manifest_and_result_sha256"]) == 9
