from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import torch

from kepler.config import DEFAULT_CONFIG
from kepler.environment import canonical_hash
from kepler.scale_robustness import (
    CONDITION_ORDER,
    SCALE_TARGETS,
    build_scale_robustness_runs,
    scale_robustness_manifest_payload,
)
from kepler.scale_robustness_runner import (
    _write_resume_verification,
    make_scale_robustness_model,
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


def test_runtime_models_match_each_registered_scale_without_architecture_changes():
    runs = build_scale_robustness_runs()
    for target in SCALE_TARGETS:
        run = next(run for run in runs if run.target_core_parameters == target)
        model = make_scale_robustness_model(run)
        assert model.width == run.width
        assert model.core_parameter_count() == run.actual_core_parameters
        assert model.total_parameter_count() == run.total_parameters
        assert model.cfg.n_layers == DEFAULT_CONFIG.model.n_layers
        assert model.cfg.n_heads == DEFAULT_CONFIG.model.n_heads


def test_valid_unreset_checkpoint_passes_resume_verification(tmp_path):
    run = build_scale_robustness_runs()[0]
    model = make_scale_robustness_model(run)
    optimizer = torch.optim.AdamW(model.parameters())
    state = {
        "run": asdict(run),
        "config_hash": canonical_hash(DEFAULT_CONFIG.to_dict()),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "exposed_examples": 2_000_000,
        "numpy_stream_state": {},
        "pair_frequency_counts": [],
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    checkpoint = tmp_path / "checkpoint_2000000.pt"
    torch.save(state, checkpoint)
    _write_resume_verification(
        tmp_path, checkpoint, state, model, optimizer, run, DEFAULT_CONFIG
    )
    result = json.loads(
        (tmp_path / "resume_verification.json").read_text(encoding="utf-8")
    )
    assert result["passed"] is True
    assert result["checks"]["optimizer_not_reset"] is True
    assert result["checks"]["scheduler_not_reset"] is True
