from dataclasses import replace
import os
from pathlib import Path

from kepler.environment import LOCKED_MPS_ENV, file_sha256
import torch

from kepler.config import DEFAULT_CONFIG
from kepler.durability import durable_torch_save, validate_checkpoint_checksum
from kepler.model import (
    KeplerTransformer,
    assert_conditioner_attachment_equivalence,
    core_parameter_count_for_width,
    load_base_state_into_conditioned,
    size_spec,
)
from kepler.pilot import build_pilot_manifest
from kepler.topology import FiniteFieldRule


PRODUCTION_MANIFEST_SHA256 = "4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10"


def _model_cfg(num_tasks: int):
    return replace(
        DEFAULT_CONFIG.model,
        n_layers=1,
        n_heads=2,
        context_length=16,
        visible_prefix_length=8,
        mlp_ratio=2,
        num_tasks=num_tasks,
    )


def _paired_models():
    torch.manual_seed(123)
    base = KeplerTransformer(32, _model_cfg(0))
    conditioned = KeplerTransformer(32, _model_cfg(2))
    load_base_state_into_conditioned(conditioned, base.state_dict())
    return base, conditioned


def test_mps_environment_is_locked_before_project_torch_use():
    assert all(os.environ[key] == value for key, value in LOCKED_MPS_ENV.items())


def test_disabled_conditioner_is_original_and_zero_rows_preserve_logits():
    base, conditioned = _paired_models()
    tokens = torch.randint(0, 257, (3, 12))
    original = base(tokens)
    assert base.task_emb is None
    assert torch.equal(original, base(tokens))
    differences = assert_conditioner_attachment_equivalence(base, conditioned, tokens)
    assert max(differences.values()) <= 1e-6


def test_scalar_and_batch_task_ids_broadcast_without_time_or_lag_changes():
    _, conditioned = _paired_models()
    assert conditioned.task_emb is not None
    with torch.no_grad():
        conditioned.task_emb.weight[0].fill_(0.25)
        conditioned.task_emb.weight[1].fill_(-0.5)
    tokens = torch.randint(0, 257, (2, 10))
    scalar = conditioned(tokens, task_id=0)
    vector = conditioned(tokens, task_id=torch.tensor([0, 0]))
    mixed = conditioned(tokens, task_id=torch.tensor([0, 1]))
    assert torch.equal(scalar, vector)
    assert scalar.shape == mixed.shape == (2, 10, 257)
    assert not torch.equal(scalar[1], mixed[1])
    rule = FiniteFieldRule.sample(4, 64, 0, 322)
    assert rule.lags == (1, 22, 43, 64)


def test_only_used_task_row_receives_gradient():
    _, conditioned = _paired_models()
    tokens = torch.randint(0, 257, (2, 10))
    conditioned(tokens, task_id=1).sum().backward()
    gradient = conditioned.task_emb.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0]) == 0
    assert torch.count_nonzero(gradient[1]) > 0


def test_checkpoint_round_trip_preserves_conditioner(tmp_path):
    _, conditioned = _paired_models()
    with torch.no_grad():
        conditioned.task_emb.weight[1].fill_(0.125)
    checkpoint = tmp_path / "conditioned.pt"
    torch.save(conditioned.state_dict(), checkpoint)
    reloaded = KeplerTransformer(32, _model_cfg(2))
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(reloaded.task_emb.weight, conditioned.task_emb.weight)


def test_production_manifest_and_phase23_parameter_definition_are_unchanged():
    manifest = Path("artifacts/manifest.json")
    assert file_sha256(manifest) == PRODUCTION_MANIFEST_SHA256
    base_cfg = DEFAULT_CONFIG.model
    assert base_cfg.num_tasks == 0
    expected_counts = (2_045_568, 10_209_152, 20_114_688, 49_906_688)
    actual = tuple(
        size_spec(target, base_cfg).actual_core_parameters
        for target in base_cfg.target_core_parameters
    )
    assert actual == expected_counts
    assert all(
        size_spec(target, base_cfg).conditioner_parameters == 0
        for target in base_cfg.target_core_parameters
    )


def test_pilot_manifest_is_separate_exploratory_and_has_distinct_paired_rules():
    tracks = build_pilot_manifest()
    assert len(tracks) == 15
    assert sum(track.tier == "2m_long" for track in tracks) == 10
    assert sum(track.tier == "10m_long" for track in tracks) == 4
    assert sum(track.tier == "50m_soak" for track in tracks) == 1
    soak = next(track for track in tracks if track.tier == "50m_soak")
    assert soak.maximum_supervised_tokens == 1_000_000
    assert all(not track.include_primary_confirmatory_fit for track in tracks)
    for track in tracks:
        assert track.rule["lags"] == track.paired_new_rule["lags"]
        assert track.rule["eta"] == track.paired_new_rule["eta"]
        assert track.rule["coefficients"] != track.paired_new_rule["coefficients"]


def test_durable_checkpoint_is_validated_checksummed_and_reloadable(tmp_path):
    output = tmp_path / "durable.pt"
    payload = {"model": {"weight": torch.arange(3)}, "marker": "ok"}
    checksum = durable_torch_save(payload, output, required_keys={"model", "marker"})
    assert len(checksum) == 64
    assert validate_checkpoint_checksum(output)
    assert torch.equal(
        torch.load(output, map_location="cpu", weights_only=False)["model"]["weight"],
        torch.arange(3),
    )
