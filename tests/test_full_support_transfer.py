from kepler.compositional_evaluation import _axis_tail_metrics
from kepler.full_support_transfer import (
    TRANSFER_CHECKPOINTS,
    TRANSFER_START,
    build_full_support_transfer_runs,
    full_support_transfer_manifest_payload,
)


def test_transfer_manifest_has_three_excluded_runs_and_frozen_sources():
    manifest = full_support_transfer_manifest_payload()
    assert manifest["kind"] == "full_support_transfer_gate"
    assert all(manifest["flags"].values())
    assert len(manifest["runs"]) == 3
    for run in manifest["runs"]:
        assert run["transfer_start_exposure"] == TRANSFER_START
        assert tuple(run["checkpoints"]) == TRANSFER_CHECKPOINTS
        assert run["sampling_support"] == "full"
        assert run["source_checkpoint_sha256"]


def test_transfer_run_pairing_and_limits():
    runs = build_full_support_transfer_runs()
    assert [run.model_initialization_seed for run in runs] == [0, 1, 2]
    assert all(run.maximum_pair_exposures == 20_000_000 for run in runs)
    assert all(run.source_run_id.endswith(f"s{run.model_initialization_seed}") for run in runs)


def test_axis_tail_summary_reports_unclipped_worst_values():
    axis = {"R": [float(index) for index in range(257)]}
    tails = _axis_tail_metrics(axis)
    assert tails["R_quantiles"]["maximum"] == 256.0
    assert tails["worst_10_values"][0] == {"value": 256, "R": 256.0}
