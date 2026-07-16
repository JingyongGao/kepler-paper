import math

import numpy as np
import torch

from kepler.config import DEFAULT_CONFIG
from kepler.environment import file_sha256
from kepler.learnability import (
    build_learnability_manifest,
    no_training_metric_audit,
    oracle_soft_label_loss,
)
from kepler.topology import anchored_lags


def test_single_parent_lag_is_diagnostic_only_and_manifest_is_exact():
    assert anchored_lags(1, 1) == (1,)
    runs = build_learnability_manifest()
    assert len(runs) == 3
    assert all(
        run.exploratory
        and run.diagnostic_only
        and run.excluded_from_confirmatory_fit
        for run in runs
    )
    run_a, run_b, run_c = runs
    assert (run_a.k, run_a.d, run_a.loss_type) == (1, 1, "sampled_nll")
    assert run_a.checkpoints == (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
    assert (run_b.target_core_parameters, run_b.k, run_b.d) == (10_000_000, 2, 4)
    assert run_b.checkpoints == (5_000_000, 10_000_000, 20_000_000, 40_000_000)
    assert run_c.loss_type == "oracle_soft"


def test_no_training_oracles_and_shift_controls(tmp_path):
    result = no_training_metric_audit(output=tmp_path / "audit.json")
    assert result["passed"]
    assert abs(result["uniform_R"] - 1.0) < 1e-12
    assert abs(result["oracle_R"]) < 1e-12
    assert result["z_t_from_h_t_minus_1"]
    assert result["shifted_previous_R"] > 0.1
    assert result["shifted_next_R"] > 0.1


def test_oracle_soft_loss_equals_noise_entropy_for_oracle_logits():
    centers = torch.tensor([[0, 17], [256, 9]])
    q = torch.full((2, 2, 257), 0.1 / 256)
    q.scatter_(-1, centers.unsqueeze(-1), 0.9)
    mask = torch.ones_like(centers, dtype=torch.bool)
    loss = oracle_soft_label_loss(torch.log(q), centers, mask)
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1 / 256))
    assert math.isclose(float(loss), expected, rel_tol=1e-6)


def test_existing_manifest_hashes_remain_frozen():
    assert file_sha256("artifacts/manifest.json") == (
        "4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10"
    )
    assert file_sha256("artifacts/pilot_manifest.json") == (
        "3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0"
    )
