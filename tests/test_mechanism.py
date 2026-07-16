import math

import torch

from kepler.config import DEFAULT_CONFIG
from kepler.diagnostic_models import ConcatenatedParentMLP
from kepler.environment import file_sha256
from kepler.gradient_audit import _OnlineGradientMoments, gradient_audit_rules
from kepler.mechanism import (
    as_learnability_run,
    build_mechanism_isolation_runs,
    mechanism_manifest_payload,
)
from kepler.pair_diagnostics import make_pair_model


def test_mechanism_manifest_is_isolated_and_exact():
    runs = build_mechanism_isolation_runs()
    assert [run.run_id for run in runs] == [
        "mechanism_M1",
        "mechanism_M2",
        "mechanism_M3",
        "mechanism_M4",
    ]
    assert all(
        run.exploratory and run.diagnostic_only and run.excluded_from_confirmatory_fit
        for run in runs
    )
    m1, m2, m3, m4 = runs
    assert m1.rule["lags"] == [4] and m1.rule["coefficients"] == [45]
    assert m2.rule["lags"] == [1, 2] and m2.rule["coefficients"] == [45, 36]
    assert m1.checkpoints == (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
    assert m2.checkpoints == (5_000_000, 10_000_000, 20_000_000, 40_000_000)
    assert m3.loss_type == m4.loss_type == "oracle_soft"
    assert as_learnability_run(m1).as_training_track().rule["lags"] == [4]
    payload = mechanism_manifest_payload()
    assert payload["prior_manifests_mutated"] is False
    assert payload["gradient_signal_audit"]["model_initializations"] == 8


def test_pair_diagnostic_models_match_registered_core_size():
    _, _, m3_run, m4_run = build_mechanism_isolation_runs()
    m3 = make_pair_model(m3_run)
    m4 = make_pair_model(m4_run)
    assert m3.core_parameter_count() == m3_run.actual_core_parameters
    assert m3.total_parameter_count() == m3_run.total_parameters
    assert isinstance(m4, ConcatenatedParentMLP)
    assert m4.core_parameter_count() == 2_045_940
    assert abs(m4.core_parameter_count() - m3.core_parameter_count()) < 1_000


def test_gradient_moments_are_online_and_match_two_orthogonal_batches():
    moments = _OnlineGradientMoments()
    moments.update(torch.tensor([1.0, 0.0]))
    moments.update(torch.tensor([0.0, 1.0]))
    result = moments.summary()
    assert result["batches"] == 2
    assert math.isclose(result["norm_of_mean_gradient"], math.sqrt(0.5), rel_tol=1e-6)
    assert math.isclose(
        result["gradient_variance_mean_squared_deviation"], 0.5, rel_tol=1e-6
    )
    assert math.isclose(result["gradient_snr"], 1.0, rel_tol=1e-6)
    assert abs(result["mean_pairwise_gradient_cosine"]) < 1e-7


def test_gradient_audit_topologies_are_exact():
    rules = gradient_audit_rules()
    assert rules["k1_d1_lag1"].lags == (1,)
    assert rules["k1_d4_lag4"].lags == (4,)
    assert rules["k2_d2_lags1_2"].lags == (1, 2)
    assert rules["k2_d4_lags1_4"].lags == (1, 4)


def test_all_prior_manifest_hashes_remain_frozen():
    assert file_sha256("artifacts/manifest.json") == (
        "4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10"
    )
    assert file_sha256("artifacts/pilot_manifest.json") == (
        "3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0"
    )
    assert file_sha256("artifacts/learnability_audit_manifest.json") == (
        "95a21c5a89af958ae986b3d3284c9e610bd37c16a0d3e68dc2d01bd83c9f5af4"
    )
