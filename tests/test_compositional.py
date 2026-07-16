import json

import numpy as np
import torch

from kepler.compositional import build_compositional_runs, compositional_manifest_payload
from kepler.environment import file_sha256
from kepler.pair_diagnostics import make_pair_model
from kepler.parent_pair_split import (
    SPLIT_SEED_SHA256,
    build_parent_pair_split,
    load_parent_pair_split,
    sample_parent_pairs,
)


def test_parent_pair_split_is_deterministic_balanced_and_leak_free():
    first = build_parent_pair_split()
    second = build_parent_pair_split()
    assert first == second
    assert first["preregistered_sha256_seed"] == SPLIT_SEED_SHA256
    assert first["assignment_sha256"] == (
        "f16c708e5b49f1aab8a0f8e4ccaadcc8da506ca22d8ac210a583a12d5a1371c9"
    )
    assert (first["train_pairs"], first["heldout_pairs"]) == (52_839, 13_210)
    assert all(first["coverage_assertions"].values())
    assert first["restricted_sampler_audit"]["pair_leakage_count"] == 0
    train, heldout, mask = load_parent_pair_split()
    assert np.intersect1d(train, heldout).size == 0
    rng = np.random.default_rng(1)
    _, linear = sample_parent_pairs(rng, 100_000, "train")
    assert not mask[linear].any()


def test_compositional_manifest_has_three_paired_seeds_and_six_excluded_runs():
    runs = build_compositional_runs()
    assert len(runs) == 6
    assert {run.model_initialization_seed for run in runs} == {0, 1, 2}
    for seed in (0, 1, 2):
        pair = [run for run in runs if run.model_initialization_seed == seed]
        assert {run.arm for run in pair} == {"G1", "G2"}
        assert len({run.data_seed for run in pair}) == 1
        assert len({run.batch_size for run in pair}) == 1
    assert all(
        run.exploratory
        and run.diagnostic_only
        and run.excluded_from_confirmatory_scaling_fit
        for run in runs
    )
    assert runs[0].checkpoints == (
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        40_000_000,
    )
    payload = compositional_manifest_payload()
    assert payload["prior_manifests_modified"] is False
    assert payload["parent_pair_split"]["assignment_sha256"] == (
        "f16c708e5b49f1aab8a0f8e4ccaadcc8da506ca22d8ac210a583a12d5a1371c9"
    )


def test_paired_arms_start_from_identical_parameters():
    g1, g2 = build_compositional_runs()[:2]
    torch.manual_seed(g1.model_initialization_seed)
    model1 = make_pair_model(g1)
    torch.manual_seed(g2.model_initialization_seed)
    model2 = make_pair_model(g2)
    assert all(
        torch.equal(model1.state_dict()[name], model2.state_dict()[name])
        for name in model1.state_dict()
    )


def test_previous_manifests_remain_frozen():
    expected = {
        "artifacts/manifest.json": "4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10",
        "artifacts/pilot_manifest.json": "3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0",
        "artifacts/learnability_audit_manifest.json": "95a21c5a89af958ae986b3d3284c9e610bd37c16a0d3e68dc2d01bd83c9f5af4",
        "artifacts/mechanism_isolation_manifest.json": "d854fa799a361287111c2f3ad996558992f72741a6560883517caf0979070b5a",
    }
    assert {path: file_sha256(path) for path in expected} == expected
