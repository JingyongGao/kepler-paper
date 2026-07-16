import math

import numpy as np

from kepler.cue_distribution import (
    ENUMERATION_TOLERANCE,
    TARGET_CONDITIONS,
    build_cue_distribution_audit,
    cue_pair_probabilities,
    cue_subset,
    sample_cue_pairs,
)
from kepler.information import noise_entropy


def test_registered_cue_conditions_satisfy_all_exact_identities():
    payload = build_cue_distribution_audit()
    assert len(payload["conditions"]) == 6
    for condition in payload["conditions"]:
        assert 0.0 <= condition["lambda"] < 1.0
        assert condition["strictly_positive_full_support"]
        assert len(condition["B_m"]) == condition["m"]
        assert condition["B_m"] == list(cue_subset(condition["m"]))
        assert max(condition["uniformity_max_absolute_error"].values()) < ENUMERATION_TOLERANCE
        assert condition["singleton_information_symmetry_absolute_error"] < ENUMERATION_TOLERANCE
        assert math.isclose(
            condition["entropies_nats"]["H_Y_given_X1_X2"],
            noise_entropy(),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert math.isclose(
            condition["mutual_information_nats"]["I_Y_X1_X2"],
            math.log(257) - noise_entropy(),
            rel_tol=0.0,
            abs_tol=ENUMERATION_TOLERANCE,
        )
        assert abs(
            condition["singleton_cue_index"]
            - condition["target_singleton_cue_index"]
        ) < 2.0e-14


def test_lambda_zero_is_exact_existing_uniform_pair_distribution():
    probabilities = cue_pair_probabilities(0.0, 64)
    assert np.array_equal(probabilities, np.full(257 * 257, 1.0 / (257 * 257)))
    assert TARGET_CONDITIONS[0] == ("C0", 64, 0.0)


def test_sampling_is_reproducible_and_lambda_zero_matches_linear_draws():
    first = np.random.default_rng(913)
    second = np.random.default_rng(913)
    pairs_a, linear_a = sample_cue_pairs(first, 4096, 0.3, 64)
    pairs_b, linear_b = sample_cue_pairs(second, 4096, 0.3, 64)
    assert np.array_equal(pairs_a, pairs_b)
    assert np.array_equal(linear_a, linear_b)

    baseline = np.random.default_rng(117)
    candidate = np.random.default_rng(117)
    expected = baseline.integers(0, 257 * 257, size=4096, dtype=np.int64)
    _, observed = sample_cue_pairs(candidate, 4096, 0.0, 64)
    assert np.array_equal(observed, expected)
    assert baseline.bit_generator.state == candidate.bit_generator.state
