import math

import numpy as np

from kepler.confirmatory_distribution import (
    ENUMERATION_TOLERANCE,
    PERMUTATION_SEEDS_SHA256,
    TARGET_CONDITIONS,
    V3_5_COEFFICIENTS,
    V3_5_PERMUTATION_SEED_SHA256,
    build_confirmatory_distribution_audit,
    field_permutation,
    generate_replication_rules,
    sample_confirmatory_pairs,
)
from kepler.information import noise_entropy


def test_independent_rules_and_permutations_are_frozen_and_valid():
    rules = generate_replication_rules()
    assert len(rules) == 2
    assert len({tuple(rule["coefficients"]) for rule in rules}) == 2
    for rule in rules:
        a1, a2 = rule["coefficients"]
        c = rule["cue_coefficient_c"]
        assert a1 != 0 and a2 != 0 and a1 != a2
        assert c != 0 and c != a1 and c != a2
        assert (a1, a2) != V3_5_COEFFICIENTS
    assert len(set(PERMUTATION_SEEDS_SHA256)) == 2
    old = field_permutation(V3_5_PERMUTATION_SEED_SHA256)
    assert all(field_permutation(seed) != old for seed in PERMUTATION_SEEDS_SHA256)
    assert field_permutation(PERMUTATION_SEEDS_SHA256[0]) != field_permutation(
        PERMUTATION_SEEDS_SHA256[1]
    )


def test_all_twelve_distributions_pass_exact_enumeration():
    payload = build_confirmatory_distribution_audit()
    assert len(payload["conditions"]) == 12
    assert all(payload["independence_checks"].values())
    for condition in payload["conditions"]:
        assert condition["strictly_positive_full_support"]
        assert max(condition["uniformity_max_absolute_error"].values()) < ENUMERATION_TOLERANCE
        assert condition["joint_information_absolute_error"] < ENUMERATION_TOLERANCE
        assert math.isclose(
            condition["entropies_nats"]["H_Y_given_X1_X2"],
            noise_entropy(),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert abs(
            condition["singleton_cue_index"]
            - condition["target_singleton_cue_index"]
        ) < 2.0e-14
        if condition["condition_id"] == "R0":
            assert condition["lambda"] == 0.0
            assert condition["minimum_pair_probability"] == 1.0 / (257 * 257)
            assert condition["maximum_pair_probability"] == 1.0 / (257 * 257)
    assert tuple(name for name, _ in TARGET_CONDITIONS) == ("R0", "R1", "R2")


def test_confirmatory_sampler_is_reproducible_and_r0_matches_uniform_draws():
    rule = generate_replication_rules()[0]
    coefficients = tuple(rule["coefficients"])
    c = rule["cue_coefficient_c"]
    seed = PERMUTATION_SEEDS_SHA256[0]
    first = np.random.default_rng(9901)
    second = np.random.default_rng(9901)
    pairs_a, linear_a = sample_confirmatory_pairs(
        first, 4096, 0.3, coefficients, c, seed
    )
    pairs_b, linear_b = sample_confirmatory_pairs(
        second, 4096, 0.3, coefficients, c, seed
    )
    assert np.array_equal(pairs_a, pairs_b)
    assert np.array_equal(linear_a, linear_b)

    baseline = np.random.default_rng(9902)
    candidate = np.random.default_rng(9902)
    expected = baseline.integers(0, 257 * 257, size=4096, dtype=np.int64)
    _, observed = sample_confirmatory_pairs(
        candidate, 4096, 0.0, coefficients, c, seed
    )
    assert np.array_equal(observed, expected)
    assert baseline.bit_generator.state == candidate.bit_generator.state
