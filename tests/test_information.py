import math

import numpy as np

from kepler.information import (
    analytic_parent_subset_mutual_information,
    conditional_distribution,
    exact_conditional_kl_from_log_probs,
    finite_sample_parent_subset_mi,
    noise_entropy,
    normalized_excess_error,
    predictable_information,
)
from kepler.config import DEFAULT_CONFIG
from kepler.topology import FiniteFieldRule


def test_noise_entropy_and_predictable_information_use_nats():
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1 / 256))
    assert math.isclose(noise_entropy(), expected, rel_tol=1e-14)
    assert math.isclose(predictable_information(), math.log(257) - expected, rel_tol=1e-14)


def test_exact_q_is_shifted_noise_and_self_kl_is_numerically_zero():
    centers = np.asarray([0, 17, 256])
    q = conditional_distribution(centers)
    assert np.allclose(q.sum(axis=-1), 1.0)
    assert np.all(np.argmax(q, axis=-1) == centers)
    kl = exact_conditional_kl_from_log_probs(np.log(q), centers)
    assert np.allclose(kl, 0.0, atol=1e-15)
    assert abs(normalized_excess_error(np.log(q), centers)) < 1e-15


def test_k2_is_analytic_pure_synergy_not_merely_fan_in():
    rule = FiniteFieldRule(
        p=257, eta=0.1, k=2, d=4, lags=(1, 4), coefficients=(45, 36)
    )
    assert analytic_parent_subset_mutual_information(rule, (0,)) == 0.0
    assert analytic_parent_subset_mutual_information(rule, (1,)) == 0.0
    assert math.isclose(
        analytic_parent_subset_mutual_information(rule, (0, 1)),
        math.log(257) - noise_entropy(),
        rel_tol=1e-14,
    )


def test_every_proper_parent_subset_has_zero_analytic_information():
    rule = FiniteFieldRule.sample(4, 4, 0, 322, DEFAULT_CONFIG.field)
    from itertools import combinations

    for size in range(4):
        for subset in combinations(range(4), size):
            assert analytic_parent_subset_mutual_information(rule, subset) == 0.0
    assert analytic_parent_subset_mutual_information(rule, range(4)) > 0.0


def test_proper_subset_independence_in_finite_samples():
    # Check both singleton and multi-parent proper subsets.  Stratification
    # avoids a sparse 257^|S| contingency table while preserving the null.
    rule = FiniteFieldRule.sample(4, 4, 0, 322, DEFAULT_CONFIG.field)
    for offset, subset in enumerate(((0,), (3,), (0, 1), (0, 1, 2))):
        estimate = finite_sample_parent_subset_mi(
            rule,
            subset,
            conditioning_states=24,
            samples_per_state=4096,
            seed=700 + offset,
        )
        assert abs(float(estimate["bias_corrected_mi_nats"])) < 0.01
