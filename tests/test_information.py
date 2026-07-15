import math

import numpy as np

from kepler.information import (
    conditional_distribution,
    exact_conditional_kl_from_log_probs,
    noise_entropy,
    normalized_excess_error,
    predictable_information,
)


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

