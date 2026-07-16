import numpy as np

from kepler.parent_pair_split import load_parent_pair_split
from kepler.support_gradient_audit import _fixed_linear_batches, _paired_differences


def test_fixed_restricted_audit_batches_never_leak():
    train, _, heldout = load_parent_pair_split()
    first = _fixed_linear_batches(123, "train", train, batches=2, batch_size=64)
    second = _fixed_linear_batches(123, "train", train, batches=2, batch_size=64)
    assert np.array_equal(first, second)
    assert not heldout[first].any()


def test_paired_difference_direction_is_g2_minus_g1():
    metrics = {
        "norm_of_mean_gradient": 1.0,
        "gradient_variance_mean_squared_deviation": 2.0,
        "gradient_snr": 3.0,
        "mean_pairwise_gradient_cosine": 4.0,
        "mean_batch_gradient_norm": 5.0,
        "rms_batch_gradient_norm": 6.0,
    }
    g1 = {"mean_exact_soft_cross_entropy_nats": 2.0, "groups": {"x": metrics}}
    g2 = {
        "mean_exact_soft_cross_entropy_nats": 3.0,
        "groups": {"x": {key: value + 0.5 for key, value in metrics.items()}},
    }
    result = _paired_differences(g1, g2)
    assert result["mean_exact_soft_cross_entropy_nats"] == 1.0
    assert all(value == 0.5 for value in result["groups"]["x"].values())
