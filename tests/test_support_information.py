import math

from kepler.information import noise_entropy
from kepler.support_information import build_support_information_audit


def test_full_support_has_zero_singleton_information():
    audit = build_support_information_audit()
    full = audit["supports"]["G1_full_support"]
    assert audit["G1_analytic_and_numeric_zero_check"]["passed"]
    assert full["singleton_cue_ratio"] < 1.0e-12
    assert math.isclose(
        full["mutual_information_nats"]["I_Y_X1_X2"],
        math.log(257) - noise_entropy(),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_frozen_restricted_support_has_complete_parent_coverage():
    audit = build_support_information_audit()
    for name in ("G2_training_support", "G2_heldout_support"):
        support = audit["supports"][name]
        assert support["support_size_distributions"]["row_X1"]["minimum"] > 0
        assert support["support_size_distributions"]["column_X2"]["minimum"] > 0
        assert support["support_size_distributions"]["target_state_Z"]["minimum"] > 0
