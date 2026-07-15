import numpy as np

from kepler.topology import FiniteFieldRule, anchored_lags, generate_episode


def test_registered_lag_layouts_are_exact_and_aperiodic():
    assert anchored_lags(2, 4) == (1, 4)
    assert anchored_lags(4, 4) == (1, 2, 3, 4)
    assert anchored_lags(2, 64) == (1, 64)
    assert anchored_lags(4, 64) == (1, 22, 43, 64)
    assert anchored_lags(3, 16) == (1, 8, 16)


def test_episode_has_exact_scored_alignment_and_no_independent_prefix_splice():
    rule = FiniteFieldRule.sample(4, 64, instance=0, seed=322)
    episode = generate_episode(rule, np.random.default_rng(11), 64, 64, 64)
    assert episode.tokens.shape == (128,)
    assert episode.loss_mask.sum() == 64
    for position in np.flatnonzero(episode.loss_mask):
        assert int(rule.center(episode.tokens[: position + 1])) == int(
            episode.target_centers[position]
        )
        residual = (
            int(episode.targets[position]) - int(episode.target_centers[position])
        ) % rule.p
        assert 0 <= residual < rule.p


def test_rule_coefficients_are_nonzero_and_enumeration_stable():
    first = FiniteFieldRule.sample(3, 16, instance=2, seed=322)
    second = FiniteFieldRule.sample(3, 16, instance=2, seed=322)
    assert first == second
    assert all(1 <= value < 257 for value in first.coefficients)

