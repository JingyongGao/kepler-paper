import numpy as np
import torch

from kepler.confirmatory import build_confirmatory_runs
from kepler.cue_dose_runner import _forward_full_support
from kepler.pair_diagnostics import make_pair_model


class _ExactOracle(torch.nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.coefficients = coefficients

    def forward(self, parents, return_residuals=False):
        centers = (
            self.coefficients[0] * parents[:, 0]
            + self.coefficients[1] * parents[:, 1]
        ) % 257
        probabilities = torch.full(
            (parents.shape[0], 257),
            0.1 / 256,
            dtype=torch.float32,
            device=parents.device,
        )
        probabilities.scatter_(1, centers[:, None], 0.9)
        logits = probabilities.log()
        if return_residuals:
            width = 8
            residuals = {
                layer: torch.zeros(
                    parents.shape[0], width, device=parents.device
                )
                for layer in (0, 3, 6)
            }
            return logits, residuals
        return logits


def test_full_support_evaluation_uses_registered_replication_coefficients():
    coefficients = (205, 5)
    _, centers, kl, correct, center_probability, _, _, _ = (
        _forward_full_support(
            _ExactOracle(coefficients),
            torch.device("cpu"),
            collect_residuals=False,
            coefficients=coefficients,
        )
    )
    assert np.all(correct)
    assert np.max(kl) < 1.0e-6
    assert np.allclose(center_probability, 0.9, atol=2.0e-7)
    assert centers[257 + 1] == (205 + 5) % 257


def test_all_three_conditions_in_a_block_start_with_identical_weights():
    cells = build_confirmatory_runs()[:3]
    states = []
    for run in cells:
        torch.manual_seed(run.model_initialization_seed)
        states.append(make_pair_model(run).state_dict())
    assert all(
        torch.equal(states[0][name], states[index][name])
        for index in (1, 2)
        for name in states[0]
    )
