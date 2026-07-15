import torch

from kepler.fisher import (
    fisher_rao_path_length,
    jeffreys_divergence_from_logits,
    registered_adaptation_arms,
)
from kepler.gates import phase0_gate
from kepler.probes import fourier_targets


def test_phase0_small_sample_smoke_uses_tolerance_not_zero():
    result = phase0_gate(samples=25_000, delta=0.16)
    assert result["passed"]
    assert result["criterion"].startswith("every reported")


def test_fourier_basis_has_all_non_dc_real_coordinates():
    target = fourier_targets(torch.tensor([0, 1, 256]))
    assert target.shape == (3, 256)
    assert torch.allclose(target[0, 0::2], torch.ones(128))
    assert torch.allclose(target[0, 1::2], torch.zeros(128))


def test_jeffreys_path_is_symmetric_and_zero_for_unchanged_model():
    a = torch.randn(8, 257)
    b = torch.randn(8, 257)
    assert torch.allclose(
        jeffreys_divergence_from_logits(a, b),
        jeffreys_divergence_from_logits(b, a),
        atol=1e-6,
    )
    unchanged = fisher_rao_path_length([a, a.clone()])
    assert unchanged["L_FR"] == 0.0
    assert len(registered_adaptation_arms()) == 5
