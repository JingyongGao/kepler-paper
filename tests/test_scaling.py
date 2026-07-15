import numpy as np

from kepler.scaling import fit_model, prepare_scaling_data


def test_positive_shared_slope_fit_recovers_synthetic_m0():
    rows = []
    topologies = [(2, 4), (4, 4), (2, 64), (4, 64), (3, 16)]
    for index, (k, d) in enumerate(topologies):
        for replicate in range(2):
            track = f"k{k}d{d}r{replicate}"
            for n in (2e6, 10e6, 50e6):
                for tokens in (5e6, 20e6, 80e6):
                    response = 0.02 + index * 0.003 + (2e3 + index * 10) * n ** -0.5 + (1e3 + index * 5) * tokens ** -0.4
                    rows.append(
                        {"track_id": track, "k": k, "d": d, "N_core": n, "D": tokens, "R": response}
                    )
    data = prepare_scaling_data(rows)
    fit = fit_model(data, "M0")
    assert fit["success"]
    assert fit["mse"] < 1e-8
    assert fit["parameters"]["shared"]["a"] > 0
    assert fit["parameters"]["shared"]["b"] > 0
