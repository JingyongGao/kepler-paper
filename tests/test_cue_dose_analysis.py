from kepler.cue_dose_analysis import classify_transition, exact_blocked_page_test


def _record(exposure, r, linear, center=0.0, phase="standard"):
    return {
        "exposed_pairs": exposure,
        "phase": phase,
        "R_full": r,
        "center_accuracy_full": center,
        "layer_6_linear_accuracy": linear,
    }


def _trajectory(pass_at):
    exposures = (2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_500_000, 10_000_000)
    records = [
        _record(0, 1.0, None, phase="initialization"),
        _record(500_000, 1.0, None),
        _record(1_000_000, 1.0, None),
    ]
    for exposure in exposures:
        passed = exposure >= pass_at
        phase = "pre_switch" if exposure == 5_000_000 else "standard"
        records.append(_record(exposure, 0.05 if passed else 1.0, 0.95 if passed else 0.01, 0.95 if passed else 0.01, phase))
        if exposure == 5_000_000:
            records.append(_record(exposure, 0.05 if passed else 1.0, 0.95 if passed else 0.01, 0.95 if passed else 0.01, "post_switch"))
    records.append(_record(5_500_000, 0.5, None))
    return records


def test_transition_requires_retention_and_does_not_initiate_at_terminal_point():
    onset = classify_transition(_trajectory(5_000_000))
    assert onset["onset_exposure"] == 5_000_000
    assert onset["establishment_exposure"] == 5_000_000

    terminal = classify_transition(_trajectory(10_000_000))
    assert terminal["onset_exposure"] is None
    assert terminal["onset_ordinal"] == 8
    assert terminal["establishment_exposure"] == 10_000_000


def test_exact_page_test_detects_perfect_order_and_handles_ties():
    perfect = {
        seed: {"C0": 8, "C1": 6, "C2": 4, "C3": 2} for seed in range(5)
    }
    result = exact_blocked_page_test(perfect)
    assert result["exact_permutation_count"] == 24**5
    assert result["exact_one_sided_p"] == 1 / 24**5

    tied = {seed: {"C0": 8, "C1": 8, "C2": 4, "C3": 4} for seed in range(5)}
    tied_result = exact_blocked_page_test(tied)
    assert tied_result["exact_permutation_count"] == 24**5
    assert all(rank * 2 == int(rank * 2) for rank in tied_result["block_ranks"][0])
