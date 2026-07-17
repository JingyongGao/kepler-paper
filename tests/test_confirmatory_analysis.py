from kepler.cue_dose_analysis import exact_blocked_page_test


def test_confirmatory_exact_page_enumerates_all_six_to_eighth_assignments():
    perfect = {
        block: {"R0": 8, "R1": 5, "R2": 2}
        for block in range(8)
    }
    result = exact_blocked_page_test(perfect, ("R0", "R1", "R2"))
    assert result["exact_permutation_count"] == 6**8
    assert result["exact_one_sided_p"] == 1 / 6**8
