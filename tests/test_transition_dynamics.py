from pathlib import Path

import pytest

from kepler.transition_dynamics import (
    PROBE_EXPOSURES,
    _probe_points,
    _spectral_selection,
    effective_spectral_dimension,
    first_retained_threshold_exposure,
    frozen_scientific_paths,
)


def test_first_retained_threshold_uses_next_registered_checkpoint():
    values = [0.1, 0.6, 0.7, 0.2, 0.9, 0.1, 0.9]
    assert first_retained_threshold_exposure(values) == PROBE_EXPOSURES[1]
    assert first_retained_threshold_exposure([0.1] * 6 + [0.9]) is None


def test_effective_spectral_dimension_matches_registered_formula():
    assert effective_spectral_dimension([2.0, 2.0]) == pytest.approx(2.0)
    assert effective_spectral_dimension([0.0, 0.0]) is None


def test_probe_points_prefers_pre_switch_over_duplicate_post_switch():
    points = []
    for exposure in PROBE_EXPOSURES:
        phase = "pre_switch" if exposure == 5_000_000 else "standard"
        points.append({"exposed_pairs": exposure, "phase": phase, "probe_metrics": {"kept": True}})
    points.append({"exposed_pairs": 5_000_000, "phase": "post_switch", "probe_metrics": {"kept": False}})
    selected = _probe_points({"run_id": "test", "checkpoint_metrics": points})
    assert selected[3]["phase"] == "pre_switch"


def test_frozen_inventory_excludes_v1_3_outputs_and_paper_tmp(tmp_path: Path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "paper/tmp").mkdir(parents=True)
    frozen = tmp_path / "artifacts/frozen.json"
    frozen.write_text("frozen", encoding="utf-8")
    (tmp_path / "artifacts/representation_transition_dynamics.json").write_text("new", encoding="utf-8")
    (tmp_path / "paper/tmp/scratch.txt").write_text("scratch", encoding="utf-8")
    relative = [path.relative_to(tmp_path).as_posix() for path in frozen_scientific_paths(tmp_path)]
    assert relative == ["artifacts/frozen.json"]


def test_spectral_selection_covers_paired_r0_and_r2_roles(tmp_path: Path):
    selected = _spectral_selection(tmp_path)
    assert [(item["condition"], item["role"]) for item in selected] == [
        ("R0", "pre_transition"),
        ("R0", "transition"),
        ("R0", "late"),
        ("R2", "pre_transition"),
        ("R2", "transition"),
        ("R2", "post_transition"),
    ]
