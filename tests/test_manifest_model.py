from kepler.config import DEFAULT_CONFIG
from kepler.manifest import build_manifest
from kepler.model import KeplerTransformer, core_parameter_count_for_width, size_spec


def test_manifest_has_registered_hierarchy_and_reproduction_pairing():
    tracks = build_manifest()
    assert len(tracks) == 80
    assert len({track.track_id for track in tracks}) == 80
    assert sum(track.reproduction for track in tracks) == 20
    for track in tracks:
        if track.reproduction:
            paired = [
                candidate
                for candidate in tracks
                if not candidate.reproduction
                and candidate.k == track.k
                and candidate.d == track.d
                and candidate.target_core_parameters == track.target_core_parameters
                and candidate.rule_instance == track.rule_instance
            ]
            assert len(paired) == 1
            assert paired[0].rule == track.rule


def test_width_is_only_scaling_axis_and_counts_are_exact():
    specs = [size_spec(target, DEFAULT_CONFIG.model) for target in DEFAULT_CONFIG.model.target_core_parameters]
    assert [spec.width for spec in specs] == sorted(spec.width for spec in specs)
    assert len({spec.width for spec in specs}) == 4
    for spec in specs:
        assert spec.width % DEFAULT_CONFIG.model.n_heads == 0
        assert spec.actual_core_parameters == core_parameter_count_for_width(
            spec.width, DEFAULT_CONFIG.model
        )


def test_instantiated_parameter_count_matches_analytic_count():
    cfg = DEFAULT_CONFIG.model
    model = KeplerTransformer(32, cfg)
    assert model.core_parameter_count() == core_parameter_count_for_width(32, cfg)

