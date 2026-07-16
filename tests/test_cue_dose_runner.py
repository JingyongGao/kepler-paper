from pathlib import Path

from kepler.cue_dose_runner import (
    _checkpoint_path,
    _evaluation_path,
    _tree_sha256,
)


def test_phase_paths_are_unambiguous():
    root = Path("run")
    assert _checkpoint_path(root, 5_000_000, "pre_switch").name.endswith(
        "_pre_switch.pt"
    )
    assert _checkpoint_path(root, 5_000_000, "post_switch").name.endswith(
        "_post_switch.pt"
    )
    assert _evaluation_path(root, 0, "initialization").name == "initialization.cue.eval.json"


def test_tree_hash_is_key_order_invariant_and_value_sensitive():
    assert _tree_sha256({"a": [1, 2], "b": 3}) == _tree_sha256(
        {"b": 3, "a": [1, 2]}
    )
    assert _tree_sha256({"a": [1, 2]}) != _tree_sha256({"a": [1, 3]})
