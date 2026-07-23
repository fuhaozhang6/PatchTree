import json

import pytest

from skillopt.engine.trainer import (
    _fallback_node_skip_reason,
    _load_selection_result_cache,
    _save_selection_result_cache,
    _slice_selection_results,
    _type_guided_fallback_sample_seed,
    _recursive_fallback_children,
)
from skillopt.envs.searchqa.dataloader import SearchQADataLoader
from skillopt.utils import compute_score


def _loader_with_val_items(n: int = 20) -> SearchQADataLoader:
    loader = SearchQADataLoader(split_mode="split_dir")
    loader._splits = {
        "train": [],
        "val": [{"id": f"q{i:02d}"} for i in range(n)],
        "test": [],
    }
    return loader


def test_eval_random_subset_is_deterministic_and_does_not_change_prefix_mode():
    loader = _loader_with_val_items()

    prefix = loader.build_eval_batch(
        env_num=6, split="valid_seen", seed=42,
    )
    sample_a = loader.build_eval_batch(
        env_num=6, split="valid_seen", seed=42, random_sample=True,
    )
    sample_b = loader.build_eval_batch(
        env_num=6, split="valid_seen", seed=42, random_sample=True,
    )
    sample_c = loader.build_eval_batch(
        env_num=6, split="valid_seen", seed=43, random_sample=True,
    )

    assert [row["id"] for row in prefix.payload] == [
        "q00", "q01", "q02", "q03", "q04", "q05",
    ]
    assert sample_a.payload == sample_b.payload
    assert sample_a.payload != sample_c.payload
    assert sample_a.metadata["sampling"] == "random_without_replacement"
    assert prefix.metadata["sampling"] == "prefix"


def test_fallback_representative_subset_seed_is_step_invariant():
    assert _type_guided_fallback_sample_seed(42) == 139
    assert _type_guided_fallback_sample_seed(42) == 139
    assert _type_guided_fallback_sample_seed(43) == 140


def test_recursive_fallback_descends_to_direct_children_but_stops_at_leaf():
    leaf_a = {"node_id": "L1", "leaf_ids": ["L1"], "leaf_coverage": 1}
    leaf_b = {"node_id": "L2", "leaf_ids": ["L2"], "leaf_coverage": 1}
    parent = {
        "node_id": "N1",
        "child_ids": ["L1", "L2"],
        "leaf_ids": ["L1", "L2"],
        "leaf_coverage": 2,
    }
    registry = {"L1": leaf_a, "L2": leaf_b, "N1": parent}

    assert _recursive_fallback_children(
        parent, node_by_id=registry, min_leaf_coverage=1,
    ) == [leaf_a, leaf_b]
    assert _recursive_fallback_children(
        leaf_a, node_by_id=registry, min_leaf_coverage=1,
    ) == []
    assert _recursive_fallback_children(
        parent, node_by_id=registry, min_leaf_coverage=2,
    ) == []


def test_fallback_node_limits_hops_and_can_exclude_leaves():
    internal = {"node_id": "N1", "node_level": "internal"}
    leaf = {"node_id": "L1", "node_level": "leaf"}

    assert _fallback_node_skip_reason(
        internal, fallback_hop=1, max_hops=1, allow_leaf=True,
    ) == ""
    assert _fallback_node_skip_reason(
        internal, fallback_hop=2, max_hops=1, allow_leaf=True,
    ) == "beyond_max_hops"
    assert _fallback_node_skip_reason(
        leaf, fallback_hop=1, max_hops=-1, allow_leaf=False,
    ) == "leaf_fallback_disabled"
    assert _fallback_node_skip_reason(
        leaf, fallback_hop=3, max_hops=-1, allow_leaf=True,
    ) == ""


def test_full_selection_cache_can_score_the_exact_fallback_subset(tmp_path):
    full_results = [
        {"id": "q0", "hard": 1.0, "soft": 0.8, "unused": "large trace"},
        {"id": "q1", "hard": 0.0, "soft": 0.3},
        {"id": "q2", "hard": 1.0, "soft": 0.9},
    ]
    saved = _save_selection_result_cache(tmp_path, "abc123", full_results)
    loaded = _load_selection_result_cache(tmp_path, "abc123")
    subset, missing = _slice_selection_results(loaded, ["q2", "q0"])

    assert saved == loaded
    assert [row["id"] for row in subset] == ["q2", "q0"]
    assert missing == []
    assert compute_score(subset) == pytest.approx((1.0, 0.85))

    cache_path = tmp_path / "selection_result_cache" / "abc123.json"
    raw = json.loads(cache_path.read_text())
    assert set(raw[0]) == {"id", "hard", "soft"}


def test_selection_cache_reports_missing_ids_for_safe_subset_reroll(tmp_path):
    _save_selection_result_cache(
        tmp_path,
        "abc123",
        [{"id": "q0", "hard": 1.0, "soft": 1.0}],
    )
    subset, missing = _slice_selection_results(
        _load_selection_result_cache(tmp_path, "abc123"),
        ["q0", "q1"],
    )

    assert [row["id"] for row in subset] == ["q0"]
    assert missing == ["q1"]
