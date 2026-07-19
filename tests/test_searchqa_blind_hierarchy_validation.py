from argparse import Namespace

import pytest

from scripts.tools.validate_searchqa_blind_hierarchy import (
    HIERARCHY_SCHEMA_VERSION,
    build_variants,
    deterministic_variant_order,
    hierarchy_decision_metrics,
    hierarchy_fingerprint,
    parse_pair_specs,
    reusable_cached_result,
    stratified_select,
    summarize_cohort,
)


def _args(**overrides):
    values = {
        "target_model": "Qwen/Qwen3.5-4B",
        "target_base_url": "http://127.0.0.1:8000/v1",
        "target_temperature": 0.2,
        "target_timeout_seconds": 300,
        "target_max_tokens": 4096,
        "repeats": 5,
        "batch_size": 100,
        "seed": 4242,
        "max_child_holdout": 40,
        "max_parent_reference": 20,
    }
    values.update(overrides)
    return Namespace(**values)


def _types():
    return {
        "P": {
            "type_id": "P",
            "revision_type": "parent",
            "shared_patch": {"op": "append", "content": "PARENT"},
        },
        "C": {
            "type_id": "C",
            "revision_type": "child",
            "shared_patch": {"op": "append", "content": "CHILD"},
        },
        "X": {
            "type_id": "X",
            "revision_type": "control",
            "shared_patch": {"op": "append", "content": "CONTROL"},
        },
    }


def test_pair_parser_supports_optional_control_and_rejects_invalid_specs():
    assert parse_pair_specs(["P:C:X", "P:Y"]) == [
        {"parent_type_id": "P", "child_type_id": "C", "control_type_id": "X"},
        {"parent_type_id": "P", "child_type_id": "Y", "control_type_id": None},
    ]
    with pytest.raises(ValueError):
        parse_pair_specs(["P:P"])
    with pytest.raises(ValueError):
        parse_pair_specs(["P:C:C"])
    with pytest.raises(ValueError):
        parse_pair_specs(["P:C", "P:C"])


def test_stratified_selection_covers_strata_and_is_deterministic():
    cards = {
        "test::a": {"outcome_status": "failure"},
        "test::b": {"outcome_status": "failure"},
        "test::c": {"outcome_status": "unstable"},
        "train::d": {"outcome_status": "failure"},
        "val::e": {"outcome_status": "unstable"},
    }
    keys = list(cards)
    selected = stratified_select(keys, cards_by_key=cards, limit=4, seed=9)
    assert selected == stratified_select(keys, cards_by_key=cards, limit=4, seed=9)
    assert len(selected) == 4
    assert len({(key.split("::")[0], cards[key]["outcome_status"]) for key in selected}) == 4


def test_variants_compile_parent_and_child_in_the_declared_order():
    types = _types()
    variants = build_variants("BASE", types["P"], types["C"], types["X"])
    assert variants["initial"] == "BASE"
    assert variants["parent"].rstrip().endswith("PARENT")
    assert variants["child"].rstrip().endswith("CHILD")
    assert variants["parent_plus_child"].index("PARENT") < variants["parent_plus_child"].index("CHILD")
    assert variants["unrelated_control"].rstrip().endswith("CONTROL")


def test_execution_order_is_deterministic_and_context_sensitive():
    names = ["initial", "parent", "child", "parent_plus_child", "unrelated_control"]
    values = dict(
        seed=42,
        pair_name="P__C",
        cohort_name="child_holdout",
        split="test",
        chunk_index=0,
    )
    first = deterministic_variant_order(names, **values)
    assert first == deterministic_variant_order(names, **values)
    assert sorted(first) == sorted(names)
    changed = deterministic_variant_order(names, **{**values, "chunk_index": 1})
    assert first != changed


def test_summary_reports_raw_winner_and_child_control_difference():
    summary = summarize_cohort(
        ["test::a", "test::b"],
        {
            "initial": {"test::a": 0.0, "test::b": 0.5},
            "parent": {"test::a": 0.5, "test::b": 0.5},
            "child": {"test::a": 1.0, "test::b": 0.5},
            "parent_plus_child": {"test::a": 1.0, "test::b": 0.0},
            "unrelated_control": {"test::a": 0.0, "test::b": 0.5},
        },
    )
    assert summary["metric_winners"] == ["child"]
    assert summary["variants"]["child"]["accuracy"] == pytest.approx(0.75)
    assert summary["variants"]["child"]["delta_vs_initial"] == pytest.approx(0.5)
    assert summary["child_vs_unrelated_control"]["child_minus_unrelated_control"] == pytest.approx(0.5)
    assert summary["winner_interpretation"] == "descriptive_raw_accuracy_only"


def test_hierarchy_decision_metrics_preserve_all_requested_raw_contrasts():
    summaries = {
        "child_holdout": {
            "variants": {
                "initial": {"accuracy": 0.1},
                "parent": {"accuracy": 0.3},
                "child": {"accuracy": 0.5},
                "parent_plus_child": {"accuracy": 0.6},
                "unrelated_control": {"accuracy": 0.2},
            },
        },
        "parent_reference": {
            "variants": {
                "initial": {"accuracy": 0.4},
                "parent": {"accuracy": 0.6},
                "child": {"accuracy": 0.35},
                "parent_plus_child": {"accuracy": 0.58},
                "unrelated_control": {"accuracy": 0.4},
            },
        },
    }
    metrics = hierarchy_decision_metrics(summaries)
    assert metrics == pytest.approx({
        "child_holdout_child_minus_parent": 0.2,
        "child_holdout_combined_minus_parent": 0.3,
        "child_holdout_combined_minus_child": 0.1,
        "child_holdout_child_minus_unrelated_control": 0.3,
        "parent_reference_child_minus_initial": -0.05,
        "parent_reference_combined_minus_parent": -0.02,
    })


def test_fingerprint_and_cache_cover_pair_cohorts_and_sampling():
    pair = {"parent_type_id": "P", "child_type_id": "C", "control_type_id": "X"}
    values = {
        "pair": pair,
        "type_items": _types(),
        "initial_skill": "BASE",
        "cohorts": {"child_holdout": ["test::a"], "parent_reference": ["val::b"]},
        "selected_items": {
            "test::a": {"id": "a", "question": "A"},
            "val::b": {"id": "b", "question": "B"},
        },
        "selected_cards": {
            "test::a": {"sample_key": "test::a", "outcome_status": "failure"},
            "val::b": {"sample_key": "val::b", "outcome_status": "unstable"},
        },
        "runtime_config_sha256": "cfg",
        "args": _args(),
    }
    fingerprint = hierarchy_fingerprint(**values)
    assert hierarchy_fingerprint(**values) == fingerprint
    assert hierarchy_fingerprint(**{**values, "args": _args(repeats=3)}) != fingerprint
    assert hierarchy_fingerprint(
        **{**values, "cohorts": {"child_holdout": [], "parent_reference": ["val::b"]}}
    ) != fingerprint
    assert reusable_cached_result(
        {
            "schema_version": HIERARCHY_SCHEMA_VERSION,
            "validation_fingerprint": fingerprint,
        },
        fingerprint,
    )
    assert not reusable_cached_result(
        {
            "schema_version": HIERARCHY_SCHEMA_VERSION,
            "validation_fingerprint": "stale",
        },
        fingerprint,
    )
