import json
from pathlib import Path

from scripts.tools.analyze_searchqa_tree_verdict import compare
from scripts.tools.eval_searchqa_skill_manifest import _validate_results
from scripts.tools.prepare_searchqa_tree_verdict import (
    audit_run,
    choose_fallback_step,
    choose_main_step,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_step(
    run: Path,
    step: int,
    *,
    action: str,
    n_leaves: int,
    groups: list[list[int]],
) -> None:
    step_dir = run / "steps" / f"step_{step:04d}"
    records = [
        {"record_id": f"R{index:04d}", "patch": {"op": "append", "content": "x"}}
        for index in range(1, n_leaves + 1)
    ]
    leaves = [
        {
            "leaf_id": f"L{index}",
            "record_ids": [f"R{index:04d}"],
            "edits": [{
                "op": "append",
                "content": f"leaf {index}",
                "record_ids": [f"R{index:04d}"],
            }],
        }
        for index in range(1, n_leaves + 1)
    ]
    mid_groups = [
        {
            "mid_id": f"M{index}",
            "leaf_ids": [f"L{leaf_index}" for leaf_index in leaf_indices],
        }
        for index, leaf_indices in enumerate(groups, 1)
    ]
    mid_patches = [
        {
            **group,
            "edits": [{
                "op": "append",
                "content": "mid",
                "leaf_ids": group["leaf_ids"],
            }],
        }
        for group in mid_groups
    ]
    _write_json(step_dir / "type_guided_v2_patch_records.json", records)
    _write_json(step_dir / "type_guided_v2_merge_artifact.json", {
        "leaf_patches": leaves,
        "mid_groups": mid_groups,
        "mid_patches": mid_patches,
        "root_children_level": "mid",
        "root_patch": {
            "edits": [{
                "op": "append",
                "content": "root",
                "mid_ids": [row["mid_id"] for row in mid_groups],
            }],
        },
    })
    _write_json(step_dir / "step_record.json", {
        "action": action,
        "edit_budget": 2,
    })


def test_audit_and_registered_step_selection(tmp_path):
    run = tmp_path / "run"
    _write_step(
        run,
        1,
        action="accept_new_best",
        n_leaves=6,
        groups=[[1, 2], [3, 4], [5], [6]],
    )
    _write_step(
        run,
        2,
        action="reject",
        n_leaves=6,
        groups=[[1, 2, 3], [4, 5, 6]],
    )
    rows = audit_run(run)
    assert rows[0]["n_multi_leaf_mids"] == 2
    assert rows[0]["multi_leaf_coverage"] == 4 / 6
    assert rows[0]["leaf_record_coverage"] == 1.0
    assert rows[0]["root_child_coverage"] == 1.0
    assert choose_main_step(rows, "auto") == 1
    assert choose_fallback_step(rows, "auto") == 2


def test_result_validation_rejects_duplicates_and_failures():
    expected = ["a", "b"]
    assert _validate_results([
        {"id": "a", "agent_ok": True},
        {"id": "b", "agent_ok": True},
    ], expected)["valid"]
    invalid = _validate_results([
        {"id": "a", "agent_ok": True},
        {"id": "a", "agent_ok": False},
    ], expected)
    assert not invalid["valid"]
    assert invalid["duplicates"] == ["a"]
    assert invalid["missing_ids"] == ["b"]


def test_paired_comparison_counts_fixed_and_broken():
    reference = {
        "a": {"hard": 0},
        "b": {"hard": 1},
        "c": {"hard": 1},
    }
    candidate = {
        "a": {"hard": 1},
        "b": {"hard": 0},
        "c": {"hard": 1},
    }
    result = compare(
        candidate,
        reference,
        candidate_name="tree",
        reference_name="flat",
    )
    assert result["n01_candidate_only"] == 1
    assert result["n10_reference_only"] == 1
    assert result["delta_correct"] == 0
    assert result["fixed_ids"] == ["a"]
    assert result["broken_ids"] == ["b"]
