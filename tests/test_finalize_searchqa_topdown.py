import json
import random
from pathlib import Path

from scripts.tools.finalize_searchqa_topdown import (
    fallback_subset_seed,
    finalize_topdown,
    select_subset_ids,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_results(path: Path, hard_by_id: dict[str, int], soft_by_id=None) -> None:
    soft_by_id = soft_by_id or hard_by_id
    path.write_text(
        "".join(
            json.dumps({
                "id": sample_id,
                "hard": hard,
                "soft": soft_by_id[sample_id],
            }) + "\n"
            for sample_id, hard in hard_by_id.items()
        ),
        encoding="utf-8",
    )


def test_finalize_topdown_selects_shared_subset_and_builds_deduped_combo(tmp_path):
    ids = [f"q{index:02d}" for index in range(8)]
    subset_ids, seed = select_subset_ids(
        ids, base_seed=42, global_step=6, subset_size=4,
    )
    expected = sorted(ids)
    random.Random(fallback_subset_seed(42, 6)).shuffle(expected)
    assert seed == 42 + 6 * 1_000_003 + 97
    assert subset_ids == expected[:4]

    parent_skill = tmp_path / "parent.md"
    root_skill = tmp_path / "root.md"
    parent_skill.write_text("# Skill\n", encoding="utf-8")
    root_skill.write_text("# Skill\n\nROOT\n", encoding="utf-8")
    artifact_path = tmp_path / "artifact.json"
    duplicate_edit = {"op": "append", "content": "Shared useful rule."}
    _write_json(artifact_path, {
        "root_children_level": "mid",
        "root_child_patches": [
            {
                "mid_id": "M1",
                "leaf_ids": ["L1", "L2"],
                "support_count": 3,
                "edits": [duplicate_edit],
            },
            {
                "mid_id": "M2",
                "leaf_ids": ["L3", "L4"],
                "support_count": 2,
                "edits": [duplicate_edit],
            },
            {
                "mid_id": "M3",
                "leaf_ids": ["L5"],
                "support_count": 1,
                "edits": [{"op": "append", "content": "Bad rule."}],
            },
        ],
    })

    parent_hard = {sample_id: 0 for sample_id in ids}
    root_hard = dict(parent_hard)
    child_hard = {
        "M1": dict(parent_hard),
        "M2": dict(parent_hard),
        "M3": dict(parent_hard),
    }
    child_hard["M1"][subset_ids[0]] = 1
    child_hard["M2"][subset_ids[1]] = 1
    parent_results = tmp_path / "parent.jsonl"
    root_results = tmp_path / "root.jsonl"
    _write_results(parent_results, parent_hard)
    _write_results(root_results, root_hard)
    child_paths = {}
    for child_id, values in child_hard.items():
        path = tmp_path / f"{child_id}.jsonl"
        _write_results(path, values)
        child_paths[child_id] = path

    out_dir = tmp_path / "out"
    report = finalize_topdown(
        merge_artifact_path=artifact_path,
        parent_skill_path=parent_skill,
        root_skill_path=root_skill,
        parent_results_path=parent_results,
        root_results_path=root_results,
        child_result_paths=child_paths,
        out_dir=out_dir,
        base_seed=42,
        global_step=6,
        subset_size=4,
        mixed_weight=0.5,
    )

    assert report["offline_only"] is True
    assert report["protocol"]["subset_ids"] == subset_ids
    assert report["kept_child_ids"] == ["M1", "M2"]
    assert report["complementarity"]["topdown_complementarity_pass"] is True
    assert report["reconcile"]["n_input_edits"] == 2
    assert report["reconcile"]["n_output_edits"] == 1
    assert report["combination"]["n_edits"] == 1
    candidate = (out_dir / "topdown_combination_candidate.md").read_text(
        encoding="utf-8"
    )
    assert candidate.count("Shared useful rule.") == 1
    assert "Bad rule." not in candidate
    saved = json.loads(
        (out_dir / "topdown_finalize_report.json").read_text(encoding="utf-8")
    )
    assert saved["skills"]["combination_hash"] == report["skills"]["combination_hash"]
