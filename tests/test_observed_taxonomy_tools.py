import json

import pytest

from scripts.tools.merge_observed_type_taxonomy import (
    choose_shortlist,
    pair_records,
    validate_manifests,
)
from scripts.tools.synthesize_observed_few_shots import validate_result


def _row(sample_id: str, split: str, status: str, q_i: float) -> dict:
    return {
        "env": "searchqa",
        "split": split,
        "id": sample_id,
        "question_type": "factoid",
        "revision_type": "answer_span_control",
        "has_reusable_revision": True,
        "outcome_status": status,
        "q_i": q_i,
        "condition": "when a short answer is requested",
        "boundary": "retain required qualifiers",
        "repair_signature": "return minimal supported span",
        "patch": {"op": "append", "content": "Return the minimal supported span."},
        "target_model": "Qwen/Qwen3.5-4B",
        "optimizer_source": "deepseek",
        "skill_hash": "abc",
        "repeats_requested": 3,
    }


def test_pair_ranking_prefers_unstable_and_replicated_evidence():
    rows = [
        _row("failure", "train", "failure", 0.0),
        _row("unstable", "val", "unstable", 1 / 3),
    ]
    pairs = pair_records(rows, max_evidence=3)
    assert pairs[0]["support_count"] == 2
    assert pairs[0]["evidence_grade"] == "B"
    assert pairs[0]["evidence"][0]["id"] == "unstable"
    assert choose_shortlist(pairs, max_few_shots=8) == pairs


def test_synthesized_example_recomputes_observed_evidence():
    stats = {
        ("factoid", "answer_span_control"): {
            "support_count": 3,
            "unstable_count": 2,
            "split_support": ["train", "val"],
        }
    }
    result = {
        "few_shots": [{
            "question_type": "factoid",
            "revision_type": "answer_span_control",
            "repair_signature": "return minimal supported span",
            "condition": "when a short answer is requested",
            "boundary": "retain required qualifiers",
            "patch": {"op": "append", "content": "Return the minimal supported span."},
            "source_pairs": [["factoid", "answer_span_control"]],
            "evidence_grade": "C",
            "selection_reason": "observed contrast",
        }]
    }
    validated = validate_result(result, stats, max_few_shots=8)
    assert validated[0]["evidence_grade"] == "A"
    assert validated[0]["observed_support_count"] == 3


def test_manifest_validation_requires_all_shards(tmp_path):
    rows = [_row("one", "train", "failure", 0.0)]
    audit_dir = tmp_path / "shard0"
    audit_dir.mkdir()
    taxonomy_path = audit_dir / "sample_taxonomy.jsonl"
    taxonomy_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (audit_dir / "summary.json").write_text(
        json.dumps({
            "env": "searchqa",
            "shard_count": 2,
            "shard_index": 0,
            "expected_samples": 1,
            "complete": True,
            "skill_hash": "abc",
            "repeats": 3,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete shard coverage"):
        validate_manifests([taxonomy_path], rows, {"searchqa"})
