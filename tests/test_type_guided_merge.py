from __future__ import annotations

import threading
import time

from skillopt.config import flatten_config
from skillopt.gradient import type_guided_merge


def test_type_guided_config_flattens_from_optimizer_section():
    flat = flatten_config({
        "optimizer": {
            "type_guided_min_support": 3,
            "type_guided_max_leaf_groups": 5,
            "type_guided_tree_depth": 3,
            "type_guided_leaf_fallback": True,
            "type_guided_leaf_merge_workers": 4,
            "type_guided_mid_merge_workers": 2,
            "type_guided_merge_strategy": "concat",
            "type_guided_grouping_mode": "random",
            "type_guided_grouping_seed": 123,
        }
    })

    assert flat["type_guided_min_support"] == 3
    assert flat["type_guided_max_leaf_groups"] == 5
    assert flat["type_guided_tree_depth"] == 3
    assert flat["type_guided_leaf_fallback"] is True
    assert flat["type_guided_leaf_merge_workers"] == 4
    assert flat["type_guided_mid_merge_workers"] == 2
    assert flat["type_guided_merge_strategy"] == "concat"
    assert flat["type_guided_grouping_mode"] == "random"
    assert flat["type_guided_grouping_seed"] == 123


def test_generic_fallback_config_flattens_from_optimizer_section():
    flat = flatten_config({
        "optimizer": {
            "type_guided_fallback_enabled": True,
            "type_guided_fallback_max_hops": 1,
            "type_guided_fallback_allow_leaf": False,
        }
    })

    assert flat["type_guided_fallback_enabled"] is True
    assert flat["type_guided_fallback_max_hops"] == 1
    assert flat["type_guided_fallback_allow_leaf"] is False


def test_legacy_fallback_override_updates_canonical_key():
    flat = flatten_config({
        "optimizer": {
            "type_guided_fallback_enabled": True,
            "type_guided_leaf_fallback": False,
        }
    })

    assert flat["type_guided_leaf_fallback"] is False
    assert flat["type_guided_fallback_enabled"] is False


def test_type_guided_merge_fallback_builds_leaf_and_root(monkeypatch):
    def fail_chat(*args, **kwargs):
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fail_chat)

    patch = {
        "reasoning": "two related failures",
        "edits": [
            {
                "op": "append",
                "content": "Check explicit constraints before final answer.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "support_sample_ids": ["a", "b"],
            },
            {
                "op": "append",
                "content": "Verify every listed constraint.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "support_sample_ids": ["b", "c"],
            },
        ],
    }

    root, artifact = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=2,
        max_leaf_groups=8,
        verbose=False,
    )

    assert artifact["raw_edit_count"] == 2
    assert len(artifact["kept_groups"]) == 1
    assert artifact["kept_groups"][0]["question_type"] == "explicit_constraint_following"
    assert len(artifact["leaf_patches"]) == 1
    assert len(root["edits"]) == 2
    assert all(edit["leaf_ids"] == ["L1"] for edit in root["edits"])


def test_min_support_one_admits_singleton_as_normal_leaf(monkeypatch):
    def fail_chat(*args, **kwargs):
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fail_chat)
    patch = {
        "edits": [{
            "op": "append",
            "content": "Check the cited value before calculating.",
            "question_type": "derived_numeric_answer",
            "revision_type": "operand_verification",
            "support_sample_ids": ["only-example"],
        }],
    }

    _root, support_one = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=1,
        low_support_fallback=False,
        verbose=False,
    )
    _root, support_two = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=2,
        low_support_fallback=False,
        verbose=False,
    )

    assert len(support_one["leaf_patches"]) == 1
    assert support_one["leaf_patches"][0]["support_count"] == 1
    assert support_one["dropped_groups"] == []
    assert support_two["leaf_patches"] == []
    assert support_two["dropped_groups"][0]["drop_reason"] == "support<2"


def test_type_guided_merge_rejects_non_dict_llm_edits(monkeypatch):
    def bad_chat(*args, **kwargs):
        return '{"reasoning": "bad but valid json", "edits": ["bad", null]}', {}

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", bad_chat)

    patch = {
        "reasoning": "one failure",
        "edits": [
            {
                "op": "append",
                "content": "Check explicit constraints.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "support_sample_ids": ["a", "b"],
            },
        ],
    }

    root, artifact = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=2,
        verbose=False,
    )

    assert len(artifact["leaf_patches"]) == 1
    assert root["edits"][0]["content"] == "Check explicit constraints."


def test_type_guided_support_counts_mix_ids_and_counts(monkeypatch):
    def fail_chat(*args, **kwargs):
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fail_chat)

    patch = {
        "reasoning": "mixed support evidence",
        "edits": [
            {
                "op": "append",
                "content": "Check explicit constraints.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "support_sample_ids": ["a", "b"],
            },
            {
                "op": "append",
                "content": "Verify listed constraints.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "support_count": 3,
            },
        ],
    }

    _root, artifact = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=4,
        verbose=False,
    )

    assert len(artifact["kept_groups"]) == 1
    assert artifact["kept_groups"][0]["support_count"] == 5


def test_type_guided_tree_depth_three_builds_mid_nodes_with_fallback(monkeypatch):
    def fail_chat(*args, **kwargs):
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fail_chat)

    patch = {
        "reasoning": "two repair families",
        "edits": [
            {
                "op": "append",
                "content": "Verify every explicit constraint.",
                "question_type": "explicit_constraint_following",
                "revision_type": "constraint_verification",
                "repair_signature": "verify constraints",
                "support_sample_ids": ["a", "b"],
            },
            {
                "op": "append",
                "content": "Check evidence before answering.",
                "question_type": "evidence_grounded_answering",
                "revision_type": "evidence_checking",
                "repair_signature": "check evidence",
                "support_sample_ids": ["c", "d"],
            },
        ],
    }

    root, artifact = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=1,
        max_leaf_groups=8,
        tree_depth=3,
        leaf_merge_workers=2,
        mid_merge_workers=2,
        verbose=False,
    )

    assert artifact["tree_depth"] == 3
    assert artifact["root_children_level"] == "mid"
    assert len(artifact["leaf_patches"]) == 2
    assert len(artifact["mid_patches"]) == 2
    assert artifact["root_child_patches"] == artifact["mid_patches"]
    assert len(root["edits"]) == 2
    assert all(edit.get("mid_ids") for edit in root["edits"])
    assert artifact["settings"]["leaf_merge_workers"] == 2
    assert artifact["settings"]["mid_merge_workers"] == 2


def test_patchtree_leaf_merges_execute_concurrently(monkeypatch):
    lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_fail_chat(*, stage, **kwargs):
        nonlocal active, max_active
        if stage == "type_guided_leaf":
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
        raise RuntimeError("use deterministic fallback")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", slow_fail_chat)
    patch = {
        "edits": [
            {
                "op": "append",
                "content": "Verify constraints.",
                "question_type": "constraint_question",
                "revision_type": "constraint_check",
                "support_count": 2,
            },
            {
                "op": "append",
                "content": "Verify evidence.",
                "question_type": "evidence_question",
                "revision_type": "evidence_check",
                "support_count": 2,
            },
        ]
    }

    _root, artifact = type_guided_merge.build_patchtree(
        "Initial skill",
        [patch],
        min_support=1,
        allow_open_types=True,
        leaf_merge_workers=2,
        verbose=False,
    )

    assert max_active == 2
    assert artifact["settings"]["leaf_merge_workers"] == 2


def test_concat_strategy_skips_root_fusion(monkeypatch):
    stages: list[str] = []

    def fake_chat(*, stage, **kwargs):
        stages.append(stage)
        raise RuntimeError("use deterministic leaf fallback")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fake_chat)
    patch = {
        "edits": [
            {
                "op": "append",
                "content": "Repair A.",
                "question_type": "type_a",
                "revision_type": "repair_a",
                "support_count": 1,
            },
            {
                "op": "append",
                "content": "Repair B.",
                "question_type": "type_b",
                "revision_type": "repair_b",
                "support_count": 1,
            },
        ]
    }

    root, artifact = type_guided_merge.build_patchtree(
        "",
        [patch],
        min_support=1,
        allow_open_types=True,
        merge_strategy="concat",
        verbose=False,
    )

    assert stages == ["type_guided_leaf", "type_guided_leaf"]
    assert artifact["hierarchy"]["builder"] == "concat"
    assert artifact["settings"]["merge_strategy"] == "concat"
    assert [edit["content"] for edit in root["edits"]] == ["Repair A.", "Repair B."]


def test_depth_one_concat_skips_direct_root_fusion(monkeypatch):
    stages: list[str] = []

    def fake_chat(*, stage, **kwargs):
        stages.append(stage)
        raise AssertionError("concat must not call the optimizer")

    monkeypatch.setattr(type_guided_merge, "chat_optimizer", fake_chat)
    patch = {
        "edits": [{
            "op": "append",
            "content": "Keep the direct repair.",
            "question_type": "math",
            "revision_type": "verify",
            "support_count": 1,
        }]
    }

    root, artifact = type_guided_merge.build_patchtree(
        "",
        [patch],
        min_support=1,
        tree_depth=1,
        merge_strategy="concat",
        allow_open_types=True,
        verbose=False,
    )

    assert stages == []
    assert artifact["settings"]["merge_strategy"] == "concat"
    assert [edit["content"] for edit in root["edits"]] == [
        "Keep the direct repair.",
    ]


def test_random_grouping_is_deterministic_and_size_matched(monkeypatch):
    monkeypatch.setattr(
        type_guided_merge,
        "chat_optimizer",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fallback")),
    )
    patch = {
        "edits": [
            {
                "op": "append",
                "content": f"Repair {idx}.",
                "question_type": "large" if idx < 3 else "small",
                "revision_type": "repair",
                "support_count": 1,
            }
            for idx in range(4)
        ]
    }

    _root_a, artifact_a = type_guided_merge.build_patchtree(
        "",
        [patch],
        min_support=1,
        allow_open_types=True,
        grouping_mode="random",
        grouping_seed=7,
        merge_strategy="concat",
        verbose=False,
    )
    _root_b, artifact_b = type_guided_merge.build_patchtree(
        "",
        [patch],
        min_support=1,
        allow_open_types=True,
        grouping_mode="random",
        grouping_seed=7,
        merge_strategy="concat",
        verbose=False,
    )

    assert [len(group["edits"]) for group in artifact_a["groups"]] == [3, 1]
    assert artifact_a["groups"] == artifact_b["groups"]


def test_success_then_type_separates_failure_evidence(monkeypatch):
    monkeypatch.setattr(
        type_guided_merge,
        "chat_optimizer",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fallback")),
    )
    patch = {
        "edits": [
            {
                "op": "append",
                "content": "Repair consistent failure.",
                "question_type": "math",
                "revision_type": "verify",
                "evidence_success_rate": 0.0,
                "support_count": 1,
            },
            {
                "op": "append",
                "content": "Repair unstable failure.",
                "question_type": "math",
                "revision_type": "verify",
                "evidence_success_rate": 0.25,
                "support_count": 1,
            },
        ]
    }

    _root, artifact = type_guided_merge.build_patchtree(
        "",
        [patch],
        min_support=1,
        allow_open_types=True,
        grouping_mode="success_then_type",
        merge_strategy="concat",
        verbose=False,
    )

    assert {group["success_bucket"] for group in artifact["groups"]} == {
        "all_failure",
        "partial_success",
    }
