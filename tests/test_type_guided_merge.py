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
        }
    })

    assert flat["type_guided_min_support"] == 3
    assert flat["type_guided_max_leaf_groups"] == 5
    assert flat["type_guided_tree_depth"] == 3
    assert flat["type_guided_leaf_fallback"] is True
    assert flat["type_guided_leaf_merge_workers"] == 4
    assert flat["type_guided_mid_merge_workers"] == 2


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
