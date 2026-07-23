"""PatchTree node construction and executable Skill compilation."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.optimizer.update_modes import get_payload_items, payload_key
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


QUESTION_TYPES = {
    "explicit_constraint_following",
    "format_controlled_generation",
    "multi_step_reasoning",
    "evidence_grounded_answering",
    "comparison_and_selection",
    "ambiguous_intent_handling",
    "tool_use_decision",
    "other",
}

REVISION_TYPES = {
    "constraint_verification",
    "format_enforcement",
    "step_decomposition",
    "evidence_checking",
    "calculation_verification",
    "ambiguity_clarification",
    "overgeneralization_control",
    "answer_completeness_check",
    "other",
}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "other"


def _normalise_question_type(value: Any, *, allow_open: bool = False) -> str:
    slug = _slug(value)
    aliases = {
        "explicit_constraint": "explicit_constraint_following",
        "constraint_following": "explicit_constraint_following",
        "format_controlled": "format_controlled_generation",
        "format_generation": "format_controlled_generation",
        "reasoning": "multi_step_reasoning",
        "evidence_grounded": "evidence_grounded_answering",
        "evidence_based": "evidence_grounded_answering",
        "comparison": "comparison_and_selection",
        "selection": "comparison_and_selection",
        "ambiguous_intent": "ambiguous_intent_handling",
        "tool_use": "tool_use_decision",
    }
    slug = aliases.get(slug, slug)
    if allow_open and slug:
        return slug
    return slug if slug in QUESTION_TYPES else "other"


def _normalise_revision_type(value: Any, *, allow_open: bool = False) -> str:
    slug = _slug(value)
    aliases = {
        "constraint_checking": "constraint_verification",
        "verify_constraints": "constraint_verification",
        "format_control": "format_enforcement",
        "decomposition": "step_decomposition",
        "evidence_verification": "evidence_checking",
        "calculation_checking": "calculation_verification",
        "clarification": "ambiguity_clarification",
        "over_generalization_control": "overgeneralization_control",
        "completeness_check": "answer_completeness_check",
    }
    slug = aliases.get(slug, slug)
    if allow_open and slug:
        return slug
    return slug if slug in REVISION_TYPES else "other"


def _sample_ids(edit: dict) -> list[str]:
    raw = edit.get("support_sample_ids") or edit.get("sample_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        sid = str(item).strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _count_field(items: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(field) or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _sum_count_maps(items: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        mapping = item.get(field) if isinstance(item, dict) else None
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if key:
                counts[str(key)] = counts.get(str(key), 0) + count
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _leaf_distribution_fields(leaves: list[dict]) -> dict[str, dict[str, int]]:
    return {
        "member_question_type_counts": _sum_count_maps(leaves, "member_question_type_counts")
        or _count_field(leaves, "question_type"),
        "member_revision_type_counts": _sum_count_maps(leaves, "member_revision_type_counts")
        or _count_field(leaves, "revision_type"),
        "member_cluster_question_type_counts": _sum_count_maps(
            leaves, "member_cluster_question_type_counts",
        ) or _count_field(leaves, "cluster_question_type"),
        "member_cluster_revision_type_counts": _sum_count_maps(
            leaves, "member_cluster_revision_type_counts",
        ) or _count_field(leaves, "cluster_revision_type"),
    }


def _support_count(edit: dict) -> int:
    ids = _sample_ids(edit)
    if ids:
        return len(ids)
    try:
        return max(int(edit.get("support_count", 0) or 0), 1)
    except (TypeError, ValueError):
        return 1


def _copy_edit(
    edit: dict,
    *,
    leaf_id: str | None = None,
    allow_open_types: bool = False,
) -> dict:
    copied = dict(edit)
    copied["question_type"] = _normalise_question_type(
        copied.get("question_type"), allow_open=allow_open_types,
    )
    copied["revision_type"] = _normalise_revision_type(
        copied.get("revision_type"), allow_open=allow_open_types,
    )
    copied["source_type"] = copied.get("source_type") or "failure"
    copied["support_count"] = _support_count(copied)
    if _sample_ids(copied):
        copied["support_sample_ids"] = _sample_ids(copied)
    if leaf_id:
        copied["leaf_ids"] = [leaf_id]
    return copied


def _patch_edits(patch: dict | None, update_mode: str = "patch") -> list[dict]:
    return [
        item for item in get_payload_items(patch, update_mode)
        if isinstance(item, dict)
    ]


def _extract_edits(
    patches: list[dict],
    update_mode: str,
    *,
    allow_open_types: bool = False,
) -> list[dict]:
    edits: list[dict] = []
    for patch in patches:
        for item in _patch_edits(patch, update_mode):
            copied = _copy_edit(item, allow_open_types=allow_open_types)
            if not copied.get("question_type"):
                copied["question_type"] = "other"
            if not copied.get("revision_type"):
                copied["revision_type"] = "other"
            edits.append(copied)
    return edits


def _group_edits(
    edits: list[dict],
    *,
    allow_open_types: bool = False,
    group_by_cluster: bool = False,
    grouping_mode: str = "type",
    grouping_seed: int = 0,
) -> list[dict]:
    grouping_mode = str(grouping_mode or "type").strip().lower()
    if grouping_mode not in {"type", "random", "success_then_type"}:
        grouping_mode = "type"

    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    if grouping_mode == "random" and not group_by_cluster:
        # Match the type-group size distribution so the ablation changes only
        # membership, not the number or approximate size of leaf groups.
        typed: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for edit in edits:
            typed[(
                _normalise_question_type(
                    edit.get("question_type"), allow_open=allow_open_types,
                ),
                _normalise_revision_type(
                    edit.get("revision_type"), allow_open=allow_open_types,
                ),
            )].append(edit)
        target_sizes = [
            len(items)
            for _key, items in sorted(
                typed.items(),
                key=lambda kv: (
                    -sum(_support_count(edit) for edit in kv[1]),
                    kv[0],
                ),
            )
        ]
        shuffled = list(edits)
        random.Random(int(grouping_seed)).shuffle(shuffled)
        offset = 0
        for idx, size in enumerate(target_sizes, start=1):
            grouped[("random", f"G{idx:03d}")].extend(
                shuffled[offset:offset + size],
            )
            offset += size
    else:
        for edit in edits:
            cluster_id = str(edit.get("cluster_id") or "").strip()
            if group_by_cluster and cluster_id:
                key = ("cluster", cluster_id)
            elif grouping_mode == "success_then_type":
                try:
                    q_i = float(edit.get("evidence_success_rate", 0.0) or 0.0)
                except (TypeError, ValueError):
                    q_i = 0.0
                success_bucket = "all_failure" if q_i <= 0.0 else "partial_success"
                key = (
                    "success_then_type",
                    success_bucket,
                    _normalise_question_type(
                        edit.get("question_type"), allow_open=allow_open_types,
                    ),
                    _normalise_revision_type(
                        edit.get("revision_type"), allow_open=allow_open_types,
                    ),
                )
            else:
                key = (
                    "type",
                    _normalise_question_type(
                        edit.get("question_type"), allow_open=allow_open_types,
                    ),
                    _normalise_revision_type(
                        edit.get("revision_type"), allow_open=allow_open_types,
                    ),
                )
            grouped[key].append(edit)

    groups: list[dict] = []
    for idx, (key, items) in enumerate(
        sorted(grouped.items(), key=lambda kv: (-sum(_support_count(e) for e in kv[1]), kv[0])),
        start=1,
    ):
        if key[0] == "cluster":
            first = items[0] if items else {}
            question_type = _normalise_question_type(
                first.get("question_type"), allow_open=allow_open_types,
            )
            revision_type = _normalise_revision_type(
                first.get("revision_type"), allow_open=allow_open_types,
            )
            cluster_id = key[1]
            cluster_label = str(first.get("cluster_label") or cluster_id)
            cluster_question_type = str(first.get("cluster_question_type") or "")
            cluster_revision_type = str(first.get("cluster_revision_type") or "")
            cluster_source = str(first.get("cluster_source") or "")
            repair_signature = str(first.get("repair_signature") or "").strip()
            success_bucket = ""
            random_group_id = ""
        elif key[0] == "success_then_type":
            question_type = key[2]
            revision_type = key[3]
            cluster_id = ""
            cluster_label = ""
            cluster_question_type = ""
            cluster_revision_type = ""
            cluster_source = ""
            repair_signature = ""
            success_bucket = key[1]
            random_group_id = ""
        elif key[0] == "random":
            question_type = "mixed"
            revision_type = "mixed"
            cluster_id = ""
            cluster_label = ""
            cluster_question_type = ""
            cluster_revision_type = ""
            cluster_source = ""
            repair_signature = ""
            success_bucket = ""
            random_group_id = key[1]
        else:
            question_type = key[1]
            revision_type = key[2]
            cluster_id = ""
            cluster_label = ""
            cluster_question_type = ""
            cluster_revision_type = ""
            cluster_source = ""
            repair_signature = ""
            success_bucket = ""
            random_group_id = ""
        support_ids: list[str] = []
        seen: set[str] = set()
        support_without_ids = 0
        for edit in items:
            edit_ids = _sample_ids(edit)
            if not edit_ids:
                support_without_ids += _support_count(edit)
                continue
            for sid in edit_ids:
                if sid not in seen:
                    seen.add(sid)
                    support_ids.append(sid)
        support = len(support_ids) + support_without_ids
        group = {
            "leaf_id": f"L{idx}",
            "question_type": question_type,
            "revision_type": revision_type,
            "support_count": support,
            "support_sample_ids": support_ids,
            "edits": items,
        }
        if cluster_id:
            group["cluster_id"] = cluster_id
            group["cluster_label"] = cluster_label
            group["cluster_question_type"] = cluster_question_type
            group["cluster_revision_type"] = cluster_revision_type
            group["cluster_source"] = cluster_source
        group["member_question_type_counts"] = _count_field(items, "question_type")
        group["member_revision_type_counts"] = _count_field(items, "revision_type")
        group["member_cluster_question_type_counts"] = _count_field(items, "cluster_question_type")
        group["member_cluster_revision_type_counts"] = _count_field(items, "cluster_revision_type")
        if repair_signature:
            group["repair_signature"] = repair_signature
        if success_bucket:
            group["success_bucket"] = success_bucket
        if random_group_id:
            group["random_group_id"] = random_group_id
        groups.append(group)
    return groups


def _filter_groups(
    groups: list[dict],
    *,
    min_support: int,
    max_leaf_groups: int,
    low_support_fallback: bool = True,
) -> tuple[list[dict], list[dict]]:
    if not groups:
        return [], []
    kept = [
        group for group in groups
        if int(group.get("support_count", 0) or 0) >= min_support
    ]
    dropped = [
        {**group, "drop_reason": f"support<{min_support}"}
        for group in groups
        if int(group.get("support_count", 0) or 0) < min_support
    ]
    if low_support_fallback and not kept:
        kept = groups[:1]
        dropped = [
            {**group, "drop_reason": "low_support_fallback"}
            for group in groups[1:]
        ]
    if max_leaf_groups > 0 and len(kept) > max_leaf_groups:
        kept, extra = kept[:max_leaf_groups], kept[max_leaf_groups:]
        dropped.extend({**group, "drop_reason": f"beyond_max_leaf_groups={max_leaf_groups}"} for group in extra)
    return kept, dropped


def _component_patch(component: dict) -> dict | None:
    patch = component.get("patch") if isinstance(component, dict) else None
    if not isinstance(patch, dict):
        return None
    op = str(patch.get("op") or "").strip()
    target = str(patch.get("target") or "").strip()
    content = str(patch.get("content") or "").strip()
    if op not in {"append", "insert_after", "replace", "delete"}:
        return None
    if op != "delete" and not content:
        return None
    if op in {"insert_after", "replace", "delete"} and not target:
        return None
    condition = str(component.get("condition") or "").strip()
    if op == "delete" and condition:
        return None
    edit = {"op": op}
    if target:
        edit["target"] = target
    if op != "delete":
        compiled_content = f"When {condition}:\n\n{content}" if condition else content
        boundary = str(component.get("boundary") or "").strip()
        if boundary:
            compiled_content += f"\n\nDo not apply this rule when {boundary}."
        edit["content"] = compiled_content
    edit["condition"] = condition
    edit["boundary"] = str(component.get("boundary") or "").strip()
    source_child_ids = _as_clean_str_list(component.get("source_child_ids"))
    if source_child_ids:
        edit["source_child_ids"] = source_child_ids
    return edit


def _compile_node_result(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return None
    shared_core = result.get("shared_core")
    residuals = result.get("conditional_residuals")
    if shared_core is not None and not isinstance(shared_core, dict):
        return None
    if not isinstance(residuals, list):
        return None

    edits: list[dict] = []
    if isinstance(shared_core, dict):
        shared_edit = _component_patch(shared_core)
        if shared_edit is None:
            return None
        shared_edit["node_component"] = "shared_core"
        edits.append(shared_edit)

    clean_residuals: list[dict] = []
    for residual in residuals:
        if not isinstance(residual, dict):
            return None
        condition = str(residual.get("condition") or "").strip()
        if not condition:
            return None
        residual_edit = _component_patch(residual)
        if residual_edit is None:
            return None
        residual_edit["node_component"] = "conditional_residual"
        edits.append(residual_edit)
        clean_residuals.append(residual)

    if not edits:
        return None
    preserved = result.get("preserved_constraints")
    conflicts = result.get("unresolved_conflicts")
    if preserved is None:
        preserved = {}
    if conflicts is None:
        conflicts = []
    if not isinstance(preserved, dict) or not isinstance(conflicts, list):
        return None
    return {
        "reasoning": str(result.get("reasoning") or ""),
        "shared_core": shared_core,
        "conditional_residuals": clean_residuals,
        "preserved_constraints": preserved,
        "unresolved_conflicts": conflicts,
        "edits": edits,
    }


def _call_merge(
    *,
    skill_content: str,
    prompt_name: str,
    user_title: str,
    payload: Any,
    stage: str,
) -> dict | None:
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## {user_title}\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        response, _ = chat_optimizer(
            system=load_prompt(prompt_name),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage=stage,
        )
        result = extract_json(response)
        return _compile_node_result(result)
    except Exception:  # noqa: BLE001
        return None
    return None


def _fallback_leaf_patch(group: dict) -> dict:
    leaf_id = str(group.get("leaf_id", "L?"))
    edits = [
        _copy_edit(
            edit,
            leaf_id=leaf_id,
            allow_open_types=bool(group.get("allow_open_types", False)),
        )
        for edit in group.get("edits", [])
        if isinstance(edit, dict)
    ]
    for edit in edits:
        edit["question_type"] = group.get("question_type", "other")
        edit["revision_type"] = group.get("revision_type", "other")
        edit.setdefault("support_count", group.get("support_count", 1))
        if group.get("support_sample_ids"):
            edit.setdefault("support_sample_ids", group["support_sample_ids"])
        for key in ("cluster_question_type", "cluster_revision_type", "cluster_source"):
            if group.get(key):
                edit.setdefault(key, group[key])
    return {
        "reasoning": f"fallback leaf patch for {group.get('question_type')}/{group.get('revision_type')}",
        "shared_core": None,
        "conditional_residuals": [],
        "preserved_constraints": {},
        "unresolved_conflicts": ["leaf merge unavailable; original record edits preserved"],
        "edits": edits,
    }


def _build_leaf_patch(
    *,
    skill_content: str,
    group: dict,
) -> dict:
    result = _call_merge(
        skill_content=skill_content,
        prompt_name="type_guided_leaf",
        user_title=(
            f"Typed Leaf Group {group.get('leaf_id')} "
            f"({group.get('question_type')} / {group.get('revision_type')})"
        ),
        payload=group,
        stage="type_guided_leaf",
    )
    patch = result or _fallback_leaf_patch(group)
    leaf_id = str(group.get("leaf_id", "L?"))
    for edit in _patch_edits(patch):
        edit["question_type"] = group.get("question_type", "other")
        edit["revision_type"] = group.get("revision_type", "other")
        edit["source_type"] = edit.get("source_type") or "failure"
        edit["support_count"] = max(_support_count(edit), int(group.get("support_count", 1) or 1))
        edit["leaf_ids"] = list(dict.fromkeys([*(edit.get("leaf_ids") or []), leaf_id]))
        if group.get("support_sample_ids"):
            edit.setdefault("support_sample_ids", group["support_sample_ids"])
        for key in ("cluster_question_type", "cluster_revision_type", "cluster_source"):
            if group.get(key):
                edit.setdefault(key, group[key])
    return patch


def _fallback_parent_patch(
    child_patches: list[dict],
    *,
    allow_open_types: bool = False,
    reasoning: str = "fallback parent patch: concatenated child edits",
) -> dict:
    edits: list[dict] = []
    for child in child_patches:
        edits.extend(
            _copy_edit(edit, allow_open_types=allow_open_types)
            for edit in _patch_edits(child)
        )
    return {
        "reasoning": reasoning,
        "shared_core": None,
        "conditional_residuals": [],
        "preserved_constraints": {},
        "unresolved_conflicts": ["parent merge unavailable; child edits preserved"],
        "edits": edits,
    }


def _build_root_patch(
    *,
    skill_content: str,
    child_patches: list[dict],
    allow_open_types: bool = False,
    child_level: str = "leaf",
) -> dict:
    payload = [
        {
            "node_id": child.get("mid_id") or child.get("leaf_id") or child.get("record_id"),
            "node_level": child_level,
            "record_id": child.get("record_id"),
            "leaf_id": child.get("leaf_id"),
            "mid_id": child.get("mid_id"),
            "leaf_ids": child.get("leaf_ids"),
            "question_type": child.get("question_type"),
            "revision_type": child.get("revision_type"),
            "cluster_question_type": child.get("cluster_question_type", ""),
            "cluster_revision_type": child.get("cluster_revision_type", ""),
            "member_question_type_counts": child.get("member_question_type_counts", {}),
            "member_revision_type_counts": child.get("member_revision_type_counts", {}),
            "member_cluster_question_type_counts": child.get("member_cluster_question_type_counts", {}),
            "member_cluster_revision_type_counts": child.get("member_cluster_revision_type_counts", {}),
            "support_count": child.get("support_count"),
            "patch": {
                "reasoning": child.get("reasoning", ""),
                "shared_core": child.get("shared_core"),
                "conditional_residuals": child.get("conditional_residuals", []),
                "preserved_constraints": child.get("preserved_constraints", {}),
                "unresolved_conflicts": child.get("unresolved_conflicts", []),
                "edits": _patch_edits(child),
            },
        }
        for child in child_patches
    ]
    result = _call_merge(
        skill_content=skill_content,
        prompt_name="type_guided_root",
        user_title=f"Typed {child_level.title()} Patches to Merge into Root Candidate",
        payload=payload,
        stage="type_guided_root",
    )
    patch = result or _fallback_parent_patch(
        child_patches,
        allow_open_types=allow_open_types,
        reasoning=f"fallback root patch: concatenated typed {child_level} edits",
    )
    support_count = sum(int(child.get("support_count", 0) or 0) for child in child_patches)
    for edit in _patch_edits(patch):
        source_ids = _as_clean_str_list(edit.get("source_child_ids"))
        if child_level == "mid":
            edit["mid_ids"] = list(dict.fromkeys([
                *_as_clean_str_list(edit.get("mid_ids")), *source_ids,
            ]))
        elif child_level == "leaf":
            edit["leaf_ids"] = list(dict.fromkeys([
                *_as_clean_str_list(edit.get("leaf_ids")), *source_ids,
            ]))
        else:
            edit["record_ids"] = list(dict.fromkeys([
                *_as_clean_str_list(edit.get("record_ids")), *source_ids,
            ]))
        edit["source_type"] = edit.get("source_type") or "failure"
        edit["support_count"] = max(_support_count(edit), support_count or 1)
    return patch


def _build_conservative_root_patch(
    *,
    skill_content: str,
    child_patches: list[dict],
    allow_open_types: bool = False,
) -> dict:
    """Integrate top-frontier nodes without inventing a global abstraction."""
    child_by_id: dict[str, dict] = {}
    payload: list[dict] = []
    for idx, child in enumerate(child_patches, start=1):
        child_id = _node_id(child, f"C{idx}")
        child_by_id[child_id] = child
        payload.append({
            "child_id": child_id,
            "node_level": child.get("node_level", ""),
            "leaf_ids": _node_leaf_ids(child),
            "question_type": child.get("question_type", ""),
            "revision_type": child.get("revision_type", ""),
            "support_count": child.get("support_count", 0),
            "support_sample_ids": child.get("support_sample_ids", []),
            "reasoning": child.get("reasoning", ""),
            "boundary": child.get("boundary", ""),
            "edits": _patch_edits(child),
        })

    fallback = _fallback_parent_patch(
        child_patches,
        allow_open_types=allow_open_types,
        reasoning=(
            "fallback conservative root: deterministically preserve top-frontier "
            "edits after integration failed"
        ),
    )
    fallback["top_integration"] = {
        "mode": "conservative_root",
        "status": "deterministic_fallback",
        "n_children": len(child_patches),
    }
    if not child_by_id:
        return fallback

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        "## Unvalidated Top-Frontier Children\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_top_integrate"),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="type_guided_top_integrate",
        )
        parsed = extract_json(response)
        raw_edits = parsed.get("edits") if isinstance(parsed, dict) else None
        raw_dropped = (
            parsed.get("dropped_child_insights")
            if isinstance(parsed, dict)
            else None
        )
        if not isinstance(raw_edits, list):
            raise ValueError("missing edits")
        if not isinstance(raw_dropped, list):
            raw_dropped = []

        allowed_ids = set(child_by_id)
        covered_ids: set[str] = set()
        dropped_ids: set[str] = set()
        dropped: list[dict] = []
        for row in raw_dropped:
            if not isinstance(row, dict):
                continue
            source_ids = _as_clean_str_list(row.get("source_child_ids"))
            source_ids = [item for item in source_ids if item in allowed_ids]
            reason = str(row.get("reason") or "").strip()
            if source_ids and reason:
                dropped_ids.update(source_ids)
                dropped.append({
                    "source_child_ids": source_ids,
                    "reason": reason,
                })

        edits: list[dict] = []
        for raw in raw_edits:
            if not isinstance(raw, dict):
                continue
            op = str(raw.get("op") or "").strip()
            target = str(raw.get("target") or "").strip()
            content = str(raw.get("content") or "").strip()
            condition = str(raw.get("condition") or "").strip()
            boundary = str(raw.get("boundary") or "").strip()
            source_ids = [
                item
                for item in _as_clean_str_list(raw.get("source_child_ids"))
                if item in allowed_ids
            ]
            source_ids = list(dict.fromkeys(source_ids))
            if (
                op not in {"append", "insert_after", "replace", "delete"}
                or (op != "delete" and not content)
                or (op in {"insert_after", "replace", "delete"} and not target)
                or (op == "delete" and condition)
                or not source_ids
            ):
                continue

            if op != "delete":
                if condition:
                    content = f"When {condition}:\n\n{content}"
                if boundary:
                    content += f"\n\nDo not apply this rule when {boundary}."
            source_children = [child_by_id[item] for item in source_ids]
            support_ids = list(dict.fromkeys(
                support_id
                for child in source_children
                for edit in _patch_edits(child)
                for support_id in _sample_ids(edit)
            ))
            leaf_ids = list(dict.fromkeys(
                leaf_id
                for child in source_children
                for leaf_id in _node_leaf_ids(child)
            ))
            question_types = {
                str(child.get("question_type") or "").strip()
                for child in source_children
                if str(child.get("question_type") or "").strip()
            }
            revision_types = {
                str(child.get("revision_type") or "").strip()
                for child in source_children
                if str(child.get("revision_type") or "").strip()
            }
            edit = {
                "op": op,
                "source_child_ids": source_ids,
                "leaf_ids": leaf_ids,
                "condition": condition,
                "boundary": boundary,
                "question_type": (
                    next(iter(question_types)) if len(question_types) == 1 else "other"
                ),
                "revision_type": (
                    next(iter(revision_types)) if len(revision_types) == 1 else "other"
                ),
                "source_type": "failure",
                "support_count": max(
                    sum(int(child.get("support_count", 0) or 0) for child in source_children),
                    1,
                ),
            }
            if target:
                edit["target"] = target
            if op != "delete":
                edit["content"] = content
            if support_ids:
                edit["support_sample_ids"] = support_ids
            edits.append(edit)
            covered_ids.update(source_ids)

        if not edits:
            raise ValueError("no valid integrated edits")
        uncovered = allowed_ids - covered_ids - dropped_ids
        if uncovered:
            raise ValueError(
                f"top integration omitted children: {sorted(uncovered)}"
            )
        return {
            "reasoning": str(parsed.get("reasoning") or ""),
            "shared_core": None,
            "conditional_residuals": [],
            "preserved_constraints": {},
            "unresolved_conflicts": [
                f"dropped {','.join(row['source_child_ids'])}: {row['reason']}"
                for row in dropped
            ],
            "edits": edits,
            "top_integration": {
                "mode": "conservative_root",
                "status": "llm_integrated",
                "n_children": len(child_patches),
                "n_output_edits": len(edits),
                "dropped_child_insights": dropped,
            },
        }
    except Exception as exc:  # noqa: BLE001
        fallback["top_integration"]["error"] = repr(exc)
        return fallback


def _node_leaf_ids(patch: dict) -> list[str]:
    ids: list[str] = []
    raw_leaf_id = patch.get("leaf_id")
    if raw_leaf_id:
        ids.append(str(raw_leaf_id))
    raw_leaf_ids = patch.get("leaf_ids") or []
    if isinstance(raw_leaf_ids, str):
        raw_leaf_ids = [raw_leaf_ids]
    if isinstance(raw_leaf_ids, list):
        ids.extend(str(item) for item in raw_leaf_ids if str(item).strip())
    for edit in _patch_edits(patch):
        raw = edit.get("leaf_ids") if isinstance(edit, dict) else []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            ids.extend(str(item) for item in raw if str(item).strip())
    return list(dict.fromkeys(ids))


def _as_clean_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _leaf_card(leaf: dict) -> dict:
    return {
        "leaf_id": str(leaf.get("leaf_id") or ""),
        "question_type": leaf.get("question_type"),
        "revision_type": leaf.get("revision_type"),
        "cluster_question_type": leaf.get("cluster_question_type", ""),
        "cluster_revision_type": leaf.get("cluster_revision_type", ""),
        "repair_signature": leaf.get("repair_signature", ""),
        "support_count": leaf.get("support_count", 0),
        "support_sample_ids": leaf.get("support_sample_ids", []),
        "member_question_type_counts": leaf.get("member_question_type_counts", {}),
        "member_revision_type_counts": leaf.get("member_revision_type_counts", {}),
        "member_cluster_question_type_counts": leaf.get("member_cluster_question_type_counts", {}),
        "member_cluster_revision_type_counts": leaf.get("member_cluster_revision_type_counts", {}),
        "reasoning": leaf.get("reasoning", ""),
        "shared_core": leaf.get("shared_core"),
        "conditional_residuals": leaf.get("conditional_residuals", []),
        "preserved_constraints": leaf.get("preserved_constraints", {}),
        "unresolved_conflicts": leaf.get("unresolved_conflicts", []),
        "edits": _patch_edits(leaf),
    }


def _fallback_mid_groups(leaf_patches: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for leaf in leaf_patches:
        key = (
            str(leaf.get("revision_type") or "other"),
            str(leaf.get("repair_signature") or ""),
        )
        grouped[key].append(leaf)

    groups: list[dict] = []
    for idx, (_key, leaves) in enumerate(
        sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])),
        start=1,
    ):
        support_ids: list[str] = []
        for leaf in leaves:
            raw_ids = leaf.get("support_sample_ids") or []
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                support_ids.extend(str(item) for item in raw_ids if str(item).strip())
        support_ids = list(dict.fromkeys(support_ids))
        groups.append({
            "mid_id": f"M{idx}",
            "mid_label": " / ".join(str(x) for x in _key if x) or f"M{idx}",
            "leaf_ids": [str(leaf.get("leaf_id")) for leaf in leaves],
            "question_type": str(leaves[0].get("question_type") or "other") if leaves else "other",
            "revision_type": str(leaves[0].get("revision_type") or "other") if leaves else "other",
            "support_count": len(support_ids) or sum(int(leaf.get("support_count", 0) or 0) for leaf in leaves),
            "support_sample_ids": support_ids,
            "merge_rationale": "deterministic fallback grouped leaves by revision type and repair signature",
            "boundary": "",
            "source": "fallback",
            **_leaf_distribution_fields(leaves),
        })
    return groups


def _validate_mid_plan(result: dict | None, leaf_patches: list[dict]) -> tuple[list[dict], list[str]]:
    by_leaf_id = {
        str(leaf.get("leaf_id") or ""): leaf
        for leaf in leaf_patches
        if str(leaf.get("leaf_id") or "").strip()
    }
    if not isinstance(result, dict) or not isinstance(result.get("mid_nodes"), list):
        return _fallback_mid_groups(leaf_patches), ["invalid_schema"]

    used: set[str] = set()
    used_mid_ids: set[str] = set()
    errors: list[str] = []
    groups: list[dict] = []

    def unique_mid_id(raw: Any, fallback: str) -> str:
        base = str(raw or fallback).strip() or fallback
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") or fallback
        if not base.upper().startswith("M"):
            base = f"M_{base}"
        mid_id = base
        suffix = 2
        while mid_id in used_mid_ids:
            mid_id = f"{base}_{suffix}"
            suffix += 1
        used_mid_ids.add(mid_id)
        return mid_id

    counter = 1
    for raw in result.get("mid_nodes", []):
        if not isinstance(raw, dict):
            errors.append("mid_node_not_dict")
            continue
        raw_leaf_ids = raw.get("leaf_ids")
        if isinstance(raw_leaf_ids, str):
            raw_leaf_ids = [raw_leaf_ids]
        if not isinstance(raw_leaf_ids, list):
            errors.append("leaf_ids_not_list")
            continue
        leaf_ids: list[str] = []
        for leaf_id_raw in raw_leaf_ids:
            leaf_id = str(leaf_id_raw or "").strip()
            if not leaf_id or leaf_id not in by_leaf_id or leaf_id in used:
                continue
            used.add(leaf_id)
            leaf_ids.append(leaf_id)
        if not leaf_ids:
            continue
        leaves = [by_leaf_id[leaf_id] for leaf_id in leaf_ids]
        support_ids: list[str] = []
        for leaf in leaves:
            raw_ids = leaf.get("support_sample_ids") or []
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                support_ids.extend(str(item) for item in raw_ids if str(item).strip())
        support_ids = list(dict.fromkeys(support_ids))
        groups.append({
            "mid_id": unique_mid_id(raw.get("mid_id"), f"M{counter}"),
            "mid_label": str(raw.get("mid_label") or raw.get("label") or f"M{counter}"),
            "leaf_ids": leaf_ids,
            "question_type": str(raw.get("question_type") or leaves[0].get("question_type") or "other"),
            "revision_type": str(raw.get("revision_type") or leaves[0].get("revision_type") or "other"),
            "support_count": len(support_ids) or sum(int(leaf.get("support_count", 0) or 0) for leaf in leaves),
            "support_sample_ids": support_ids,
            "merge_rationale": str(raw.get("merge_rationale") or raw.get("rationale") or ""),
            "boundary": str(raw.get("boundary") or ""),
            "source": "llm",
            **_leaf_distribution_fields(leaves),
        })
        counter += 1

    missing = [leaf_id for leaf_id in by_leaf_id if leaf_id not in used]
    if missing:
        errors.append(f"unassigned={len(missing)}")
        for leaf_id in missing:
            leaf = by_leaf_id[leaf_id]
            groups.append({
                "mid_id": unique_mid_id("", f"M{counter}"),
                "mid_label": str(leaf.get("repair_signature") or leaf.get("revision_type") or leaf_id),
                "leaf_ids": [leaf_id],
                "question_type": str(leaf.get("question_type") or "other"),
                "revision_type": str(leaf.get("revision_type") or "other"),
                "support_count": int(leaf.get("support_count", 0) or 0),
                "support_sample_ids": leaf.get("support_sample_ids", []),
                "merge_rationale": "fallback singleton for unassigned leaf",
                "boundary": str(leaf.get("boundary") or ""),
                "source": "fallback_singleton",
                **_leaf_distribution_fields([leaf]),
            })
            counter += 1
    return groups, errors


def _build_mid_groups(
    *,
    skill_content: str,
    leaf_patches: list[dict],
    target_children: int = 3,
    max_children: int = 4,
) -> tuple[list[dict], dict]:
    if not leaf_patches:
        return [], {"enabled": True, "status": "empty", "errors": []}
    payload = {
        "n_leaf_patches": len(leaf_patches),
        "target_children": max(int(target_children or 3), 2),
        "max_children": max(int(max_children or 4), max(int(target_children or 3), 2)),
        "leaves": [_leaf_card(leaf) for leaf in leaf_patches],
    }
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        "## Typed Leaf Patches to Plan into Mid-Level Nodes\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    result: dict | None = None
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_mid_plan"),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="type_guided_mid_plan",
        )
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            result = parsed
    except Exception:  # noqa: BLE001
        result = None
    groups, errors = _validate_mid_plan(result, leaf_patches)
    return groups, {
        "enabled": True,
        "status": "ok" if not errors else "fallback" if errors == ["invalid_schema"] else "partial_fallback",
        "errors": errors,
        "raw_result": result,
        "groups": groups,
    }


def _build_mid_patch(
    *,
    skill_content: str,
    group: dict,
    leaf_patches_by_id: dict[str, dict],
    allow_open_types: bool = False,
) -> dict:
    mid_id = str(group.get("mid_id") or "M?")
    leaves = [
        leaf_patches_by_id[leaf_id]
        for leaf_id in group.get("leaf_ids", [])
        if leaf_id in leaf_patches_by_id
    ]
    payload = {
        **group,
        "leaves": [_leaf_card(leaf) for leaf in leaves],
    }
    result = _call_merge(
        skill_content=skill_content,
        prompt_name="type_guided_mid",
        user_title=f"Typed Mid-Level Group {mid_id}",
        payload=payload,
        stage="type_guided_mid",
    )
    patch = result or _fallback_parent_patch(
        leaves,
        allow_open_types=allow_open_types,
        reasoning=f"fallback mid patch for {mid_id}: concatenated leaf edits",
    )
    leaf_ids = list(dict.fromkeys([
        *[str(leaf_id) for leaf_id in group.get("leaf_ids", [])],
        *[leaf_id for leaf in leaves for leaf_id in _node_leaf_ids(leaf)],
    ]))
    for edit in _patch_edits(patch):
        edit["question_type"] = edit.get("question_type") or group.get("question_type", "other")
        edit["revision_type"] = edit.get("revision_type") or group.get("revision_type", "other")
        edit["source_type"] = edit.get("source_type") or "failure"
        edit["support_count"] = max(_support_count(edit), int(group.get("support_count", 1) or 1))
        edit["mid_ids"] = list(dict.fromkeys([*_as_clean_str_list(edit.get("mid_ids")), mid_id]))
        edit["leaf_ids"] = list(dict.fromkeys([*_as_clean_str_list(edit.get("leaf_ids")), *leaf_ids]))
        if group.get("support_sample_ids"):
            edit.setdefault("support_sample_ids", group["support_sample_ids"])
    patch["mid_id"] = mid_id
    patch["mid_label"] = group.get("mid_label", "")
    patch["leaf_ids"] = leaf_ids
    patch["question_type"] = group.get("question_type", "other")
    patch["revision_type"] = group.get("revision_type", "other")
    patch["support_count"] = group.get("support_count", 0)
    patch["support_sample_ids"] = group.get("support_sample_ids", [])
    for key in (
        "member_question_type_counts",
        "member_revision_type_counts",
        "member_cluster_question_type_counts",
        "member_cluster_revision_type_counts",
    ):
        patch[key] = group.get(key, {})
    patch["merge_rationale"] = group.get("merge_rationale", "")
    patch["boundary"] = group.get("boundary", "")
    return patch


def _node_id(node: dict, fallback: str = "") -> str:
    return str(
        node.get("node_id")
        or node.get("mid_id")
        or node.get("leaf_id")
        or node.get("record_id")
        or fallback
    )


def _dynamic_frontier_groups(
    *,
    skill_content: str,
    frontier: list[dict],
    level: int,
    target_children: int,
    max_children: int,
) -> tuple[list[dict], dict]:
    """Plan compatible non-overlapping groups over one executable frontier."""
    planner_nodes: list[dict] = []
    node_by_id: dict[str, dict] = {}
    for idx, node in enumerate(frontier, start=1):
        node_id = _node_id(node, f"F{level}_{idx}")
        copied = dict(node)
        copied["leaf_id"] = node_id
        copied["node_id"] = node_id
        planner_nodes.append(copied)
        node_by_id[node_id] = node

    groups, report = _build_mid_groups(
        skill_content=skill_content,
        leaf_patches=planner_nodes,
        target_children=target_children,
        max_children=max_children,
    )
    clean_groups: list[dict] = []
    used: set[str] = set()
    split_count = 0
    for group in groups:
        child_ids = [
            str(child_id)
            for child_id in group.get("leaf_ids", [])
            if str(child_id) in node_by_id and str(child_id) not in used
        ]
        for start in range(0, len(child_ids), max_children):
            chunk = child_ids[start:start + max_children]
            if len(chunk) < 2:
                continue
            split_count += int(len(child_ids) > max_children)
            for child_id in chunk:
                used.add(child_id)
            children = [node_by_id[child_id] for child_id in chunk]
            support_ids = list(dict.fromkeys(
                support_id
                for child in children
                for support_id in _as_clean_str_list(child.get("support_sample_ids"))
            ))
            clean_groups.append({
                **group,
                "mid_id": f"N{level}_{len(clean_groups) + 1}",
                "leaf_ids": chunk,
                "child_ids": chunk,
                "support_count": len(support_ids) or sum(
                    int(child.get("support_count", 0) or 0) for child in children
                ),
                "support_sample_ids": support_ids,
                "target_children": target_children,
                "max_children": max_children,
            })
    return clean_groups, {
        **report,
        "level": level,
        "target_children": target_children,
        "max_children": max_children,
        "n_frontier_in": len(frontier),
        "n_merge_groups": len(clean_groups),
        "n_grouped_nodes": len(used),
        "n_carried_nodes": len(frontier) - len(used),
        "n_oversized_groups_split": split_count,
    }


def _build_dynamic_hierarchy(
    *,
    skill_content: str,
    leaf_patches: list[dict],
    max_tree_depth: int,
    target_children: int,
    max_children: int,
    top_mode: str,
    allow_open_types: bool,
    merge_workers: int,
    verbose: bool,
) -> tuple[dict, dict]:
    """Build an optimizer-planned hierarchy until no valid parent is proposed."""
    node_by_id: dict[str, dict] = {}
    frontier: list[dict] = []
    levels: list[list[str]] = []
    for idx, leaf in enumerate(leaf_patches, start=1):
        node_id = _node_id(leaf, f"L{idx}")
        leaf["node_id"] = node_id
        leaf["node_level"] = "leaf"
        leaf["tree_level"] = 0
        leaf["child_ids"] = []
        leaf["leaf_ids"] = list(dict.fromkeys([
            *_as_clean_str_list(leaf.get("leaf_ids")),
            str(leaf.get("leaf_id") or node_id),
        ]))
        leaf["leaf_coverage"] = len(leaf["leaf_ids"])
        node_by_id[node_id] = leaf
        frontier.append(leaf)
    levels.append([_node_id(node) for node in frontier])

    level_reports: list[dict] = []
    internal_nodes: list[dict] = []
    max_merge_levels = max(max(int(max_tree_depth), 2) - 1, 1)
    for level in range(1, max_merge_levels + 1):
        if len(frontier) <= 1:
            break
        groups, plan = _dynamic_frontier_groups(
            skill_content=skill_content,
            frontier=frontier,
            level=level,
            target_children=target_children,
            max_children=max_children,
        )
        level_reports.append(plan)
        if not groups:
            break
        frontier_by_id = {_node_id(node): node for node in frontier}

        def build_parent(group: dict) -> dict | None:
            child_ids = _as_clean_str_list(group.get("child_ids"))
            planner_children: dict[str, dict] = {}
            for child_id in child_ids:
                child = dict(frontier_by_id[child_id])
                child["leaf_id"] = child_id
                planner_children[child_id] = child
            parent = _build_mid_patch(
                skill_content=skill_content,
                group=group,
                leaf_patches_by_id=planner_children,
                allow_open_types=allow_open_types,
            )
            if not isinstance(parent.get("shared_core"), dict):
                return None
            node_id = str(group["mid_id"])
            actual_children = [frontier_by_id[child_id] for child_id in child_ids]
            leaf_ids = list(dict.fromkeys(
                leaf_id
                for child in actual_children
                for leaf_id in _as_clean_str_list(child.get("leaf_ids"))
            ))
            parent.update({
                "node_id": node_id,
                "node_level": "internal",
                "tree_level": level,
                "child_ids": child_ids,
                "leaf_ids": leaf_ids,
                "leaf_coverage": len(leaf_ids),
            })
            for edit in _patch_edits(parent):
                edit["leaf_ids"] = leaf_ids
                edit["node_ids"] = [node_id]
            return parent

        workers = max(1, min(int(merge_workers or 1), len(groups)))
        if workers == 1:
            built_parents = [build_parent(group) for group in groups]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                built_parents = list(executor.map(build_parent, groups))
        parents = [parent for parent in built_parents if isinstance(parent, dict)]
        if not parents:
            level_reports[-1]["status"] = "no_executable_abstraction"
            break
        grouped_ids = {
            child_id
            for parent in parents
            for child_id in _as_clean_str_list(parent.get("child_ids"))
        }
        carried = [node for node in frontier if _node_id(node) not in grouped_ids]
        frontier = [*parents, *carried]
        for parent in parents:
            node_by_id[_node_id(parent)] = parent
        internal_nodes.extend(parents)
        levels.append([_node_id(node) for node in frontier])
        if verbose:
            print(
                f"    [PatchTree dynamic] level={level} "
                f"parents={len(parents)} carried={len(carried)} frontier={len(frontier)}"
            )

    top_nodes = frontier
    requested_top_mode = str(top_mode or "auto").strip().lower()
    if requested_top_mode not in {
        "auto", "real_root", "virtual_root", "conservative_root",
    }:
        requested_top_mode = "auto"
    resolved_top_mode = "real_root" if len(top_nodes) == 1 else requested_top_mode
    root_patch: dict
    real_root_attempt: dict | None = None
    conservative_root_attempt: dict | None = None
    if len(top_nodes) == 1:
        root_patch = top_nodes[0]
    elif requested_top_mode == "conservative_root":
        conservative_root_attempt = _build_conservative_root_patch(
            skill_content=skill_content,
            child_patches=top_nodes,
            allow_open_types=allow_open_types,
        )
        resolved_top_mode = "conservative_root"
        root_patch = conservative_root_attempt
        root_patch.update({
            "node_id": "ROOT",
            "node_level": "root",
            "tree_level": len(levels),
            "child_ids": [_node_id(node) for node in top_nodes],
            "leaf_ids": list(dict.fromkeys(
                leaf_id
                for node in top_nodes
                for leaf_id in _as_clean_str_list(node.get("leaf_ids"))
            )),
        })
        root_patch["leaf_coverage"] = len(root_patch["leaf_ids"])
        node_by_id["ROOT"] = root_patch
    elif (
        requested_top_mode in {"auto", "real_root"}
        and len(top_nodes) <= max_children
        and len(levels) < max_tree_depth
    ):
        real_root_attempt = _build_root_patch(
            skill_content=skill_content,
            child_patches=top_nodes,
            allow_open_types=allow_open_types,
            child_level="frontier",
        )
        top_ids = {_node_id(node) for node in top_nodes}
        covered_ids: set[str] = set()
        shared_core = real_root_attempt.get("shared_core")
        if isinstance(shared_core, dict):
            covered_ids.update(_as_clean_str_list(shared_core.get("source_child_ids")))
        for residual in real_root_attempt.get("conditional_residuals", []):
            if isinstance(residual, dict):
                covered_ids.update(_as_clean_str_list(residual.get("source_child_ids")))
        has_genuine_core = (
            isinstance(shared_core, dict)
            and top_ids.issubset(covered_ids)
        )
        if requested_top_mode == "real_root" or has_genuine_core:
            resolved_top_mode = "real_root"
            root_patch = real_root_attempt
            root_patch.update({
                "node_id": "ROOT",
                "node_level": "root",
                "tree_level": len(levels),
                "child_ids": [_node_id(node) for node in top_nodes],
                "leaf_ids": list(dict.fromkeys(
                    leaf_id
                    for node in top_nodes
                    for leaf_id in _as_clean_str_list(node.get("leaf_ids"))
                )),
            })
            root_patch["leaf_coverage"] = len(root_patch["leaf_ids"])
            node_by_id["ROOT"] = root_patch
        else:
            resolved_top_mode = "virtual_root"
            root_patch = _fallback_parent_patch(
                top_nodes,
                allow_open_types=allow_open_types,
                reasoning="virtual-root frontier carrier; not an executable abstraction",
            )
    else:
        resolved_top_mode = "virtual_root"
        root_patch = _fallback_parent_patch(
            top_nodes,
            allow_open_types=allow_open_types,
            reasoning="virtual-root frontier carrier; not an executable abstraction",
        )

    virtual_root = resolved_top_mode == "virtual_root"
    fallback_top_ids = (
        _as_clean_str_list(root_patch.get("child_ids"))
        if len(top_nodes) == 1 and not virtual_root
        else [_node_id(node) for node in top_nodes]
    )
    return root_patch, {
        "builder": "recursive",
        "top_mode_requested": requested_top_mode,
        "top_mode": resolved_top_mode,
        "virtual_root": virtual_root,
        "top_node_ids": [_node_id(node) for node in top_nodes],
        "top_patches": top_nodes,
        "fallback_top_node_ids": fallback_top_ids,
        "fallback_top_patches": [
            node_by_id[node_id]
            for node_id in fallback_top_ids
            if node_id in node_by_id
        ],
        "node_by_id": node_by_id,
        "levels": levels,
        "level_reports": level_reports,
        "internal_nodes": internal_nodes,
        "real_root_attempt": real_root_attempt,
        "conservative_root_attempt": conservative_root_attempt,
        "max_tree_depth": max_tree_depth,
        "actual_tree_depth": len(levels) + int("ROOT" in node_by_id),
        "target_children": target_children,
        "max_children": max_children,
    }


def _empty_patch(update_mode: str, reasoning: str) -> dict:
    return {"reasoning": reasoning, payload_key(update_mode): []}


def build_patchtree(
    skill_content: str,
    patches: list[dict],
    *,
    min_support: int = 2,
    max_leaf_groups: int = 8,
    allow_open_types: bool = False,
    group_by_cluster: bool = False,
    low_support_fallback: bool = True,
    tree_depth: int = 2,
    tree_builder: str = "fixed",
    max_tree_depth: int = 4,
    merge_target_children: int = 3,
    merge_max_children: int = 4,
    merge_strategy: str = "hierarchical",
    grouping_mode: str = "type",
    grouping_seed: int = 0,
    top_mode: str = "auto",
    leaf_merge_workers: int = 1,
    mid_merge_workers: int = 1,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Build a direct/leaf/mid PatchTree and compile the root candidate.

    Depth 1 is the true flat ablation: typed PatchRecord edits are passed
    directly to the Root merger without constructing Leaf or Mid nodes.
    """
    update_mode = "patch"
    patches = list(patches)
    tree_depth = max(int(tree_depth or 1), 1)
    merge_strategy = str(merge_strategy or "hierarchical").strip().lower()
    if merge_strategy not in {"hierarchical", "concat", "flat_fuse"}:
        merge_strategy = "hierarchical"
    grouping_mode = str(grouping_mode or "type").strip().lower()
    if grouping_mode not in {"type", "random", "success_then_type"}:
        grouping_mode = "type"

    raw_edits = _extract_edits(
        patches, update_mode, allow_open_types=allow_open_types,
    )
    if tree_depth == 1:
        record_patches: list[dict] = []
        for idx, edit in enumerate(raw_edits, start=1):
            record_ids = _as_clean_str_list(edit.get("record_ids"))
            record_id = record_ids[0] if record_ids else f"R{idx:04d}"
            copied = _copy_edit(edit, allow_open_types=allow_open_types)
            copied["record_ids"] = list(dict.fromkeys([
                *_as_clean_str_list(copied.get("record_ids")), record_id,
            ]))
            record_patches.append({
                "record_id": record_id,
                "reasoning": str(edit.get("repair_signature") or "direct PatchRecord repair"),
                "question_type": copied.get("question_type", "other"),
                "revision_type": copied.get("revision_type", "other"),
                "support_count": _support_count(copied),
                "support_sample_ids": _sample_ids(copied),
                "shared_core": None,
                "conditional_residuals": [],
                "preserved_constraints": {},
                "unresolved_conflicts": [],
                "edits": [copied],
            })
        root_patch = (
            _build_root_patch(
                skill_content=skill_content,
                child_patches=record_patches,
                allow_open_types=allow_open_types,
                child_level="record",
            )
            if record_patches
            else _empty_patch(update_mode, "no typed PatchRecords")
        )
        if verbose:
            print(
                f"    [type-guided aggregate] depth=1 direct_records={len(record_patches)}"
            )
        artifact = {
            "enabled": True,
            "raw_edit_count": len(raw_edits),
            "groups": [],
            "kept_groups": [],
            "dropped_groups": [],
            "leaf_patches": [],
            "mid_plan": {
                "enabled": False,
                "status": "depth_1_direct",
                "groups": [],
                "errors": [],
            },
            "mid_groups": [],
            "mid_patches": [],
            "tree_depth": 1,
            "root_children_level": "record",
            "root_child_patches": record_patches,
            "root_patch": root_patch,
            "settings": {
                "min_support": min_support,
                "max_leaf_groups": max_leaf_groups,
                "allow_open_types": allow_open_types,
                "group_by_cluster": False,
                "grouping_mode": grouping_mode,
                "grouping_seed": int(grouping_seed),
                "merge_strategy": merge_strategy,
                "low_support_fallback": False,
                "tree_depth": 1,
                "leaf_merge_workers": 0,
                "mid_merge_workers": 0,
            },
        }
        return root_patch, artifact

    groups = _group_edits(
        raw_edits,
        allow_open_types=allow_open_types,
        group_by_cluster=group_by_cluster,
        grouping_mode=grouping_mode,
        grouping_seed=grouping_seed,
    )
    for group in groups:
        group["allow_open_types"] = bool(allow_open_types)
    kept_groups, dropped_groups = _filter_groups(
        groups,
        min_support=max(int(min_support), 1),
        max_leaf_groups=max(int(max_leaf_groups), 0),
        low_support_fallback=low_support_fallback,
    )

    if verbose:
        print(
            f"    [type-guided aggregate] edits={len(raw_edits)} "
            f"groups={len(groups)} kept={len(kept_groups)} dropped={len(dropped_groups)}"
        )

    if not kept_groups:
        artifact = {
            "enabled": True,
            "raw_edit_count": len(raw_edits),
            "groups": groups,
            "kept_groups": [],
            "dropped_groups": dropped_groups,
            "leaf_patches": [],
            "mid_plan": {"enabled": False, "status": "no_leaf_groups", "groups": [], "errors": []},
            "mid_groups": [],
            "mid_patches": [],
            "tree_depth": tree_depth,
            "root_children_level": "leaf",
            "root_child_patches": [],
            "root_patch": _empty_patch(update_mode, "no typed leaf groups"),
        }
        return artifact["root_patch"], artifact

    def build_leaf(group: dict) -> dict:
        patch = _build_leaf_patch(
            skill_content=skill_content,
            group=group,
        )
        patch["leaf_id"] = group["leaf_id"]
        patch["question_type"] = group["question_type"]
        patch["revision_type"] = group["revision_type"]
        patch["support_count"] = group["support_count"]
        patch["support_sample_ids"] = group.get("support_sample_ids", [])
        if group.get("cluster_id"):
            patch["cluster_id"] = group.get("cluster_id")
            patch["cluster_label"] = group.get("cluster_label", "")
            patch["cluster_question_type"] = group.get("cluster_question_type", "")
            patch["cluster_revision_type"] = group.get("cluster_revision_type", "")
            patch["cluster_source"] = group.get("cluster_source", "")
        for key in (
            "member_question_type_counts",
            "member_revision_type_counts",
            "member_cluster_question_type_counts",
            "member_cluster_revision_type_counts",
        ):
            patch[key] = group.get(key, {})
        if group.get("repair_signature"):
            patch["repair_signature"] = group.get("repair_signature")
        return patch

    leaf_workers = max(1, min(int(leaf_merge_workers or 1), len(kept_groups)))
    if verbose and leaf_workers > 1:
        print(f"    [PatchTree aggregate] leaf merges parallel workers={leaf_workers}")
    if leaf_workers == 1:
        leaf_patches = [build_leaf(group) for group in kept_groups]
    else:
        with ThreadPoolExecutor(max_workers=leaf_workers) as executor:
            leaf_patches = list(executor.map(build_leaf, kept_groups))

    mid_groups: list[dict] = []
    mid_patches: list[dict] = []
    mid_plan: dict = {
        "enabled": False,
        "status": "disabled",
        "groups": [],
        "errors": [],
    }
    root_children_level = "leaf"
    root_child_patches = leaf_patches
    if (
        merge_strategy == "hierarchical"
        and
        str(tree_builder or "fixed").strip().lower() != "recursive"
        and tree_depth >= 3
        and len(leaf_patches) > 1
    ):
        mid_groups, mid_plan = _build_mid_groups(
            skill_content=skill_content,
            leaf_patches=leaf_patches,
        )
        leaf_patches_by_id = {
            str(leaf.get("leaf_id") or ""): leaf
            for leaf in leaf_patches
            if str(leaf.get("leaf_id") or "").strip()
        }
        def build_mid(group: dict) -> dict:
            return _build_mid_patch(
                skill_content=skill_content,
                group=group,
                leaf_patches_by_id=leaf_patches_by_id,
                allow_open_types=allow_open_types,
            )

        mid_workers = max(1, min(int(mid_merge_workers or 1), len(mid_groups)))
        if verbose and mid_workers > 1:
            print(f"    [PatchTree aggregate] mid merges parallel workers={mid_workers}")
        if mid_workers == 1:
            mid_patches = [build_mid(group) for group in mid_groups]
        else:
            with ThreadPoolExecutor(max_workers=mid_workers) as executor:
                mid_patches = list(executor.map(build_mid, mid_groups))
        if mid_patches:
            root_children_level = "mid"
            root_child_patches = mid_patches

    hierarchy: dict = {"builder": "fixed", "virtual_root": False}
    if merge_strategy == "concat":
        root_children_level = "leaf"
        root_child_patches = leaf_patches
        root_patch = _fallback_parent_patch(
            leaf_patches,
            allow_open_types=allow_open_types,
            reasoning="concat ablation: preserve all compiled leaf edits without root fusion",
        )
        hierarchy = {
            "builder": "concat",
            "top_mode": "concat",
            "virtual_root": False,
        }
    elif merge_strategy == "flat_fuse":
        root_children_level = "leaf"
        root_child_patches = leaf_patches
        root_patch = _build_root_patch(
            skill_content=skill_content,
            child_patches=leaf_patches,
            allow_open_types=allow_open_types,
            child_level="leaf",
        )
        hierarchy = {
            "builder": "flat_fuse",
            "top_mode": "real_root",
            "virtual_root": False,
        }
    elif str(tree_builder or "fixed").strip().lower() == "recursive":
        root_patch, hierarchy = _build_dynamic_hierarchy(
            skill_content=skill_content,
            leaf_patches=leaf_patches,
            max_tree_depth=max(int(max_tree_depth or 4), 2),
            target_children=max(int(merge_target_children or 3), 2),
            max_children=max(
                int(merge_max_children or 4),
                max(int(merge_target_children or 3), 2),
            ),
            top_mode=top_mode,
            allow_open_types=allow_open_types,
            merge_workers=mid_merge_workers,
            verbose=verbose,
        )
        root_children_level = "frontier"
        root_child_patches = hierarchy["fallback_top_patches"]
        mid_patches = hierarchy["internal_nodes"]
        tree_depth = hierarchy["actual_tree_depth"]
    else:
        root_patch = _build_root_patch(
            skill_content=skill_content,
            child_patches=root_child_patches,
            allow_open_types=allow_open_types,
            child_level=root_children_level,
        )

    artifact = {
        "enabled": True,
        "raw_edit_count": len(raw_edits),
        "groups": groups,
        "kept_groups": kept_groups,
        "dropped_groups": dropped_groups,
        "leaf_patches": leaf_patches,
        "mid_plan": mid_plan,
        "mid_groups": mid_groups,
        "mid_patches": mid_patches,
        "tree_depth": tree_depth,
        "root_children_level": root_children_level,
        "root_child_patches": root_child_patches,
        "root_patch": root_patch,
        "hierarchy": hierarchy,
        "settings": {
            "min_support": min_support,
            "max_leaf_groups": max_leaf_groups,
            "allow_open_types": allow_open_types,
            "group_by_cluster": group_by_cluster,
            "grouping_mode": grouping_mode,
            "grouping_seed": int(grouping_seed),
            "merge_strategy": merge_strategy,
            "low_support_fallback": low_support_fallback,
            "tree_depth": tree_depth,
            "tree_builder": hierarchy.get("builder", "fixed"),
            "max_tree_depth": max_tree_depth,
            "merge_target_children": merge_target_children,
            "merge_max_children": merge_max_children,
            "top_mode": hierarchy.get("top_mode", "real_root"),
            "leaf_merge_workers": leaf_workers,
            "mid_merge_workers": (
                max(1, min(int(mid_merge_workers or 1), len(mid_groups)))
                if mid_groups else 0
            ),
        },
    }
    return root_patch, artifact
