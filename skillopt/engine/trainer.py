"""PatchTree trainer — the main training loop.

Orchestrates the 6-stage PatchTree pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

The trainer is environment-agnostic; all environment-specific logic is
delegated to an :class:`~skillopt.envs.base.EnvAdapter` instance.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.evaluation.gate import GateResult, evaluate_gate, select_gate_score
from skillopt.gradient.type_guided_merge_v2 import (
    generate_patch_records,
    merge_type_guided_v2_records,
)
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.scheduler import build_scheduler
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.update_modes import (
    get_payload_items,
    payload_label,
    short_item_summary,
)
from skillopt.model import (
    chat_optimizer,
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    get_token_summary,
    reset_token_tracker,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json
from skillopt.utils import compute_score, skill_hash


_TG_ORIGINAL_ID_FIELD = "_type_guided_original_id"
_TG_REPEAT_ID_FIELD = "_type_guided_repeat_id"
_TG_PREDICTION_ID_FIELD = "_prediction_id"


def _can_flatten_type_guided_repeats(train_env: object) -> bool:
    """Return whether repeated rollouts can share one list-backed batch."""
    return isinstance(train_env, list) and all(isinstance(item, dict) for item in train_env)


def _flatten_type_guided_repeat_env(
    train_env: list[dict],
    *,
    repeats: int,
) -> tuple[list[dict], dict[str, tuple[str, int]]]:
    flat_env: list[dict] = []
    id_map: dict[str, tuple[str, int]] = {}
    for repeat_id in range(max(int(repeats or 1), 1)):
        for item in train_env:
            original_id = str(item.get("id"))
            cloned = dict(item)
            if repeat_id == 0:
                prediction_id = original_id
            else:
                prediction_id = f"{original_id}::tg_repeat{repeat_id}"
                cloned["id"] = prediction_id
            cloned[_TG_ORIGINAL_ID_FIELD] = original_id
            cloned[_TG_REPEAT_ID_FIELD] = repeat_id
            id_map[prediction_id] = (original_id, repeat_id)
            flat_env.append(cloned)
    return flat_env, id_map


def _split_flattened_type_guided_results(
    flat_results: list[dict],
    *,
    repeats: int,
    id_map: dict[str, tuple[str, int]],
    prediction_dir: str,
) -> list[dict]:
    repeated_rollouts = [
        {"repeat_id": repeat_id, "results": [], "prediction_dir": prediction_dir}
        for repeat_id in range(max(int(repeats or 1), 1))
    ]
    for result in flat_results:
        if not isinstance(result, dict) or result.get("id") is None:
            continue
        prediction_id = str(result.get("id"))
        original_id, repeat_id = id_map.get(
            prediction_id,
            (
                str(result.get(_TG_ORIGINAL_ID_FIELD) or result.get("source_id") or prediction_id),
                int(result.get(_TG_REPEAT_ID_FIELD, 0) or 0),
            ),
        )
        if repeat_id < 0 or repeat_id >= len(repeated_rollouts):
            repeat_id = 0
        restored = dict(result)
        restored[_TG_PREDICTION_ID_FIELD] = prediction_id
        restored["source_id"] = original_id
        restored["original_id"] = original_id
        restored["id"] = original_id
        restored[_TG_REPEAT_ID_FIELD] = repeat_id
        repeated_rollouts[repeat_id]["results"].append(restored)
    return repeated_rollouts


def _cfg_int(cfg: dict, key: str, default: int) -> int:
    value = cfg.get(key, default)
    if value is None or value == "":
        value = default
    return int(value)


def _type_guided_fallback_sample_seed(seed: int) -> int:
    """Return the run-stable seed for the fallback representative subset."""
    return int(seed) + 97


def _recursive_fallback_children(
    node: dict,
    *,
    node_by_id: dict[str, dict],
    min_leaf_coverage: int,
) -> list[dict]:
    """Return direct children only when a rejected node is still refinable."""
    leaf_coverage = int(
        node.get("leaf_coverage")
        or len(node.get("leaf_ids") or [])
        or 1
    )
    if leaf_coverage <= max(int(min_leaf_coverage or 1), 1):
        return []
    child_ids = [
        str(node_id)
        for node_id in node.get("child_ids", [])
        if str(node_id)
    ]
    return [
        node_by_id[node_id]
        for node_id in child_ids
        if node_id in node_by_id
    ]


def _append_jsonl(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except json.JSONDecodeError:
                    continue
    return rows


def _extract_type_guided_tail_records(
    *,
    dropped_groups: list[dict],
    epoch: int,
    step: int,
    current_skill: str,
) -> list[dict]:
    """Convert low-support dropped V2 groups into epoch tail-bank records."""
    rows: list[dict] = []
    skill_id = skill_hash(current_skill)
    for group in dropped_groups or []:
        if not isinstance(group, dict):
            continue
        reason = str(group.get("drop_reason") or "")
        if not (reason.startswith("support<") or reason == "low_support_fallback"):
            continue
        for edit in group.get("edits") or []:
            if not isinstance(edit, dict):
                continue
            patch = {
                key: edit.get(key)
                for key in ("op", "target", "content")
                if key in edit
            }
            if not patch.get("op"):
                continue
            record_id = str(edit.get("record_ids", [""])[0] if isinstance(edit.get("record_ids"), list) and edit.get("record_ids") else edit.get("record_id") or "")
            rows.append({
                "tail_id": f"E{epoch:02d}_S{step:04d}_{len(rows)+1:04d}",
                "epoch": epoch,
                "step": step,
                "record_id": record_id,
                "question_type": edit.get("question_type", ""),
                "revision_type": edit.get("revision_type", ""),
                "repair_signature": edit.get("repair_signature", ""),
                "condition": edit.get("condition") or edit.get("applicability", ""),
                "boundary": edit.get("boundary", ""),
                "patch": patch,
                "drop_reason": reason,
                "source_skill_hash": skill_id,
            })
    return rows


def _tail_records_to_patch_records(rows: list[dict], *, max_records: int) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for row in sorted(
        rows,
        key=lambda r: (
            str(r.get("question_type") or ""),
            str(r.get("revision_type") or ""),
            str(r.get("repair_signature") or ""),
            int(r.get("step") or 0),
            str(r.get("tail_id") or ""),
        ),
    ):
        key = str(row.get("tail_id") or "")
        if key in seen:
            continue
        seen.add(key)
        record_id = str(row.get("tail_id") or f"T{len(records)+1:04d}")
        records.append({
            "record_id": record_id,
            "question_type": row.get("question_type", ""),
            "revision_type": row.get("revision_type", ""),
            "repair_signature": row.get("repair_signature", ""),
            "condition": row.get("condition", ""),
            "boundary": row.get("boundary", ""),
            "patch": row.get("patch") or {},
            "tail_source": {
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "drop_reason": row.get("drop_reason"),
                "source_skill_hash": row.get("source_skill_hash"),
            },
        })
        if max_records > 0 and len(records) >= max_records:
            break
    return records


def _edit_signature(edit: dict) -> tuple[str, str, str]:
    return (
        str(edit.get("op") or "").strip().lower(),
        re.sub(r"\s+", " ", str(edit.get("target") or "").strip()),
        re.sub(r"\s+", " ", str(edit.get("content") or "").strip()),
    )


def _reconcile_fallback_edits(
    *,
    child_patches: list[dict],
    child_rows: list[dict],
    update_mode: str,
    mode: str,
    min_children: int,
) -> tuple[list[dict], dict]:
    """Drop duplicate/conflicting fallback edits without rewriting content."""
    edit_cards: list[dict] = []
    seen: dict[tuple[str, str, str], str] = {}
    keep_ids: list[str] = []
    dropped: list[dict] = []
    child_by_id = {
        str(row.get("child_id") or ""): row
        for row in child_rows
        if isinstance(row, dict)
    }
    for child_idx, child in enumerate(child_patches, start=1):
        child_id = str(child.get("mid_id") or child.get("leaf_id") or f"C{child_idx}")
        child_score = child_by_id.get(child_id, {}).get("gate_score")
        for edit_idx, edit in enumerate(get_payload_items(child, update_mode), start=1):
            if not isinstance(edit, dict):
                continue
            edit_id = f"{child_id}_E{edit_idx}"
            card = {
                "edit_id": edit_id,
                "child_id": child_id,
                "child_score": child_score,
                "op": edit.get("op"),
                "target": edit.get("target", ""),
                "content": edit.get("content", ""),
                "question_type": edit.get("question_type", ""),
                "revision_type": edit.get("revision_type", ""),
                "support_count": edit.get("support_count", 0),
                "edit": edit,
            }
            sig = _edit_signature(edit)
            duplicate_of = seen.get(sig)
            if duplicate_of:
                dropped.append({
                    "edit_id": edit_id,
                    "reason": f"exact_duplicate_of:{duplicate_of}",
                    "source": "deterministic",
                })
                continue
            seen[sig] = edit_id
            keep_ids.append(edit_id)
            edit_cards.append(card)

    report = {
        "mode": mode,
        "min_children": min_children,
        "n_children": len(child_patches),
        "n_input_edits": len(keep_ids) + len(dropped),
        "n_after_dedup": len(keep_ids),
        "dropped_edits": dropped,
        "llm_used": False,
        "status": "deduplicated",
    }
    if mode != "llm_select" or len(child_patches) < min_children or len(edit_cards) <= 1:
        by_id = {card["edit_id"]: card["edit"] for card in edit_cards}
        report["n_output_edits"] = len(keep_ids)
        return [by_id[edit_id] for edit_id in keep_ids if edit_id in by_id], report

    payload = {
        "instruction": (
            "Select original fallback edits to keep. Do not rewrite content. "
            "Drop only duplicates, near-duplicates, or direct conflicts."
        ),
        "edits": [
            {key: value for key, value in card.items() if key != "edit"}
            for card in edit_cards
        ],
    }
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_fallback_reconcile"),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            max_completion_tokens=8192,
            retries=2,
            stage="type_guided_fallback_reconcile",
        )
        parsed = extract_json(response)
        raw_keep = parsed.get("keep_edit_ids") if isinstance(parsed, dict) else None
        raw_drop = parsed.get("drop_edit_ids") if isinstance(parsed, dict) else None
        if not isinstance(raw_keep, list):
            raise ValueError("missing keep_edit_ids")
        allowed = {card["edit_id"] for card in edit_cards}
        llm_keep = [str(edit_id) for edit_id in raw_keep if str(edit_id) in allowed]
        if not llm_keep:
            raise ValueError("empty keep_edit_ids")
        keep_set = set(llm_keep)
        for edit_id in sorted(allowed - keep_set):
            dropped.append({
                "edit_id": edit_id,
                "reason": "not_selected_by_llm",
                "source": "llm_select",
            })
        if isinstance(raw_drop, list):
            report["llm_drop_edits"] = raw_drop
        report["llm_used"] = True
        report["status"] = "llm_selected"
        keep_ids = llm_keep
    except Exception as exc:  # noqa: BLE001
        report["status"] = "llm_failed_dedup_only"
        report["error"] = repr(exc)

    report["dropped_edits"] = dropped
    report["n_output_edits"] = len(keep_ids)
    by_id = {card["edit_id"]: card["edit"] for card in edit_cards}
    return [by_id[edit_id] for edit_id in keep_ids if edit_id in by_id], report


def _fuse_validated_frontier_edits(
    *,
    skill_content: str,
    child_patches: list[dict],
    child_rows: list[dict],
    update_mode: str,
) -> tuple[list[dict], dict]:
    """Semantically integrate validated children without re-abstracting them."""
    child_by_id = {
        str(row.get("child_id") or ""): row
        for row in child_rows
        if isinstance(row, dict)
    }
    selected_children: list[dict] = []
    allowed_child_ids: set[str] = set()
    child_support_counts: dict[str, int] = {}
    child_support_ids: dict[str, list[str]] = {}
    n_input_edits = 0
    for child_idx, child in enumerate(child_patches, start=1):
        child_id = str(child.get("mid_id") or child.get("leaf_id") or f"C{child_idx}")
        row = child_by_id.get(child_id, {})
        allowed_child_ids.add(child_id)
        child_support_counts[child_id] = int(
            child.get("support_count", row.get("support_count", 0)) or 0
        )
        raw_support_ids = child.get("support_sample_ids") or []
        if isinstance(raw_support_ids, str):
            raw_support_ids = [raw_support_ids]
        child_support_ids[child_id] = (
            [str(item) for item in raw_support_ids if str(item).strip()]
            if isinstance(raw_support_ids, list)
            else []
        )
        edits = [
            dict(edit)
            for edit in get_payload_items(child, update_mode)
            if isinstance(edit, dict)
        ]
        n_input_edits += len(edits)
        selected_children.append({
            "child_id": child_id,
            "child_level": row.get("child_level", ""),
            "leaf_ids": child.get("leaf_ids", row.get("leaf_ids", [])),
            "question_type": child.get("question_type", row.get("question_type", "")),
            "revision_type": child.get("revision_type", row.get("revision_type", "")),
            "support_count": child.get("support_count", row.get("support_count", 0)),
            "validation": {
                "hard": row.get("hard"),
                "soft": row.get("soft"),
                "gate_score": row.get("gate_score"),
                "improvement": row.get("improvement"),
            },
            "boundary": child.get("boundary", ""),
            "edits": edits,
        })

    payload = {
        "current_skill": skill_content,
        "selected_children": selected_children,
    }
    report = {
        "mode": "llm_fuse",
        "status": "pending",
        "n_children": len(child_patches),
        "n_input_edits": n_input_edits,
        "n_output_edits": 0,
        "llm_used": False,
        "dropped_child_insights": [],
    }
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_validated_frontier_fuse"),
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            max_completion_tokens=16384,
            retries=3,
            stage="type_guided_validated_frontier_fuse",
        )
        parsed = extract_json(response)
        raw_edits = parsed.get("edits") if isinstance(parsed, dict) else None
        if not isinstance(raw_edits, list):
            raise ValueError("missing edits")

        fused_edits: list[dict] = []
        for raw in raw_edits:
            if not isinstance(raw, dict):
                continue
            op = str(raw.get("op") or "").strip()
            target = str(raw.get("target") or "").strip()
            content = str(raw.get("content") or "").strip()
            condition = str(raw.get("condition") or "").strip()
            boundary = str(raw.get("boundary") or "").strip()
            source_ids = raw.get("source_child_ids")
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            if not isinstance(source_ids, list):
                source_ids = []
            source_ids = list(dict.fromkeys(
                str(item).strip()
                for item in source_ids
                if str(item).strip() in allowed_child_ids
            ))
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
            support_ids = list(dict.fromkeys(
                support_id
                for child_id in source_ids
                for support_id in child_support_ids.get(child_id, [])
            ))
            support_count = (
                len(support_ids)
                if support_ids
                else sum(child_support_counts.get(child_id, 0) for child_id in source_ids)
            )
            edit = {
                "op": op,
                "content": content,
                "condition": condition,
                "boundary": boundary,
                "source_child_ids": source_ids,
                "source_type": "failure",
                "support_count": max(support_count, 1),
            }
            if support_ids:
                edit["support_sample_ids"] = support_ids
            if target:
                edit["target"] = target
            if op == "delete":
                edit.pop("content", None)
            fused_edits.append(edit)
        if not fused_edits:
            raise ValueError("no valid fused edits")

        report.update({
            "status": "llm_fused",
            "llm_used": True,
            "n_output_edits": len(fused_edits),
            "reasoning": str(parsed.get("reasoning") or ""),
            "dropped_child_insights": (
                parsed.get("dropped_child_insights")
                if isinstance(parsed.get("dropped_child_insights"), list)
                else []
            ),
        })
        return fused_edits, report
    except Exception as exc:  # noqa: BLE001
        fallback_edits, fallback_report = _reconcile_fallback_edits(
            child_patches=child_patches,
            child_rows=child_rows,
            update_mode=update_mode,
            mode="deterministic",
            min_children=1,
        )
        report.update({
            "status": "llm_fuse_failed_dedup_only",
            "error": repr(exc),
            "n_output_edits": len(fallback_edits),
            "fallback_report": fallback_report,
        })
        return fallback_edits, report


def _summarise_apply_report(report: list[dict]) -> dict:
    return {
        "total": len(report),
        "applied": sum(
            1 for row in report
            if str(row.get("status", "")).startswith("applied")
        ),
        "skipped": sum(
            1 for row in report
            if str(row.get("status", "")).startswith("skipped")
        ),
        "errors": sum(1 for row in report if row.get("status") == "error"),
    }


def _normalise_lr_control_mode(mode: str | None) -> str:
    raw = str(mode or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "manual": "fixed",
        "scheduler": "fixed",
        "scheduled": "fixed",
        "autonomous": "autonomous",
        "auto": "autonomous",
        "optimizer": "autonomous",
        "none": "none",
        "off": "none",
        "no_lr": "none",
    }
    if raw not in aliases:
        raise ValueError("optimizer.lr_control_mode must be one of fixed, autonomous, none")
    return aliases[raw]


# ── History / persistence helpers ─────────────────────────────────────────────

_SECRET_KEYS = {
    "azure_api_key",
    "api_key",
    "openai_api_key",
}


def _redact_value(val: str) -> str:
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}...{val[-4:]}"


def _redact_cfg(cfg: dict) -> dict:
    redacted = dict(cfg)
    for key in list(redacted):
        if key.lower() in _SECRET_KEYS and redacted.get(key):
            redacted[key] = _redact_value(str(redacted[key]))
    return redacted

def _load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_history(out_root: str, history: list[dict]) -> None:
    path = os.path.join(out_root, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _save_skill(out_root: str, step: int, content: str) -> None:
    skills_dir = os.path.join(out_root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    with open(os.path.join(skills_dir, f"skill_v{step:04d}.md"), "w") as f:
        f.write(content)


def _load_skill(out_root: str, step: int) -> str:
    path = os.path.join(out_root, "skills", f"skill_v{step:04d}.md")
    with open(path) as f:
        return f.read()


def _load_runtime_state(out_root: str) -> dict | None:
    path = os.path.join(out_root, "runtime_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def _save_runtime_state(out_root: str, state: dict) -> None:
    path = os.path.join(out_root, "runtime_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _row_id(row: object) -> str | None:
    """Return the stable dataset identifier carried by an env/result row."""
    if isinstance(row, dict):
        value = row.get("source_id")
        if value is None:
            value = row.get("id")
    else:
        value = getattr(row, "source_id", None)
        if value is None:
            value = getattr(row, "id", None)
    return None if value is None else str(value)


def _compact_selection_results(results: list) -> list[dict]:
    """Keep only fields needed to reconstruct a score on a val subset."""
    compact: list[dict] = []
    for row in results or []:
        row_id = _row_id(row)
        if row_id is None:
            continue
        if isinstance(row, dict):
            hard = row.get("hard", 0.0)
            soft = row.get("soft", 0.0)
        else:
            hard = getattr(row, "hard", 0.0)
            soft = getattr(row, "soft", 0.0)
        compact.append({
            "id": row_id,
            "hard": float(hard or 0.0),
            "soft": float(soft or 0.0),
        })
    return compact


def _selection_result_cache_path(out_root: str, skill_hash_value: str) -> str:
    return os.path.join(
        out_root, "selection_result_cache", f"{skill_hash_value}.json",
    )


def _save_selection_result_cache(
    out_root: str,
    skill_hash_value: str,
    results: list,
) -> list[dict]:
    """Persist compact per-item full-val results for later subset slicing."""
    compact = _compact_selection_results(results)
    if not compact:
        return []
    cache_dir = os.path.join(out_root, "selection_result_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = _selection_result_cache_path(out_root, skill_hash_value)
    with open(path, "w") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    return compact


def _load_selection_result_cache(
    out_root: str,
    skill_hash_value: str,
) -> list[dict]:
    path = _selection_result_cache_path(out_root, skill_hash_value)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            rows = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return _compact_selection_results(rows if isinstance(rows, list) else [])


def _slice_selection_results(
    cached_results: list[dict],
    item_ids: list[str],
) -> tuple[list[dict], list[str]]:
    """Select cached rows in requested order and report absent identifiers."""
    by_id = {
        str(row["id"]): row
        for row in cached_results
        if isinstance(row, dict) and row.get("id") is not None
    }
    selected: list[dict] = []
    missing: list[str] = []
    for item_id in item_ids:
        key = str(item_id)
        if key in by_id:
            selected.append(by_id[key])
        else:
            missing.append(key)
    return selected, missing


def _environment_item_ids(env_manager: object) -> list[str]:
    if not isinstance(env_manager, (list, tuple)):
        return []
    ids: list[str] = []
    for row in env_manager:
        row_id = _row_id(row)
        if row_id is None:
            return []
        ids.append(row_id)
    return ids


def _resolve_train_size(cfg: dict, dataloader) -> int:
    configured = int(cfg.get("train_size", 0) or 0)
    inferred: int | None = None

    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if value is not None:
                inferred = int(value)
        elif hasattr(dataloader, "train_items"):
            try:
                inferred = len(getattr(dataloader, "train_items"))
            except Exception:
                inferred = None

    if inferred is not None and inferred <= 0:
        inferred = None

    if configured > 0 and inferred is not None and configured != inferred:
        raise ValueError(
            f"Configured train_size={configured} does not match loaded train split "
            f"size={inferred}. Fix the config or the dataset split."
        )

    train_size = configured if configured > 0 else inferred
    if train_size is None or train_size <= 0:
        raise ValueError(
            "Unable to determine train_size automatically. "
            "Provide train.train_size in the config for this environment."
        )
    return int(train_size)


def _compute_task_type_buckets(results: list[dict], task_types: list[str]) -> dict[str, dict]:
    """Compute per-task-type success rates."""
    buckets: dict[str, dict] = {}
    for task in task_types + ["overall"]:
        buckets[task] = {"total": 0, "hard": 0, "soft": 0.0}
    for r in results:
        tt = r.get("task_type", "other")
        for key in [tt, "overall"]:
            if key not in buckets:
                buckets[key] = {"total": 0, "hard": 0, "soft": 0.0}
            buckets[key]["total"] += 1
            buckets[key]["hard"] += float(r.get("hard", 0))
            buckets[key]["soft"] += float(r.get("soft", 0.0))
    return buckets


def _format_rejection_buffer(buffer: list[dict]) -> str:
    """**DEPRECATED** — kept for backward compat; use _format_step_buffer."""
    return _format_step_buffer(buffer)


def _extract_failure_patterns(
    rollout_results: list[dict],
    step_dir: str,
) -> list[dict]:
    """Extract compact failure patterns from rollout results.

    Uses analyst ``failure_summary`` from minibatch patches when available,
    otherwise falls back to ``fail_reason`` prefix grouping.
    """
    failures = [r for r in rollout_results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9]
    if not failures:
        return []

    # Group by fail_reason prefix
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        reason = r.get("fail_reason", "unknown")
        prefix = reason.split(":")[0].strip() if ":" in reason else reason
        groups[prefix].append(r)

    # Try richer descriptions from analyst patches
    analyst_descs: list[str] = []
    patch_globs = [
        os.path.join(step_dir, "patches", "minibatch_fail_*.json"),
        os.path.join(step_dir, "batch_*", "patches", "minibatch_fail_*.json"),
    ]
    seen_patch_files: set[str] = set()
    for pattern in patch_globs:
        for fname in sorted(glob.glob(pattern)):
            if fname in seen_patch_files:
                continue
            seen_patch_files.add(fname)
            try:
                with open(fname) as f:
                    patch = json.load(f)
                for fs in patch.get("failure_summary", []):
                    ft = fs.get("failure_type", "")
                    sd = fs.get("description", "")
                    analyst_descs.append(f"{ft}: {sd}" if sd else ft)
            except Exception:
                pass

    patterns = []
    desc_iter = iter(analyst_descs)
    for prefix, items in groups.items():
        desc = next(desc_iter, None) or prefix
        patterns.append({
            "pattern": desc,
            "count": len(items),
            "task_ids": [str(r.get("id", "?")) for r in items],
        })
    return patterns


def _format_step_buffer(buffer: list[dict]) -> str:
    """Format the unified step buffer into a single context block.

    Each entry captures what happened at a previous step: failure patterns
    observed during rollout, and — when the step was rejected — the specific
    edits that were tried and the resulting score drop.

    Returns empty string when *buffer* is empty.
    """
    if not buffer:
        return ""

    parts = [
        "Below is a summary of previous steps in this epoch. "
        "Use it to avoid repeating ineffective edits and to prioritise "
        "failure patterns that remain unsolved.\n"
    ]

    for entry in buffer:
        step = entry["step"]
        action = entry["action"]
        n_fail = entry.get("n_fail", 0)
        n_total = entry.get("n_total", "?")

        parts.append(f"### Step {step} — {action.upper()} ({n_fail}/{n_total} failed)")

        # Failure patterns
        for p in entry.get("failure_patterns", []):
            ids = ", ".join(p["task_ids"])
            parts.append(f'  - "{p["pattern"]}" (×{p["count"]}, tasks: {ids})')

        # Rejected edits (only present on reject)
        rejected = entry.get("rejected_edits", [])
        if rejected:
            score_before = entry.get("score_before", "?")
            score_after = entry.get("score_after", "?")
            parts.append(
                f"  Rejected edits (score {score_before} → {score_after}):"
            )
            for i, e in enumerate(rejected, 1):
                if e.get("op") is not None:
                    op = e.get("op", "?")
                    content = e.get("content", "")
                    target = e.get("target", "")
                    if target:
                        parts.append(f'    {i}. [{op}] target="{target}" → "{content}"')
                    else:
                        parts.append(f'    {i}. [{op}] "{content}"')
                else:
                    kind = e.get("type", "?")
                    title = e.get("title", "")
                    instruction = e.get("instruction", "")
                    parts.append(f'    {i}. [{kind}] "{title}" → "{instruction}"')

    return "\n".join(parts)


# ── Trainer ──────────────────────────────────────────────────────────────────

class PatchTreeTrainer:
    """Main PatchTree training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. See ``configs/alfworld_default.yaml``
        for the full list of keys.
    adapter : EnvAdapter
        Environment adapter instance.
    """

    def __init__(self, cfg: dict, adapter: EnvAdapter) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict:
        """Execute the PatchTree training loop. Returns summary dict."""
        cfg = self.cfg
        adapter = self.adapter
        out_root = cfg["out_root"]
        os.makedirs(out_root, exist_ok=True)

        # ── Adapter setup (one-time init) ────────────────────────────
        adapter.setup(cfg)
        dataloader = adapter.get_dataloader()

        def _build_train_env(batch: BatchSpec):
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size, batch.seed

        def _build_eval_env(
            split: str,
            env_num: int,
            seed: int,
            *,
            random_sample: bool = False,
        ):
            if dataloader is None:
                env_manager = adapter.build_eval_env(
                    env_num=env_num,
                    split=split,
                    seed=seed,
                    out_root=out_root,
                    random_sample=random_sample,
                )
                actual_n = len(env_manager) if hasattr(env_manager, "__len__") else env_num
                return env_manager, actual_n

            batch = dataloader.build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                out_root=out_root,
                random_sample=random_sample,
            )
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size

        # ── Configure models ─────────────────────────────────────────────
        backend = cfg.get("model_backend", "azure_openai")
        configure_azure_openai(
            endpoint=(
                cfg.get("azure_openai_endpoint")
                or cfg.get("azure_endpoint")
                or None
            ),
            api_version=(
                cfg.get("azure_openai_api_version")
                or cfg.get("azure_api_version")
                or None
            ),
            api_key=(
                cfg.get("azure_openai_api_key")
                or cfg.get("azure_api_key")
                or None
            ),
            auth_mode=cfg.get("azure_openai_auth_mode") or None,
            ad_scope=cfg.get("azure_openai_ad_scope") or None,
            managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
            optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
            optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
            optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
            optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
            optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
            optimizer_managed_identity_client_id=(
                cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
            ),
            target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
            target_api_version=cfg.get("target_azure_openai_api_version") or None,
            target_api_key=cfg.get("target_azure_openai_api_key") or None,
            target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
            target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
            target_managed_identity_client_id=(
                cfg.get("target_azure_openai_managed_identity_client_id") or None
            ),
        )
        optimizer_backend = cfg.get("optimizer_backend")
        target_backend = cfg.get("target_backend")
        if not optimizer_backend or not target_backend:
            if backend in {"claude", "claude_chat"}:
                optimizer_backend = optimizer_backend or "claude_chat"
                target_backend = target_backend or "claude_chat"
            elif backend in {"codex", "codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "codex_exec"
            elif backend == "claude_code_exec":
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "claude_code_exec"
            elif backend in {"qwen", "qwen_chat"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "qwen_chat"
            else:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "openai_chat"
            cfg["optimizer_backend"] = optimizer_backend
            cfg["target_backend"] = target_backend
        set_optimizer_backend(optimizer_backend)
        set_target_backend(target_backend)
        set_optimizer_deployment(cfg["optimizer_model"])
        set_target_deployment(cfg["target_model"])
        configure_codex_exec(
            path=cfg.get("codex_exec_path", "codex"),
            sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
            profile=cfg.get("codex_exec_profile", ""),
            full_auto=cfg.get("codex_exec_full_auto", False),
            reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
            use_sdk=cfg.get("codex_exec_use_sdk", None),
            network_access=cfg.get("codex_exec_network_access", False),
            web_search=cfg.get("codex_exec_web_search", False),
            approval_policy=cfg.get("codex_exec_approval_policy", "never"),
        )
        configure_claude_code_exec(
            path=cfg.get("claude_code_exec_path", "claude"),
            profile=cfg.get("claude_code_exec_profile", ""),
            use_sdk=cfg.get("claude_code_exec_use_sdk", None),
            effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
            max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
        )
        configure_qwen_chat(
            base_url=cfg.get("qwen_chat_base_url") or None,
            api_key=cfg.get("qwen_chat_api_key") or None,
            temperature=cfg.get("qwen_chat_temperature"),
            timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
            max_tokens=cfg.get("qwen_chat_max_tokens"),
            enable_thinking=cfg.get("qwen_chat_enable_thinking"),
            optimizer_base_url=cfg.get("optimizer_qwen_chat_base_url") or None,
            optimizer_api_key=cfg.get("optimizer_qwen_chat_api_key") or None,
            optimizer_temperature=cfg.get("optimizer_qwen_chat_temperature"),
            optimizer_timeout_seconds=cfg.get("optimizer_qwen_chat_timeout_seconds"),
            optimizer_max_tokens=cfg.get("optimizer_qwen_chat_max_tokens"),
            optimizer_enable_thinking=cfg.get("optimizer_qwen_chat_enable_thinking"),
            target_base_url=cfg.get("target_qwen_chat_base_url") or None,
            target_api_key=cfg.get("target_qwen_chat_api_key") or None,
            target_temperature=cfg.get("target_qwen_chat_temperature"),
            target_timeout_seconds=cfg.get("target_qwen_chat_timeout_seconds"),
            target_max_tokens=cfg.get("target_qwen_chat_max_tokens"),
            target_enable_thinking=cfg.get("target_qwen_chat_enable_thinking"),
        )
        configure_minimax_chat(
            base_url=cfg.get("minimax_base_url") or None,
            api_key=cfg.get("minimax_api_key") or None,
            temperature=cfg.get("minimax_temperature"),
            max_tokens=cfg.get("minimax_max_tokens"),
            enable_thinking=cfg.get("minimax_enable_thinking"),
        )
        minimax_model_cfg = cfg.get("minimax_model")
        if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
            set_target_deployment(str(minimax_model_cfg))
        os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
            "1"
            if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
            else "0"
        )
        reasoning = cfg.get("reasoning_effort", "") or None
        set_reasoning_effort(reasoning)
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )

        # ── Initialize Ray ───────────────────────────────────────────────
        if adapter.requires_ray():
            try:
                import ray
            except ImportError as e:
                raise ImportError(
                    "This environment requires ray, but ray is not installed."
                ) from e

            if not ray.is_initialized():
                ray.init(num_gpus=0)

        # ── Load initial skill ───────────────────────────────────────────
        skill_init_path = os.path.abspath(cfg["skill_init"])
        if os.path.exists(skill_init_path):
            with open(skill_init_path) as f:
                skill_init = f.read()
            print(f"  [initial skill] {skill_init_path} ({len(skill_init)} chars)")
        else:
            skill_init = ""
            print("  [initial skill] no initial skill file — starting from blank")

        # ── Training parameters ──────────────────────────────────────────
        batch_size = cfg["batch_size"]
        num_epochs = cfg["num_epochs"]
        accumulation = cfg["accumulation"]
        seed = cfg["seed"]
        update_mode = "patch"
        use_type_guided_merge = True
        type_guided_version = "v2"
        type_guided_min_support = int(cfg.get("type_guided_min_support", 2) or 2)
        type_guided_max_leaf_groups = _cfg_int(cfg, "type_guided_max_leaf_groups", 8)
        type_guided_tree_depth = max(int(cfg.get("type_guided_tree_depth", 2) or 2), 1)
        requested_tree_builder = str(
            cfg.get("type_guided_tree_builder", "recursive") or "recursive"
        ).strip().lower()
        if requested_tree_builder == "fixed" and type_guided_tree_depth > 3:
            print(
                "  [type-guided] type_guided_tree_depth > 3 is not implemented yet; "
                "using 3"
            )
            type_guided_tree_depth = 3
        type_guided_tree_builder = requested_tree_builder
        if type_guided_tree_builder not in {"fixed", "recursive"}:
            type_guided_tree_builder = "recursive"
        type_guided_max_tree_depth = max(
            int(cfg.get("type_guided_max_tree_depth", 4) or 4), 2,
        )
        type_guided_merge_target_children = max(
            int(cfg.get("type_guided_merge_target_children", 3) or 3), 2,
        )
        type_guided_merge_max_children = max(
            int(cfg.get("type_guided_merge_max_children", 4) or 4),
            type_guided_merge_target_children,
        )
        type_guided_top_mode = str(
            cfg.get("type_guided_top_mode", "auto") or "auto"
        ).strip().lower()
        if type_guided_top_mode not in {"auto", "real_root", "virtual_root"}:
            type_guided_top_mode = "auto"
        type_guided_leaf_fallback = bool(
            cfg.get("type_guided_leaf_fallback", True)
        )
        type_guided_rollout_repeats = max(int(cfg.get("type_guided_rollout_repeats", 3) or 3), 1)
        type_guided_tau_succ = float(cfg.get("type_guided_tau_succ", 1.0) or 1.0)
        type_guided_max_patch_records = max(
            _cfg_int(cfg, "type_guided_max_patch_records", 24), 0,
        )
        type_guided_cache_dir = str(cfg.get("type_guided_cache_dir", "") or "")
        if type_guided_cache_dir:
            type_guided_cache_dir = os.path.abspath(type_guided_cache_dir)
        type_guided_patch_record_workers = int(
            cfg.get("type_guided_patch_record_workers", 0) or cfg.get("analyst_workers", 16) or 16
        )
        type_guided_clustering = bool(cfg.get("type_guided_clustering", False))
        type_guided_cluster_target_size = max(
            int(cfg.get("type_guided_cluster_target_size", 6) or 6), 1,
        )
        type_guided_cluster_max_size = max(
            int(cfg.get("type_guided_cluster_max_size", 10) or 10),
            type_guided_cluster_target_size,
        )
        type_guided_leaf_merge_workers = max(
            int(cfg.get("type_guided_leaf_merge_workers", 4) or 4), 1,
        )
        type_guided_mid_merge_workers = max(
            int(cfg.get("type_guided_mid_merge_workers", 4) or 4), 1,
        )
        type_guided_tail_bank = bool(cfg.get("type_guided_tail_bank", False))
        type_guided_tail_min_support = max(int(cfg.get("type_guided_tail_min_support", 2) or 2), 1)
        type_guided_tail_max_records = max(int(cfg.get("type_guided_tail_max_records", 32) or 32), 0)
        type_guided_tail_max_leaf_groups = max(int(cfg.get("type_guided_tail_max_leaf_groups", 4) or 4), 0)
        type_guided_tail_window_epochs = max(int(cfg.get("type_guided_tail_window_epochs", 1) or 1), 1)
        type_guided_tail_require_cross_step = bool(
            cfg.get("type_guided_tail_require_cross_step", True)
        )
        type_guided_fallback_eval_all_leaves = bool(
            cfg.get("type_guided_fallback_eval_all_leaves", True)
        )
        type_guided_fallback_top_k = max(int(cfg.get("type_guided_fallback_top_k", 0) or 0), 0)
        if not type_guided_fallback_eval_all_leaves and type_guided_fallback_top_k == 0:
            type_guided_fallback_top_k = 1
        type_guided_fallback_tau_child = float(cfg.get("type_guided_fallback_tau_child", 0.0) or 0.0)
        type_guided_fallback_min_leaf_coverage = max(
            int(cfg.get("type_guided_fallback_min_leaf_coverage", 1) or 1), 1,
        )
        type_guided_validation_budget = max(
            int(cfg.get("type_guided_validation_budget", 16) or 16), 1,
        )
        type_guided_fallback_sel_env_num = max(
            int(cfg.get("type_guided_fallback_sel_env_num", 0) or 0), 0,
        )
        type_guided_fallback_reconcile = str(
            cfg.get("type_guided_fallback_reconcile", "llm_fuse") or "llm_fuse"
        ).strip().lower()
        if type_guided_fallback_reconcile not in {
            "off", "deterministic", "llm_select", "llm_fuse",
        }:
            type_guided_fallback_reconcile = "llm_fuse"
        type_guided_fallback_reconcile_min_children = max(
            int(cfg.get("type_guided_fallback_reconcile_min_children", 2) or 2), 1,
        )
        lr_control_mode = _normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if accumulation <= 0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")

        train_size = _resolve_train_size(cfg, dataloader)
        steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
        batches_per_epoch = steps_per_epoch * accumulation
        total_steps = num_epochs * steps_per_epoch

        # Persist resolved derived fields so config.json / summary.json match
        # the actual runtime recipe.
        cfg["train_size"] = train_size
        cfg["steps_per_epoch"] = steps_per_epoch
        cfg["batches_per_epoch"] = batches_per_epoch
        cfg["samples_per_epoch"] = train_size
        cfg["lr_control_mode"] = lr_control_mode
        cfg["type_guided_merge"] = use_type_guided_merge
        cfg["type_guided_version"] = type_guided_version
        cfg["type_guided_min_support"] = type_guided_min_support
        cfg["type_guided_max_leaf_groups"] = type_guided_max_leaf_groups
        cfg["type_guided_tree_depth"] = type_guided_tree_depth
        cfg["type_guided_tree_builder"] = type_guided_tree_builder
        cfg["type_guided_max_tree_depth"] = type_guided_max_tree_depth
        cfg["type_guided_merge_target_children"] = type_guided_merge_target_children
        cfg["type_guided_merge_max_children"] = type_guided_merge_max_children
        cfg["type_guided_top_mode"] = type_guided_top_mode
        cfg["type_guided_leaf_fallback"] = type_guided_leaf_fallback
        cfg["type_guided_rollout_repeats"] = type_guided_rollout_repeats
        cfg["type_guided_tau_succ"] = type_guided_tau_succ
        cfg["type_guided_max_patch_records"] = type_guided_max_patch_records
        cfg["type_guided_cache_dir"] = type_guided_cache_dir
        cfg["type_guided_patch_record_workers"] = type_guided_patch_record_workers
        cfg["type_guided_clustering"] = type_guided_clustering
        cfg["type_guided_cluster_target_size"] = type_guided_cluster_target_size
        cfg["type_guided_cluster_max_size"] = type_guided_cluster_max_size
        cfg["type_guided_leaf_merge_workers"] = type_guided_leaf_merge_workers
        cfg["type_guided_mid_merge_workers"] = type_guided_mid_merge_workers
        cfg["type_guided_tail_bank"] = type_guided_tail_bank
        cfg["type_guided_tail_min_support"] = type_guided_tail_min_support
        cfg["type_guided_tail_max_records"] = type_guided_tail_max_records
        cfg["type_guided_tail_max_leaf_groups"] = type_guided_tail_max_leaf_groups
        cfg["type_guided_tail_window_epochs"] = type_guided_tail_window_epochs
        cfg["type_guided_tail_require_cross_step"] = type_guided_tail_require_cross_step
        cfg["type_guided_fallback_eval_all_leaves"] = type_guided_fallback_eval_all_leaves
        cfg["type_guided_fallback_top_k"] = type_guided_fallback_top_k
        cfg["type_guided_fallback_tau_child"] = type_guided_fallback_tau_child
        cfg["type_guided_fallback_min_leaf_coverage"] = type_guided_fallback_min_leaf_coverage
        cfg["type_guided_validation_budget"] = type_guided_validation_budget
        cfg["type_guided_fallback_sel_env_num"] = type_guided_fallback_sel_env_num
        cfg["type_guided_fallback_reconcile"] = type_guided_fallback_reconcile
        cfg["type_guided_fallback_reconcile_min_children"] = type_guided_fallback_reconcile_min_children

        # Save config after deriving runtime values.
        with open(os.path.join(out_root, "config.json"), "w") as f:
            json.dump(_redact_cfg(cfg), f, indent=2, ensure_ascii=False)

        train_pool_size = train_size

        scheduler = build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=total_steps,
        )

        # Fixed training pool: base seeds (each seed = one deterministic batch)
        if dataloader is not None:
            base_seeds = dataloader.make_base_seeds(
                steps_per_epoch=steps_per_epoch,
                accumulation=accumulation,
                seed=seed,
            )
        else:
            base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]

        print(f"\n  [config] epochs={num_epochs} steps/epoch={steps_per_epoch} "
              f"(auto) accum={accumulation} batch_size={batch_size}")
        print(f"  [config] train_size={train_size}")
        print(f"  [config] batches/epoch={batches_per_epoch} "
              f"total_steps={total_steps} "
              f"games/epoch={train_pool_size}")
        print(f"  [config] lr_scheduler={cfg.get('lr_scheduler', 'constant')} "
              f"edit_budget={cfg['edit_budget']} "
              f"min_edit_budget={cfg.get('min_edit_budget', 2)}")
        print(f"  [config] PatchTree edit mode; lr_control_mode={lr_control_mode}")
        print(
            f"  [config] PatchTree version={type_guided_version} "
            f"tree={type_guided_tree_builder}/depth={type_guided_tree_depth} "
            f"max_depth={type_guided_max_tree_depth} "
            f"fanout={type_guided_merge_target_children}/{type_guided_merge_max_children} "
            f"top={type_guided_top_mode} "
            f"min_support={type_guided_min_support} "
            f"max_leaf_groups={type_guided_max_leaf_groups} "
            f"leaf_fallback={type_guided_leaf_fallback} "
            f"rollout_repeats={type_guided_rollout_repeats} "
            f"leaf_workers={type_guided_leaf_merge_workers} "
            f"mid_workers={type_guided_mid_merge_workers}"
        )
        if use_type_guided_merge and type_guided_version == "v2":
            print(
                f"  [config] type_guided_v2 clustering={type_guided_clustering} "
                f"tail_bank={type_guided_tail_bank} "
                f"tail_min_support={type_guided_tail_min_support} "
                f"tail_window={type_guided_tail_window_epochs}"
            )
        print(f"  [config] base_seeds={base_seeds}")

        # ── Resume check ─────────────────────────────────────────────────
        history = _load_history(out_root)
        runtime_state = _load_runtime_state(out_root)
        if runtime_state:
            last_step = int(runtime_state.get("last_completed_step", 0) or 0)
            current_skill_path = runtime_state.get("current_skill_path") or os.path.join(
                out_root, "skills", f"skill_v{last_step:04d}.md",
            )
            with open(current_skill_path) as f:
                current_skill = f.read()
            best_skill_path = runtime_state.get("best_skill_path") or os.path.join(
                out_root, "best_skill.md",
            )
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = current_skill
            current_score = float(runtime_state.get("current_score", -1.0) or -1.0)
            best_score = float(runtime_state.get("best_score", current_score) or current_score)
            best_step = runtime_state.get("best_step", last_step)
            current_origin = str(
                runtime_state.get("current_origin")
                or (f"step_{last_step:04d}" if last_step > 0 else "initial_skill")
            )
            best_origin = str(runtime_state.get("best_origin") or current_origin)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f} "
                f"(origin={current_origin})"
            )
        elif history:
            last_step = history[-1]["step"]
            current_skill = _load_skill(out_root, last_step)
            best_rec = max(history, key=lambda h: h.get("best_score", 0.0))
            best_score = best_rec["best_score"]
            best_step = best_rec["best_step"]
            best_skill_path = os.path.join(out_root, "best_skill.md")
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = _load_skill(out_root, best_step)
            current_score = history[-1].get("current_score", best_score)
            current_origin = f"step_{last_step:04d}"
            best_origin = f"step_{int(best_step):04d}" if isinstance(best_step, int) else str(best_step)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f}"
            )
        else:
            current_skill = skill_init
            best_skill = skill_init
            best_score = -1.0
            current_score = -1.0
            best_step = 0
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            resume_from = 1

        _save_skill(out_root, 0, skill_init)

        def _persist_runtime_state(last_completed_step: int) -> None:
            _save_runtime_state(
                out_root,
                {
                    "last_completed_step": last_completed_step,
                    "current_skill_path": os.path.join(
                        out_root, "skills", f"skill_v{last_completed_step:04d}.md",
                    ),
                    "current_score": current_score,
                    "current_origin": current_origin,
                    "best_skill_path": os.path.join(out_root, "best_skill.md"),
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_origin": best_origin,
                },
            )

        # ── Selection cache ──────────────────────────────────────────────
        sel_cache: dict[str, tuple[float, float]] = {}
        for rec in history:
            sh = rec.get("candidate_hash", "")
            if sh and rec.get("selection_hard") is not None:
                sel_cache[sh] = (rec["selection_hard"], rec["selection_soft"])

        # ── Baseline evaluation on selection set ─────────────────────────
        # `use_gate=False` keeps validation running (selection rollout +
        # scoring are unconditional below) but force-accepts every candidate
        # instead of gating it; final skill is chosen manually afterwards.
        use_gate = cfg.get("use_gate", True) is not False
        gate_metric = str(cfg.get("gate_metric", "hard")).strip().lower()
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError(
                f"evaluation.gate_metric must be 'hard' | 'soft' | 'mixed', "
                f"got {gate_metric!r}"
            )
        gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5))
        if not 0.0 <= gate_mixed_weight <= 1.0:
            raise ValueError(
                f"evaluation.gate_mixed_weight must be in [0, 1], "
                f"got {gate_mixed_weight}"
            )
        print(
            f"  [gate] metric={gate_metric}"
            + (
                f" mixed_weight={gate_mixed_weight}"
                if gate_metric == "mixed"
                else ""
            )
            + ("" if use_gate
               else "  (DISABLED → validation runs, candidates force-accepted)")
        )
        if current_score < 0:
            print(f"\n{'='*60}")
            print("  BASELINE — evaluate initial skill on Selection set (valid_seen)")
            print(f"{'='*60}")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            print(f"  Selection items: {sel_n}")
            baseline_dir = os.path.join(out_root, "selection_eval_baseline")
            baseline_results = adapter.rollout(sel_env, skill_init, baseline_dir)
            baseline_hard, baseline_soft = compute_score(baseline_results)
            current_score = select_gate_score(
                baseline_hard, baseline_soft, gate_metric, gate_mixed_weight,
            )
            best_score = current_score
            sh = skill_hash(skill_init)
            sel_cache[sh] = (baseline_hard, baseline_soft)
            _save_selection_result_cache(out_root, sh, baseline_results)
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            _persist_runtime_state(0)
            print(
                f"  [baseline result] selection hard={baseline_hard:.4f} "
                f"soft={baseline_soft:.4f} "
                f"gate[{gate_metric}]={current_score:.4f}"
            )

        # ── Training loop ────────────────────────────────────────────────
        t_loop_start = time.time()

        if resume_from > total_steps:
            print(f"\n  [skip] all {total_steps} steps complete — jumping to evaluation")

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            if dataloader is not None:
                epoch_batches = dataloader.plan_train_epoch(
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    accumulation=accumulation,
                    batch_size=batch_size,
                    seed=seed,
                    out_root=out_root,
                )
                shuffled_seeds = [batch.seed for batch in epoch_batches]
            else:
                epoch_batches = []
                epoch_rng = random.Random(seed + epoch * 1000)
                shuffled_seeds = base_seeds.copy()
                epoch_rng.shuffle(shuffled_seeds)

            # Step buffer: accumulates per-step context (failure patterns +
            # rejected edits) within this epoch so optimizers see full history.
            step_buffer: list[dict] = []
            print(
                f"\n  [EPOCH {epoch}/{num_epochs}] "
                f"shuffled_seeds={shuffled_seeds}"
            )

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                if global_step < resume_from:
                    continue

                step_t0 = time.time()
                step_dir = os.path.join(out_root, "steps", f"step_{global_step:04d}")
                os.makedirs(step_dir, exist_ok=True)

                tokens_before = get_token_summary()

                print(
                    f"\n  [STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"{'='*30}"
                )

                step_rec: dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "timing": {},
                    "tokens": {},
                }

                # ── Accumulation: Rollout + Reflect ──────────────────────
                all_rollout_results: list[dict] = []
                all_type_guided_v2_records: list[dict] = []
                type_guided_v2_record_artifacts: list[dict] = []
                accum_rollout_stats: list[dict] = []
                total_rollout_time = 0.0
                total_reflect_time = 0.0

                for a in range(accumulation):
                    batch_idx = step_in_epoch * accumulation + a
                    if dataloader is not None:
                        batch_spec = epoch_batches[batch_idx]
                        train_env, train_n, batch_seed = _build_train_env(batch_spec)
                    else:
                        batch_seed = shuffled_seeds[batch_idx]
                        train_env = adapter.build_train_env(
                            batch_size=batch_size,
                            seed=batch_seed,
                            out_root=out_root,
                        )
                        train_n = len(train_env) if hasattr(train_env, "__len__") else batch_size

                    # Directory routing
                    if accumulation > 1:
                        batch_dir = os.path.join(step_dir, f"batch_{a}")
                    else:
                        batch_dir = step_dir

                    rollout_dir = os.path.join(batch_dir, "rollout")
                    patches_dir = os.path.join(batch_dir, "patches")

                    # ① ROLLOUT ────────────────────────────────────────────
                    t_phase = time.time()
                    repeated_rollouts: list[dict] | None = None
                    flatten_repeats = (
                        use_type_guided_merge
                        and type_guided_version == "v2"
                        and type_guided_rollout_repeats > 1
                        and _can_flatten_type_guided_repeats(train_env)
                    )
                    if flatten_repeats:
                        flat_env, flat_id_map = _flatten_type_guided_repeat_env(
                            train_env,
                            repeats=type_guided_rollout_repeats,
                        )
                        print(
                            f"    [1/6 ROLLOUT flattened] train items={train_n} "
                            f"repeats={type_guided_rollout_repeats} "
                            f"total_items={len(flat_env)} "
                            f"(from pool, batch_seed={batch_seed})"
                        )
                        flat_rollout_results = adapter.rollout(
                            flat_env, current_skill, rollout_dir,
                            use_eval_feedback=True,
                        )
                        repeated_rollouts = _split_flattened_type_guided_results(
                            flat_rollout_results,
                            repeats=type_guided_rollout_repeats,
                            id_map=flat_id_map,
                            prediction_dir=os.path.join(rollout_dir, "predictions"),
                        )
                        rollout_results = repeated_rollouts[0]["results"] if repeated_rollouts else []
                    else:
                        print(f"    [1/6 ROLLOUT] train items={train_n} (from pool, batch_seed={batch_seed})")
                        rollout_results = adapter.rollout(
                            train_env, current_skill, rollout_dir,
                            use_eval_feedback=True,
                        )
                    r_hard, r_soft = compute_score(rollout_results)
                    total_rollout_time += time.time() - t_phase
                    all_rollout_results.extend(rollout_results)
                    print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")
                    if repeated_rollouts:
                        for repeat in repeated_rollouts[1:]:
                            rep_hard, rep_soft = compute_score(repeat.get("results", []) or [])
                            print(
                                f"    [1/6 repeat done] repeat={int(repeat.get('repeat_id', 0)) + 1} "
                                f"hard={rep_hard:.4f} soft={rep_soft:.4f} "
                                f"(flattened)"
                            )

                    # ② REFLECT ────────────────────────────────────────────
                    t_phase = time.time()
                    pred_dir = os.path.join(rollout_dir, "predictions")
                    step_buffer_context = _format_step_buffer(step_buffer)

                    if use_type_guided_merge and type_guided_version == "v2":
                        if repeated_rollouts is None:
                            repeated_rollouts = [{
                                "repeat_id": 0,
                                "results": rollout_results,
                                "prediction_dir": pred_dir,
                            }]
                            for repeat_id in range(1, type_guided_rollout_repeats):
                                rep_t0 = time.time()
                                if dataloader is not None:
                                    repeat_env, _repeat_n, _repeat_seed = _build_train_env(batch_spec)
                                else:
                                    repeat_env = adapter.build_train_env(
                                        batch_size=batch_size,
                                        seed=batch_seed,
                                        out_root=out_root,
                                    )
                                repeat_dir = os.path.join(batch_dir, f"rollout_repeat_{repeat_id}")
                                print(
                                    f"    [1/6 ROLLOUT repeat] "
                                    f"{repeat_id + 1}/{type_guided_rollout_repeats} "
                                    f"train items={train_n}"
                                )
                                repeat_results = adapter.rollout(
                                    repeat_env, current_skill, repeat_dir,
                                    use_eval_feedback=True,
                                )
                                rep_hard, rep_soft = compute_score(repeat_results)
                                total_rollout_time += time.time() - rep_t0
                                repeated_rollouts.append({
                                    "repeat_id": repeat_id,
                                    "results": repeat_results,
                                    "prediction_dir": os.path.join(repeat_dir, "predictions"),
                                })
                                print(
                                    f"    [1/6 repeat done] "
                                    f"repeat={repeat_id + 1} hard={rep_hard:.4f} "
                                    f"soft={rep_soft:.4f}"
                                )

                        records, record_artifact = generate_patch_records(
                            skill_content=current_skill,
                            repeated_rollouts=repeated_rollouts,
                            tau_succ=type_guided_tau_succ,
                            max_patch_records=type_guided_max_patch_records,
                            workers=type_guided_patch_record_workers,
                            optimizer_model=str(cfg.get("optimizer_model", "")),
                            cache_dir=type_guided_cache_dir,
                            step_cache_dir=os.path.join(batch_dir, "type_guided_v2_patch_record_cache"),
                            include_trajectories=True,
                            env_name=str(cfg.get("env") or ""),
                            verbose=True,
                        )
                        all_type_guided_v2_records.extend(records)
                        type_guided_v2_record_artifacts.append({
                            "batch_idx": a,
                            "batch_seed": batch_seed,
                            **record_artifact,
                        })
                        raw_patches = []
                        failure_patches = []
                        success_patches = []
                        total_reflect_time += time.time() - t_phase
                        print(
                            f"    [2/6 done] type_guided_v2_records={len(records)} "
                            f"stable_success={record_artifact.get('n_stable_success', 0)}"
                        )

                    # Track per-batch stats
                    accum_rollout_stats.append({
                        "batch_idx": a,
                        "batch_seed": batch_seed,
                        "n_envs": len(rollout_results),
                        "hard": r_hard,
                        "soft": r_soft,
                        "n_failure_patches": len(failure_patches),
                        "n_success_patches": len(success_patches),
                        "n_type_guided_v2_records": len(records) if (
                            use_type_guided_merge and type_guided_version == "v2"
                        ) else 0,
                    })

                # ── End of accumulation loop ─────────────────────────────

                # Aggregate rollout stats across batches
                total_n = sum(b["n_envs"] for b in accum_rollout_stats)
                agg_hard = sum(b["hard"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)
                agg_soft = sum(b["soft"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)

                step_rec["rollout_hard"] = round(agg_hard, 6)
                step_rec["rollout_soft"] = round(agg_soft, 6)
                step_rec["rollout_n"] = total_n
                step_rec["accumulation_batches"] = accum_rollout_stats
                step_rec["timing"]["rollout_s"] = round(total_rollout_time, 1)
                step_rec["timing"]["reflect_s"] = round(total_reflect_time, 1)

                n_total_patches = len(all_type_guided_v2_records)
                step_rec["n_patches"] = n_total_patches
                step_rec["n_type_guided_v2_records"] = len(all_type_guided_v2_records)

                if accumulation > 1:
                    print(
                        f"    [accum done] total: "
                        f"patch_records={len(all_type_guided_v2_records)} "
                        f"from {accumulation} batches"
                    )

                # ── No patches? Skip ─────────────────────────────────────
                no_update_items = not all_type_guided_v2_records
                if no_update_items:
                    step_rec["action"] = "skip_no_patches"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable patches — skill unchanged")
                    continue

                # ③ AGGREGATE ──────────────────────────────────────────────
                t_phase = time.time()
                type_guided_artifact = None
                if use_type_guided_merge and type_guided_version == "v2":
                    # Re-number records after accumulation so support IDs are
                    # stable across batches in the step artifact.
                    for rec_idx, record in enumerate(all_type_guided_v2_records, start=1):
                        record["record_id"] = f"R{rec_idx:04d}"
                    with open(os.path.join(step_dir, "type_guided_v2_patch_records.json"), "w") as f:
                        json.dump(all_type_guided_v2_records, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_rollouts.json"), "w") as f:
                        json.dump(type_guided_v2_record_artifacts, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_rollouts.jsonl"), "w") as f:
                        for artifact_row in type_guided_v2_record_artifacts:
                            f.write(json.dumps(artifact_row, ensure_ascii=False) + "\n")
                    merged_patch, type_guided_artifact = merge_type_guided_v2_records(
                        skill_content=current_skill,
                        patch_records=all_type_guided_v2_records,
                        min_support=type_guided_min_support,
                        max_leaf_groups=type_guided_max_leaf_groups,
                        tree_depth=type_guided_tree_depth,
                        tree_builder=type_guided_tree_builder,
                        max_tree_depth=type_guided_max_tree_depth,
                        merge_target_children=type_guided_merge_target_children,
                        merge_max_children=type_guided_merge_max_children,
                        top_mode=type_guided_top_mode,
                        clustering_enabled=type_guided_clustering,
                        cluster_target_size=type_guided_cluster_target_size,
                        cluster_max_size=type_guided_cluster_max_size,
                        leaf_merge_workers=type_guided_leaf_merge_workers,
                        mid_merge_workers=type_guided_mid_merge_workers,
                        cache_dir=type_guided_cache_dir,
                        optimizer_model=str(cfg.get("optimizer_model", "")),
                        verbose=True,
                    )
                    with open(os.path.join(step_dir, "type_guided_v2_clustering.json"), "w") as f:
                        json.dump(type_guided_artifact.get("clustering", {}), f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_leaf_clusters.json"), "w") as f:
                        json.dump({
                            "groups": type_guided_artifact.get("groups", []),
                            "kept_groups": type_guided_artifact.get("kept_groups", []),
                            "dropped_groups": type_guided_artifact.get("dropped_groups", []),
                            "leaf_patches": type_guided_artifact.get("leaf_patches", []),
                        }, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_mid_nodes.json"), "w") as f:
                        json.dump({
                            "tree_depth": type_guided_artifact.get("tree_depth", type_guided_tree_depth),
                            "root_children_level": type_guided_artifact.get("root_children_level", "leaf"),
                            "tree_builder": type_guided_artifact.get("hierarchy", {}).get("builder", "fixed"),
                            "top_mode": type_guided_artifact.get("hierarchy", {}).get("top_mode", "real_root"),
                            "virtual_root": type_guided_artifact.get("hierarchy", {}).get("virtual_root", False),
                            "mid_plan": type_guided_artifact.get("mid_plan", {}),
                            "mid_groups": type_guided_artifact.get("mid_groups", []),
                            "mid_patches": type_guided_artifact.get("mid_patches", []),
                        }, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_root.json"), "w") as f:
                        json.dump(type_guided_artifact.get("root_patch", merged_patch), f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_cache_report.json"), "w") as f:
                        json.dump({
                            "patch_record_batches": [
                                {
                                    "batch_idx": row.get("batch_idx"),
                                    "analyst_reports": row.get("analyst_reports", []),
                                    "timing_s": row.get("timing_s"),
                                }
                                for row in type_guided_v2_record_artifacts
                            ],
                            "cache_dir": type_guided_cache_dir,
                            "clustering": {
                                "enabled": type_guided_artifact.get("clustering", {}).get("enabled"),
                                "status": type_guided_artifact.get("clustering", {}).get("status"),
                                "bucket_reports": type_guided_artifact.get("clustering", {}).get("bucket_reports", []),
                            },
                        }, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(step_dir, "type_guided_v2_merge_artifact.json"), "w") as f:
                        json.dump(type_guided_artifact, f, ensure_ascii=False, indent=2)
                    tail_rows = []
                    if type_guided_tail_bank:
                        tail_rows = _extract_type_guided_tail_records(
                            dropped_groups=type_guided_artifact.get("dropped_groups", []),
                            epoch=epoch,
                            step=global_step,
                            current_skill=current_skill,
                        )
                        tail_dir = os.path.join(out_root, "type_guided_tail_bank", f"epoch_{epoch:02d}")
                        _append_jsonl(os.path.join(tail_dir, "tail_records.jsonl"), tail_rows)
                        _append_jsonl(os.path.join(out_root, "type_guided_tail_bank", "tail_records_all.jsonl"), tail_rows)
                        with open(os.path.join(step_dir, "type_guided_v2_tail_records.jsonl"), "w") as f:
                            for row in tail_rows:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    step_rec["type_guided_merge"] = {
                        "enabled": True,
                        "version": "v2",
                        "n_patch_records": len(all_type_guided_v2_records),
                        "raw_edit_count": type_guided_artifact.get("raw_edit_count", 0),
                        "n_leaf_groups": len(type_guided_artifact.get("kept_groups", [])),
                        "n_dropped_groups": len(type_guided_artifact.get("dropped_groups", [])),
                        "tree_depth": type_guided_artifact.get("tree_depth", type_guided_tree_depth),
                        "root_children_level": type_guided_artifact.get("root_children_level", "leaf"),
                        "n_mid_nodes": len(type_guided_artifact.get("mid_patches", [])),
                        "n_tail_records": len(tail_rows),
                        "clustering": type_guided_clustering,
                        "n_clusters": len(type_guided_artifact.get("clustering", {}).get("clusters", [])),
                        "tail_bank": type_guided_tail_bank,
                    }
                with open(os.path.join(step_dir, "merged_patch.json"), "w") as f:
                    json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                merged_items = get_payload_items(merged_patch, update_mode)
                n_edits_merged = len(merged_items)
                step_rec["n_edits_merged"] = n_edits_merged
                step_rec["timing"]["aggregate_s"] = round(time.time() - t_phase, 1)
                print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

                # ④ SELECT ─────────────────────────────────────────────────
                t_phase = time.time()
                lr_decision = None
                if lr_control_mode == "autonomous":
                    lr_decision = decide_autonomous_learning_rate(
                        skill_content=current_skill,
                        merged_patch=merged_patch,
                        update_mode=update_mode,
                        rollout_hard=agg_hard,
                        rollout_soft=agg_soft,
                        rollout_n=total_n,
                        step_buffer_context=step_buffer_context,
                    )
                    edit_budget = int(lr_decision["learning_rate"])
                    with open(os.path.join(step_dir, "lr_decision.json"), "w") as f:
                        json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            **lr_decision,
                        }, ensure_ascii=False) + "\n")
                else:
                    edit_budget = scheduler.step()
                ranked_patch = rank_and_select(
                    current_skill, merged_patch,
                    max_edits=edit_budget,
                    update_mode="patch",
                )
                with open(os.path.join(step_dir, "ranked_edits.json"), "w") as f:
                    json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

                ranked_items = get_payload_items(ranked_patch, "patch")
                n_edits_ranked = len(ranked_items)
                step_rec["n_edits_ranked"] = n_edits_ranked
                step_rec["edit_budget"] = edit_budget
                step_rec["lr_control_mode"] = lr_control_mode
                if lr_decision is not None:
                    step_rec["lr_decision"] = lr_decision
                step_rec["timing"]["select_s"] = round(time.time() - t_phase, 1)

                support_counts = [
                    item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
                ]
                step_rec["support_counts"] = support_counts
                print(
                    f"    [4/6 SELECT] "
                    f"{n_edits_merged} -> {n_edits_ranked} edits "
                    f"(budget={edit_budget}, lr_control={lr_control_mode})"
                )

                if n_edits_ranked == 0:
                    step_rec["action"] = "skip_no_ranked_edits"
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no ranked edits — skill unchanged")
                    continue

                # ⑤ UPDATE ─────────────────────────────────────────────────
                t_phase = time.time()
                candidate_skill, apply_report = apply_patch_with_report(current_skill, ranked_patch)
                with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                    f.write(candidate_skill)
                if apply_report:
                    with open(os.path.join(step_dir, "edit_apply_report.json"), "w") as f:
                        json.dump(apply_report, f, indent=2, ensure_ascii=False)

                cand_hash = skill_hash(candidate_skill)
                step_rec["candidate_hash"] = cand_hash
                step_rec["candidate_skill_len"] = len(candidate_skill)
                if apply_report:
                    step_rec["edit_apply_summary"] = {
                        "total": len(apply_report),
                        "applied": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("applied")
                        ),
                        "skipped": sum(
                            1 for row in apply_report if str(row.get("status", "")).startswith("skipped")
                        ),
                        "errors": sum(
                            1 for row in apply_report if row.get("status") == "error"
                        ),
                    }
                step_rec["timing"]["update_s"] = round(time.time() - t_phase, 1)
                print(
                    f"    [5/6 UPDATE] "
                    f"skill_len {len(current_skill)} -> {len(candidate_skill)}"
                )

                # ⑥ EVALUATE ───────────────────────────────────────────────
                t_phase = time.time()
                virtual_root_candidate = bool(
                    use_gate
                    and type_guided_artifact
                    and type_guided_artifact.get("hierarchy", {}).get(
                        "virtual_root", False,
                    )
                )
                current_hash_for_virtual = skill_hash(current_skill)
                current_full_scores = sel_cache.get(current_hash_for_virtual)
                if virtual_root_candidate and current_full_scores is None:
                    current_cached_results = _load_selection_result_cache(
                        out_root, current_hash_for_virtual,
                    )
                    if current_cached_results:
                        current_full_scores = compute_score(current_cached_results)
                        sel_cache[current_hash_for_virtual] = current_full_scores
                if virtual_root_candidate and current_full_scores is not None:
                    cand_hard, cand_soft = current_full_scores
                    step_rec["virtual_root_full_eval_skipped"] = True
                    print(
                        "    [6/6 EVALUATE] virtual root is non-executable; "
                        "skip carrier validation and start branch-wise fallback"
                    )
                elif cand_hash in sel_cache:
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    print(
                        f"    [6/6 EVALUATE] "
                        f"cache hit {cand_hash}: hard={cand_hard:.4f}"
                    )
                else:
                    sel_env, sel_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    print(f"    [6/6 EVALUATE] selection items={sel_n}")
                    sel_eval_dir = os.path.join(step_dir, "selection_eval")
                    sel_results = adapter.rollout(sel_env, candidate_skill, sel_eval_dir)
                    cand_hard, cand_soft = compute_score(sel_results)
                    sel_cache[cand_hash] = (cand_hard, cand_soft)
                    _save_selection_result_cache(out_root, cand_hash, sel_results)

                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft

                gate = evaluate_gate(
                    candidate_skill=candidate_skill,
                    cand_hard=cand_hard,
                    current_skill=current_skill,
                    current_score=current_score,
                    best_skill=best_skill,
                    best_score=best_score,
                    best_step=best_step,
                    global_step=global_step,
                    cand_soft=cand_soft,
                    metric=gate_metric,
                    mixed_weight=gate_mixed_weight,
                ) if use_gate else None
                cand_gate_score = select_gate_score(
                    cand_hard, cand_soft, gate_metric, gate_mixed_weight,
                )
                if not use_gate:
                    # Validation ran (scores recorded above) but the gate is
                    # disabled: force-accept the candidate as the new current
                    # skill. Best-so-far is still tracked for convenience; the
                    # final skill is selected manually from the trajectory.
                    if cand_gate_score > best_score:
                        fa_best_skill = candidate_skill
                        fa_best_score = cand_gate_score
                        fa_best_step = global_step
                    else:
                        fa_best_skill = best_skill
                        fa_best_score = best_score
                        fa_best_step = best_step
                    gate = GateResult(
                        action="force_accept",
                        current_skill=candidate_skill,
                        current_score=cand_gate_score,
                        best_skill=fa_best_skill,
                        best_score=fa_best_score,
                        best_step=fa_best_step,
                    )
                elif (
                    use_type_guided_merge
                    and type_guided_leaf_fallback
                    and type_guided_artifact
                    and gate is not None
                    and (
                        gate.action == "reject"
                        or bool(
                            type_guided_artifact.get("hierarchy", {}).get(
                                "virtual_root", False,
                            )
                        )
                    )
                ):
                    fallback_t0 = time.time()
                    hierarchy = type_guided_artifact.get("hierarchy", {})
                    if not isinstance(hierarchy, dict):
                        hierarchy = {}
                    recursive_fallback = hierarchy.get("builder") == "recursive"
                    virtual_root = bool(hierarchy.get("virtual_root", False))
                    if virtual_root and gate.action != "reject":
                        # A virtual root is not an executable abstraction. Even if
                        # its carrier edit set passed, branch-wise selection must
                        # begin from its top executable nodes.
                        gate = GateResult(
                            action="reject",
                            current_skill=current_skill,
                            current_score=current_score,
                            best_skill=best_skill,
                            best_score=best_score,
                            best_step=best_step,
                        )
                    fallback_level = str(
                        type_guided_artifact.get("root_children_level") or "leaf"
                    )
                    child_patches = (
                        hierarchy.get("fallback_top_patches")
                        if recursive_fallback
                        else type_guided_artifact.get("root_child_patches")
                    )
                    if not isinstance(child_patches, list):
                        child_patches = type_guided_artifact.get("leaf_patches", [])
                        fallback_level = "leaf"
                    node_by_id = hierarchy.get("node_by_id", {})
                    if not isinstance(node_by_id, dict):
                        node_by_id = {}
                    child_patches = [
                        child for child in child_patches
                        if isinstance(child, dict) and get_payload_items(child, update_mode)
                    ]
                    if type_guided_fallback_top_k > 0 and len(child_patches) > type_guided_fallback_top_k:
                        child_patches = sorted(
                            child_patches,
                            key=lambda child: (
                                -int(child.get("support_count", 0) or 0),
                                str(child.get("mid_id") or child.get("leaf_id", "")),
                            ),
                        )[:type_guided_fallback_top_k]
                    fallback_env_num = (
                        type_guided_fallback_sel_env_num
                        if type_guided_fallback_sel_env_num > 0
                        else cfg["sel_env_num"]
                    )
                    # Keep the representative subset fixed across steps so child
                    # comparisons are not confounded by a changing validation sample.
                    fallback_seed = _type_guided_fallback_sample_seed(seed)
                    fallback_env = None
                    fallback_n = 0
                    fallback_item_ids: list[str] = []
                    fallback_current_hard = None
                    fallback_current_soft = None
                    fallback_current_score = current_score
                    fallback_current_source = "full_selection_score"
                    fallback_current_eval_dir = ""
                    if child_patches:
                        fallback_env, fallback_n = _build_eval_env(
                            split="valid_seen",
                            env_num=fallback_env_num,
                            seed=fallback_seed,
                            random_sample=True,
                        )
                        fallback_item_ids = _environment_item_ids(fallback_env)
                        current_hash = skill_hash(current_skill)
                        current_cached = _load_selection_result_cache(
                            out_root, current_hash,
                        )
                        current_subset, missing_ids = _slice_selection_results(
                            current_cached, fallback_item_ids,
                        )
                        if (
                            fallback_item_ids
                            and not missing_ids
                            and len(current_subset) == fallback_n
                        ):
                            fallback_current_hard, fallback_current_soft = compute_score(
                                current_subset,
                            )
                            fallback_current_source = "full_selection_cache_slice"
                        else:
                            fallback_current_eval_dir = os.path.join(
                                step_dir, "type_guided_fallback_current_eval",
                            )
                            current_subset = adapter.rollout(
                                fallback_env,
                                current_skill,
                                fallback_current_eval_dir,
                            )
                            fallback_current_hard, fallback_current_soft = compute_score(
                                current_subset,
                            )
                            fallback_current_source = "subset_rollout"
                        fallback_current_score = select_gate_score(
                            fallback_current_hard,
                            fallback_current_soft,
                            gate_metric,
                            gate_mixed_weight,
                        )
                    fallback_rec = {
                        "attempted": bool(child_patches),
                        "fallback_level": fallback_level,
                        "root_gate_score": cand_gate_score,
                        "current_score": current_score,
                        "current_subset": {
                            "hard": fallback_current_hard,
                            "soft": fallback_current_soft,
                            "gate_score": fallback_current_score,
                            "source": fallback_current_source,
                            "eval_dir": fallback_current_eval_dir,
                        },
                        "tau_child": type_guided_fallback_tau_child,
                        "fallback_sel_env_num": fallback_env_num,
                        "fallback_sample_seed": fallback_seed,
                        "fallback_sample_size": fallback_n,
                        "fallback_sample_ids": fallback_item_ids,
                        "sampling": "fixed_random_without_replacement",
                        "eval_all_leaves": type_guided_fallback_eval_all_leaves,
                        "top_k": type_guided_fallback_top_k,
                        "recursive": recursive_fallback,
                        "virtual_root": virtual_root,
                        "min_leaf_coverage": type_guided_fallback_min_leaf_coverage,
                        "validation_budget": type_guided_validation_budget,
                        "node_evaluations": 0,
                        "budget_exhausted": False,
                        "reconcile": {
                            "mode": type_guided_fallback_reconcile,
                            "min_children": type_guided_fallback_reconcile_min_children,
                        },
                        "version": type_guided_version if use_type_guided_merge else "off",
                        "leaf_results": [],
                        "kept_leaf_ids": [],
                        "child_results": [],
                        "kept_child_ids": [],
                        "accepted": False,
                    }
                    kept_child_patches: list[dict] = []
                    child_subset_cache: dict[str, tuple[float, float, str]] = {}
                    pending_children = list(child_patches)
                    visited_child_ids: set[str] = set()
                    node_budget = (
                        type_guided_validation_budget
                        if recursive_fallback
                        else max(len(pending_children), 1)
                    )
                    while pending_children and fallback_rec["node_evaluations"] < node_budget:
                        pending_children.sort(
                            key=lambda node: (
                                -int(
                                    node.get("leaf_coverage")
                                    or len(node.get("leaf_ids") or [])
                                    or 1
                                ),
                                str(
                                    node.get("node_id")
                                    or node.get("mid_id")
                                    or node.get("leaf_id")
                                    or node.get("record_id")
                                    or ""
                                ),
                            ),
                        )
                        child = pending_children.pop(0)
                        child_id = str(
                            child.get("node_id")
                            or child.get("mid_id")
                            or child.get("leaf_id")
                            or child.get("record_id")
                            or ""
                        )
                        if not child_id or child_id in visited_child_ids:
                            continue
                        visited_child_ids.add(child_id)
                        fallback_rec["node_evaluations"] += 1
                        child_level = str(
                            child.get("node_level")
                            or ("mid" if child.get("mid_id") else "leaf")
                        )
                        leaf_id = str(child.get("leaf_id") or child_id)
                        child_candidate, child_apply_report = apply_patch_with_report(
                            current_skill, child,
                        )
                        child_hash = skill_hash(child_candidate)
                        if child_hash in child_subset_cache:
                            child_hard, child_soft, child_score_source = (
                                child_subset_cache[child_hash]
                            )
                            child_eval_dir = ""
                        else:
                            child_cached = _load_selection_result_cache(
                                out_root, child_hash,
                            )
                            child_subset, child_missing_ids = _slice_selection_results(
                                child_cached, fallback_item_ids,
                            )
                            if (
                                fallback_item_ids
                                and not child_missing_ids
                                and len(child_subset) == fallback_n
                            ):
                                child_hard, child_soft = compute_score(child_subset)
                                child_eval_dir = ""
                                child_score_source = "full_selection_cache_slice"
                            else:
                                child_eval_dir = os.path.join(
                                    step_dir,
                                    f"type_guided_{child_level}_eval",
                                    child_id or child_hash,
                                )
                                child_results = adapter.rollout(
                                    fallback_env, child_candidate, child_eval_dir,
                                )
                                child_hard, child_soft = compute_score(child_results)
                                child_score_source = "subset_rollout"
                            child_subset_cache[child_hash] = (
                                child_hard, child_soft, child_score_source,
                            )
                        child_score = select_gate_score(
                            child_hard, child_soft, gate_metric, gate_mixed_weight,
                        )
                        improvement = child_score - fallback_current_score
                        keep_child = improvement > type_guided_fallback_tau_child
                        child_row = {
                            "child_id": child_id,
                            "child_level": child_level,
                            "leaf_id": leaf_id,
                            "mid_id": child.get("mid_id", ""),
                            "leaf_ids": child.get("leaf_ids", []),
                            "question_type": child.get("question_type", ""),
                            "revision_type": child.get("revision_type", ""),
                            "support_count": child.get("support_count", 0),
                            "hash": child_hash,
                            "hard": child_hard,
                            "soft": child_soft,
                            "gate_score": child_score,
                            "current_subset_gate_score": fallback_current_score,
                            "improvement": improvement,
                            "kept": keep_child,
                            "eval_scope": "shared_random_subset",
                            "eval_items": fallback_n,
                            "sample_seed": fallback_seed,
                            "score_source": child_score_source,
                            "eval_dir": child_eval_dir,
                            "apply_report": child_apply_report,
                        }
                        fallback_rec["child_results"].append(child_row)
                        if child_level == "leaf":
                            fallback_rec["leaf_results"].append(child_row)
                        if keep_child:
                            kept_child_patches.append(child)
                            fallback_rec["kept_child_ids"].append(child_id)
                            if child_level == "leaf":
                                fallback_rec["kept_leaf_ids"].append(leaf_id)
                        elif recursive_fallback:
                            next_children = _recursive_fallback_children(
                                child,
                                node_by_id=node_by_id,
                                min_leaf_coverage=type_guided_fallback_min_leaf_coverage,
                            )
                            if next_children:
                                child_row["descended"] = True
                                child_row["descended_to"] = [
                                    str(
                                        node.get("node_id")
                                        or node.get("mid_id")
                                        or node.get("leaf_id")
                                        or node.get("record_id")
                                        or ""
                                    )
                                    for node in next_children
                                ]
                                pending_children.extend(next_children)
                            else:
                                child_row["descended"] = False
                    fallback_rec["budget_exhausted"] = bool(pending_children)

                    if kept_child_patches:
                        if type_guided_fallback_reconcile == "off":
                            combined_edits: list[dict] = []
                            reconcile_report = {
                                "mode": "off",
                                "status": "skipped",
                                "n_children": len(kept_child_patches),
                            }
                            for child in kept_child_patches:
                                combined_edits.extend(get_payload_items(child, update_mode))
                            reconcile_report["n_output_edits"] = len(combined_edits)
                        elif (
                            type_guided_fallback_reconcile == "llm_fuse"
                            and len(kept_child_patches)
                            >= type_guided_fallback_reconcile_min_children
                        ):
                            combined_edits, reconcile_report = (
                                _fuse_validated_frontier_edits(
                                    skill_content=current_skill,
                                    child_patches=kept_child_patches,
                                    child_rows=fallback_rec["child_results"],
                                    update_mode=update_mode,
                                )
                            )
                        elif type_guided_fallback_reconcile == "llm_fuse":
                            combined_edits, reconcile_report = _reconcile_fallback_edits(
                                child_patches=kept_child_patches,
                                child_rows=fallback_rec["child_results"],
                                update_mode=update_mode,
                                mode="deterministic",
                                min_children=1,
                            )
                            reconcile_report.update({
                                "mode": "llm_fuse",
                                "status": "skipped_below_min_children",
                                "min_children": type_guided_fallback_reconcile_min_children,
                            })
                        else:
                            combined_edits, reconcile_report = _reconcile_fallback_edits(
                                child_patches=kept_child_patches,
                                child_rows=fallback_rec["child_results"],
                                update_mode=update_mode,
                                mode=type_guided_fallback_reconcile,
                                min_children=type_guided_fallback_reconcile_min_children,
                            )
                        fallback_rec["reconcile"] = reconcile_report
                        child_combo_patch = {
                            "reasoning": (
                                f"type-guided fallback: combine {fallback_level} patches "
                                "that individually improved validation score; "
                                f"reconcile={reconcile_report.get('status', 'unknown')}"
                            ),
                            "edits": combined_edits,
                        }
                        combo_candidate, combo_apply_report = apply_patch_with_report(
                            current_skill, child_combo_patch,
                        )
                        combo_hash = skill_hash(combo_candidate)
                        with open(os.path.join(step_dir, "type_guided_leaf_combination_patch.json"), "w") as f:
                            json.dump(child_combo_patch, f, ensure_ascii=False, indent=2)
                        with open(os.path.join(step_dir, "type_guided_leaf_combination_candidate.md"), "w") as f:
                            f.write(combo_candidate)
                        if combo_hash in sel_cache:
                            combo_hard, combo_soft = sel_cache[combo_hash]
                            combo_eval_dir = ""
                            combo_eval_n = int(cfg["sel_env_num"] or 0)
                        else:
                            combo_env, combo_eval_n = _build_eval_env(
                                split="valid_seen",
                                env_num=cfg["sel_env_num"],
                                seed=seed,
                            )
                            combo_eval_dir = os.path.join(
                                step_dir, "type_guided_leaf_combination_eval",
                            )
                            combo_results = adapter.rollout(
                                combo_env, combo_candidate, combo_eval_dir,
                            )
                            combo_hard, combo_soft = compute_score(combo_results)
                            sel_cache[combo_hash] = (combo_hard, combo_soft)
                            _save_selection_result_cache(
                                out_root, combo_hash, combo_results,
                            )
                        combo_gate = evaluate_gate(
                            candidate_skill=combo_candidate,
                            cand_hard=combo_hard,
                            current_skill=current_skill,
                            current_score=current_score,
                            best_skill=best_skill,
                            best_score=best_score,
                            best_step=best_step,
                            global_step=global_step,
                            cand_soft=combo_soft,
                            metric=gate_metric,
                            mixed_weight=gate_mixed_weight,
                        )
                        combo_gate_score = select_gate_score(
                            combo_hard, combo_soft, gate_metric, gate_mixed_weight,
                        )
                        fallback_rec["combination"] = {
                            "hash": combo_hash,
                            "hard": combo_hard,
                            "soft": combo_soft,
                            "gate_score": combo_gate_score,
                            "action": combo_gate.action,
                            "eval_scope": "full_selection",
                            "eval_items": combo_eval_n,
                            "eval_dir": combo_eval_dir,
                            "apply_report": combo_apply_report,
                        }
                        if combo_gate.action in {"accept", "accept_new_best"}:
                            gate = combo_gate
                            candidate_skill = combo_candidate
                            apply_report = combo_apply_report
                            cand_hash = combo_hash
                            cand_hard = combo_hard
                            cand_soft = combo_soft
                            cand_gate_score = combo_gate_score
                            fallback_rec["accepted"] = True
                            with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                                f.write(candidate_skill)
                            with open(os.path.join(step_dir, "edit_apply_report.json"), "w") as f:
                                json.dump(apply_report, f, indent=2, ensure_ascii=False)
                            step_rec["candidate_hash"] = combo_hash
                            step_rec["candidate_skill_len"] = len(combo_candidate)
                            step_rec["edit_apply_summary"] = {
                                "total": len(apply_report),
                                "applied": sum(
                                    1 for row in apply_report
                                    if str(row.get("status", "")).startswith("applied")
                                ),
                                "skipped": sum(
                                    1 for row in apply_report
                                    if str(row.get("status", "")).startswith("skipped")
                                ),
                                "errors": sum(
                                    1 for row in apply_report
                                    if row.get("status") == "error"
                                ),
                            }
                            step_rec["type_guided_fallback_selected"] = "leaf_combination"
                            if fallback_level != "leaf":
                                step_rec["type_guided_fallback_selected"] = f"{fallback_level}_combination"
                    fallback_rec["timing_s"] = round(time.time() - fallback_t0, 1)
                    with open(os.path.join(step_dir, "type_guided_leaf_fallback.json"), "w") as f:
                        json.dump(fallback_rec, f, ensure_ascii=False, indent=2)
                    if type_guided_version == "v2":
                        with open(os.path.join(step_dir, "type_guided_v2_fallback.json"), "w") as f:
                            json.dump(fallback_rec, f, ensure_ascii=False, indent=2)
                    step_rec["type_guided_leaf_fallback"] = {
                        "attempted": fallback_rec["attempted"],
                        "fallback_level": fallback_level,
                        "n_leaves": len(child_patches) if fallback_level == "leaf" else len(type_guided_artifact.get("leaf_patches", []) or []),
                        "n_children": len(child_patches),
                        "n_kept": len(fallback_rec["kept_child_ids"]),
                        "accepted": fallback_rec["accepted"],
                    }
                    if fallback_rec["accepted"]:
                        print(
                            f"    [type-guided fallback] accepted {fallback_level} combination "
                            f"with {len(fallback_rec['kept_child_ids'])}/{len(child_patches)} children"
                        )
                    elif child_patches:
                        print(
                            f"    [type-guided fallback] no {fallback_level} combination passed "
                            f"({len(fallback_rec['kept_child_ids'])}/{len(child_patches)} children kept)"
                        )
                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft
                step_rec["gate_metric"] = gate_metric
                step_rec["candidate_gate_score"] = cand_gate_score
                step_rec["action"] = gate.action
                prev_current = current_score
                prev_best = best_score
                current_skill = gate.current_skill
                current_score = gate.current_score
                best_skill = gate.best_skill
                best_score = gate.best_score
                best_step = gate.best_step
                if gate.action in {"accept", "accept_new_best", "force_accept"}:
                    current_origin = f"step_{global_step:04d}"
                if gate.action == "accept_new_best" or (
                    gate.action == "force_accept" and best_step == global_step
                ):
                    best_origin = current_origin

                if gate_metric == "hard":
                    score_label = f"hard={cand_hard:.4f}"
                elif gate_metric == "soft":
                    score_label = f"soft={cand_soft:.4f}"
                else:
                    score_label = (
                        f"mixed[w={gate_mixed_weight}]={cand_gate_score:.4f} "
                        f"(hard={cand_hard:.4f} soft={cand_soft:.4f})"
                    )
                if gate.action == "accept_new_best":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT (new best) "
                        f"{score_label} > prev best {prev_best:.4f}"
                    )
                elif gate.action == "accept":
                    print(
                        f"    [6/6 EVALUATE] ACCEPT "
                        f"{score_label} > current={prev_current:.4f}"
                    )
                elif gate.action == "force_accept":
                    print(
                        f"    [6/6 EVALUATE] FORCE-ACCEPT (gate disabled) "
                        f"{score_label}"
                    )
                else:
                    print(
                        f"    [6/6 EVALUATE] REJECT "
                        f"{score_label} <= current={current_score:.4f}"
                    )

                step_rec["timing"]["evaluate_s"] = round(time.time() - t_phase, 1)

                # ── Step buffer: unified failure patterns + rejected edits ─
                action = step_rec.get("action", "unknown")
                n_total = len(all_rollout_results) or 1
                n_fail = sum(1 for r in all_rollout_results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9)
                failure_patterns = _extract_failure_patterns(
                    all_rollout_results, step_dir,
                )

                buf_entry: dict = {
                    "step": global_step,
                    "action": action,
                    "n_total": n_total,
                    "n_fail": n_fail,
                    "failure_patterns": failure_patterns,
                }

                # Attach rejected edits when the step was rejected
                if "reject" in action and ranked_patch:
                    rejected_edits = [
                        short_item_summary(item, update_mode)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    buf_entry["score_before"] = current_score
                    buf_entry["score_after"] = cand_gate_score
                    buf_entry["rejected_edits"] = rejected_edits

                step_buffer.append(buf_entry)

                # Persist step digest for step buffer context
                digest_path = os.path.join(step_dir, "trajectory_digest.json")
                with open(digest_path, "w") as f:
                    json.dump(buf_entry, f, indent=2, ensure_ascii=False)

                # ── Token snapshot ───────────────────────────────────────
                tokens_after = get_token_summary()
                step_tokens: dict = {}
                for stage in tokens_after:
                    if stage == "_total":
                        continue
                    after = tokens_after[stage]
                    before = tokens_before.get(stage, {})
                    step_tokens[stage] = {
                        "calls": after.get("calls", 0) - before.get("calls", 0),
                        "prompt_tokens": after.get("prompt_tokens", 0)
                        - before.get("prompt_tokens", 0),
                        "completion_tokens": after.get("completion_tokens", 0)
                        - before.get("completion_tokens", 0),
                    }
                step_rec["tokens"] = step_tokens

                # ── Save state ───────────────────────────────────────────
                step_rec["current_score"] = current_score
                step_rec["best_score"] = best_score
                step_rec["best_step"] = best_step
                step_rec["current_origin"] = current_origin
                step_rec["best_origin"] = best_origin
                step_rec["skill_len"] = len(current_skill)
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)

                _save_skill(out_root, global_step, current_skill)
                with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                    f.write(best_skill)
                history.append(step_rec)
                _save_history(out_root, history)
                _persist_runtime_state(global_step)
                with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                    json.dump(step_rec, f, indent=2, ensure_ascii=False)

                timing = step_rec["timing"]
                print(
                    f"\n  [STEP {global_step} done] "
                    f"epoch={epoch} action={step_rec['action']} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={step_rec['wall_time_s']}s\n"
                    f"    timing: rollout={timing.get('rollout_s',0)}s "
                    f"reflect={timing.get('reflect_s',0)}s "
                    f"aggregate={timing.get('aggregate_s',0)}s "
                    f"select={timing.get('select_s',0)}s "
                    f"evaluate={timing.get('evaluate_s',0)}s"
                )

            # ── TYPE-GUIDED TAIL-BANK (end of epoch) ───────────────────
            if (
                use_type_guided_merge
                and type_guided_version == "v2"
                and type_guided_tail_bank
            ):
                tail_dir = os.path.join(out_root, "type_guided_tail_bank", f"epoch_{epoch:02d}")
                os.makedirs(tail_dir, exist_ok=True)
                tail_done_path = os.path.join(tail_dir, "tail_result.json")
                if os.path.exists(tail_done_path):
                    print(f"\n  [TYPE-GUIDED TAIL epoch {epoch}] resumed — already done")
                    try:
                        with open(tail_done_path) as f:
                            tail_saved = json.load(f)
                        if tail_saved.get("accepted_skill_path") and os.path.exists(tail_saved["accepted_skill_path"]):
                            with open(tail_saved["accepted_skill_path"]) as f:
                                current_skill = f.read()
                            current_score = float(tail_saved.get("current_score", current_score))
                            best_score = float(tail_saved.get("best_score", best_score))
                            best_step = int(tail_saved.get("best_step", best_step))
                            current_origin = tail_saved.get("current_origin", current_origin)
                            best_origin = tail_saved.get("best_origin", best_origin)
                            restored_best_path = tail_saved.get("best_skill_path")
                            if (
                                tail_saved.get("action") == "accept_new_best"
                                and restored_best_path
                                and os.path.exists(restored_best_path)
                            ):
                                with open(restored_best_path) as f:
                                    best_skill = f.read()
                            elif tail_saved.get("action") == "accept_new_best":
                                best_skill = current_skill
                    except Exception:
                        pass
                else:
                    all_tail_rows = _load_jsonl(
                        os.path.join(out_root, "type_guided_tail_bank", "tail_records_all.jsonl"),
                    )
                    min_tail_epoch = max(1, epoch - type_guided_tail_window_epochs + 1)
                    tail_rows = [
                        row for row in all_tail_rows
                        if min_tail_epoch <= int(row.get("epoch") or 0) <= epoch
                    ]
                    if not tail_rows:
                        tail_rows = _load_jsonl(os.path.join(tail_dir, "tail_records.jsonl"))
                    grouped_tail: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
                    for row in tail_rows:
                        key = (
                            str(row.get("question_type") or "other"),
                            str(row.get("revision_type") or "other"),
                            str(row.get("repair_signature") or "").strip().lower(),
                        )
                        grouped_tail[key].append(row)
                    selected_rows: list[dict] = []
                    selected_groups: list[dict] = []
                    for key, rows in sorted(grouped_tail.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                        record_ids = {str(row.get("tail_id") or "") for row in rows if row.get("tail_id")}
                        steps_seen = {int(row.get("step") or 0) for row in rows}
                        if len(record_ids) < type_guided_tail_min_support:
                            continue
                        if type_guided_tail_require_cross_step and len(steps_seen) < 2:
                            continue
                        selected_groups.append({
                            "question_type": key[0],
                            "revision_type": key[1],
                            "repair_signature": key[2],
                            "n_records": len(record_ids),
                            "steps": sorted(steps_seen),
                        })
                        selected_rows.extend(rows)
                    tail_records = _tail_records_to_patch_records(
                        selected_rows,
                        max_records=type_guided_tail_max_records,
                    )
                    tail_result = {
                        "epoch": epoch,
                        "enabled": True,
                        "window_epochs": type_guided_tail_window_epochs,
                        "min_epoch": min_tail_epoch,
                        "n_tail_rows": len(tail_rows),
                        "n_selected_groups": len(selected_groups),
                        "n_tail_records": len(tail_records),
                        "selected_groups": selected_groups,
                        "action": "skip_no_tail_records",
                    }
                    with open(os.path.join(tail_dir, "selected_tail_records.json"), "w") as f:
                        json.dump(tail_records, f, ensure_ascii=False, indent=2)
                    if tail_records:
                        print(
                            f"\n  [TYPE-GUIDED TAIL epoch {epoch}] "
                            f"records={len(tail_records)} groups={len(selected_groups)}"
                        )
                        tail_patch, tail_artifact = merge_type_guided_v2_records(
                            skill_content=current_skill,
                            patch_records=tail_records,
                            min_support=type_guided_tail_min_support,
                            max_leaf_groups=type_guided_tail_max_leaf_groups,
                            tree_depth=type_guided_tree_depth,
                            # Tail updates currently have one joint gate rather than
                            # branch-wise recursive validation, so keep their legacy
                            # executable-root construction.
                            tree_builder="fixed",
                            top_mode="real_root",
                            clustering_enabled=type_guided_clustering,
                            cluster_target_size=type_guided_cluster_target_size,
                            cluster_max_size=type_guided_cluster_max_size,
                            leaf_merge_workers=type_guided_leaf_merge_workers,
                            mid_merge_workers=type_guided_mid_merge_workers,
                            cache_dir=type_guided_cache_dir,
                            optimizer_model=str(cfg.get("optimizer_model", "")),
                            verbose=True,
                        )
                        with open(os.path.join(tail_dir, "tail_merge_artifact.json"), "w") as f:
                            json.dump(tail_artifact, f, ensure_ascii=False, indent=2)
                        with open(os.path.join(tail_dir, "tail_patch.json"), "w") as f:
                            json.dump(tail_patch, f, ensure_ascii=False, indent=2)
                        tail_candidate, tail_apply_report = apply_patch_with_report(
                            current_skill, tail_patch,
                        )
                        with open(os.path.join(tail_dir, "candidate_skill.md"), "w") as f:
                            f.write(tail_candidate)
                        with open(os.path.join(tail_dir, "edit_apply_report.json"), "w") as f:
                            json.dump(tail_apply_report, f, ensure_ascii=False, indent=2)
                        tail_hash = skill_hash(tail_candidate)
                        if get_payload_items(tail_patch, update_mode) and tail_hash != skill_hash(current_skill):
                            if tail_hash in sel_cache:
                                tail_hard, tail_soft = sel_cache[tail_hash]
                                tail_eval_dir = ""
                            else:
                                sel_env, sel_n = _build_eval_env(
                                    split="valid_seen",
                                    env_num=cfg["sel_env_num"],
                                    seed=seed,
                                )
                                print(f"    [type-guided tail gate] selection items={sel_n}")
                                tail_eval_dir = os.path.join(tail_dir, "selection_eval")
                                tail_eval_results = adapter.rollout(
                                    sel_env, tail_candidate, tail_eval_dir,
                                )
                                tail_hard, tail_soft = compute_score(tail_eval_results)
                                sel_cache[tail_hash] = (tail_hard, tail_soft)
                                _save_selection_result_cache(
                                    out_root, tail_hash, tail_eval_results,
                                )
                            tail_gate = evaluate_gate(
                                candidate_skill=tail_candidate,
                                cand_hard=tail_hard,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_skill=best_skill,
                                best_score=best_score,
                                best_step=best_step,
                                global_step=global_step,
                                cand_soft=tail_soft,
                                metric=gate_metric,
                                mixed_weight=gate_mixed_weight,
                            )
                            tail_gate_score = select_gate_score(
                                tail_hard, tail_soft, gate_metric, gate_mixed_weight,
                            )
                            prev_current = current_score
                            prev_best = best_score
                            current_skill = tail_gate.current_skill
                            current_score = tail_gate.current_score
                            best_skill = tail_gate.best_skill
                            best_score = tail_gate.best_score
                            best_step = tail_gate.best_step
                            if tail_gate.action in {"accept", "accept_new_best"}:
                                current_origin = f"type_guided_tail_epoch_{epoch:02d}"
                            if tail_gate.action == "accept_new_best":
                                best_origin = current_origin
                            tail_result.update({
                                "action": tail_gate.action,
                                "selection_hard": tail_hard,
                                "selection_soft": tail_soft,
                                "gate_score": tail_gate_score,
                                "current_before": prev_current,
                                "best_before": prev_best,
                                "eval_dir": tail_eval_dir,
                                "candidate_hash": tail_hash,
                            })
                            if tail_gate.action in {"accept", "accept_new_best"}:
                                accepted_path = os.path.join(tail_dir, "accepted_skill.md")
                                with open(accepted_path, "w") as f:
                                    f.write(current_skill)
                                tail_result["accepted_skill_path"] = accepted_path
                                if tail_gate.action == "accept_new_best":
                                    best_tail_path = os.path.join(tail_dir, "best_skill.md")
                                    with open(best_tail_path, "w") as f:
                                        f.write(best_skill)
                                    tail_result["best_skill_path"] = best_tail_path
                                print(
                                    f"    [type-guided tail] ACCEPT "
                                    f"gate={tail_gate_score:.4f} current={current_score:.4f}"
                                )
                            else:
                                print(
                                    f"    [type-guided tail] REJECT "
                                    f"gate={tail_gate_score:.4f} <= current={current_score:.4f}"
                                )
                        else:
                            tail_result["action"] = "skip_empty_or_unchanged_patch"
                    else:
                        print(
                            f"\n  [TYPE-GUIDED TAIL epoch {epoch}] "
                            f"skipped — no repeated tail mechanism"
                        )
                    tail_result.update({
                        "current_score": current_score,
                        "best_score": best_score,
                        "best_step": best_step,
                        "current_origin": current_origin,
                        "best_origin": best_origin,
                    })
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    _persist_runtime_state(global_step)
                    with open(tail_done_path, "w") as f:
                        json.dump(tail_result, f, ensure_ascii=False, indent=2)


        # ── Save best skill ──────────────────────────────────────────────
        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
            f.write(best_skill)
        _persist_runtime_state(global_step)
        print(
            f"\n  [done] best skill from step {best_step}, "
            f"score={best_score:.4f}"
        )

        # ── Final test evaluation (valid_unseen) ─────────────────────────
        baseline_test_hard = None
        baseline_test_soft = None
        test_hard = None
        test_soft = None
        final_test_hard = None
        final_test_soft = None
        final_selection_hard = None
        final_selection_soft = None

        if cfg["eval_test"]:
            task_types = adapter.get_task_types()

            # ── Final skill validation (valid_seen) + best promotion ─────
            # Validate the final current skill as well as the incumbent best.
            # When both are identical, reuse the known selection score.
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    final_selection_hard, final_selection_soft = best_score, None
                    print(
                        "\n  [final skill == best skill] "
                        f"final_selection_hard={best_score:.4f} (reused)"
                    )
                else:
                    fval_env, fval_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    fval_dir = os.path.join(out_root, "final_selection_eval")
                    fval_results = adapter.rollout(fval_env, current_skill, fval_dir)
                    final_selection_hard, final_selection_soft = compute_score(fval_results)
                    _save_selection_result_cache(
                        out_root, skill_hash(current_skill), fval_results,
                    )
                    final_gate_score = select_gate_score(
                        final_selection_hard, final_selection_soft,
                        gate_metric, gate_mixed_weight,
                    )
                    print(
                        f"\n  [final skill val] items={fval_n} "
                        f"final_selection_hard={final_selection_hard:.4f} "
                        f"gate={final_gate_score:.4f} "
                        f"(best={best_score:.4f})"
                    )
                    if final_gate_score > best_score:
                        # Promote a validation-better final skill before test.
                        print(
                            f"  [promote] final {final_gate_score:.4f} > "
                            f"best {best_score:.4f} → final becomes new best "
                            f"(step {global_step}, origin {current_origin})"
                        )
                        best_skill = current_skill
                        best_score = final_gate_score
                        best_step = global_step
                        best_origin = current_origin
                        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                            f.write(best_skill)
                        _persist_runtime_state(global_step)
            except Exception as _e:  # noqa: BLE001
                final_selection_hard = None
                final_selection_soft = None
                print(f"\n  [final skill val FAILED: {_e!r}]")

            # Baseline: S_0 on test set (valid_unseen)
            print(f"\n{'='*60}")
            print("  BASELINE TEST — evaluate initial skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env, test_n = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n}")
            baseline_test_dir = os.path.join(out_root, "test_eval_baseline")
            os.makedirs(baseline_test_dir, exist_ok=True)
            baseline_test_results = adapter.rollout(test_env, skill_init, baseline_test_dir)
            baseline_test_hard, baseline_test_soft = compute_score(baseline_test_results)
            baseline_buckets = _compute_task_type_buckets(baseline_test_results, task_types)
            print("\n  === Baseline Test Results (S_0) ===")
            for task_type in task_types + ["overall"]:
                b = baseline_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(baseline_test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in baseline_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Best skill on test set
            print(f"\n{'='*60}")
            print("  BEST SKILL TEST — evaluate best skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env2, test_n2 = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n2}")
            test_dir = os.path.join(out_root, "test_eval")
            os.makedirs(test_dir, exist_ok=True)
            test_results = adapter.rollout(test_env2, best_skill, test_dir)
            test_hard, test_soft = compute_score(test_results)
            best_buckets = _compute_task_type_buckets(test_results, task_types)
            print("\n  === Best Skill Test Results ===")
            for task_type in task_types + ["overall"]:
                b = best_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in best_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Final skill (last skill in trajectory) on test set.
            # Distinct from best_skill: with use_gate=False every candidate is
            # force-accepted so the final skill is whatever the last step
            # produced; with use_gate=True it is the last accepted skill, which
            # may differ from the best-on-val skill. We always evaluate it so
            # every run reports baseline / best-on-val / final on test.
            # Guarded so a failure here never prevents summary.json from being
            # written (the orchestrator's post-hoc safety net fills it in).
            try:
                if skill_hash(current_skill) == skill_hash(best_skill):
                    # Final == best: reuse results, skip a redundant rollout.
                    final_test_hard, final_test_soft = test_hard, test_soft
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in best_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
                    print(
                        "\n  [final skill == best skill] "
                        f"final_test_hard={final_test_hard:.4f} (reused)"
                    )
                else:
                    print(f"\n{'='*60}")
                    print("  FINAL SKILL TEST — evaluate last skill on Test set (valid_unseen)")
                    print(f"{'='*60}")
                    test_env3, test_n3 = _build_eval_env(
                        split="valid_unseen",
                        env_num=cfg["test_env_num"],
                        seed=seed,
                    )
                    print(f"  Test items: {test_n3}")
                    final_test_dir = os.path.join(out_root, "test_eval_final")
                    os.makedirs(final_test_dir, exist_ok=True)
                    final_test_results = adapter.rollout(test_env3, current_skill, final_test_dir)
                    final_test_hard, final_test_soft = compute_score(final_test_results)
                    final_buckets = _compute_task_type_buckets(final_test_results, task_types)
                    print("\n  === Final Skill Test Results ===")
                    for task_type in task_types + ["overall"]:
                        b = final_buckets.get(task_type, {"total": 0, "hard": 0})
                        t = max(b["total"], 1)
                        print(
                            f"    {task_type:<40s}: "
                            f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                        )
                    with open(os.path.join(final_test_dir, "summary.json"), "w") as f:
                        json.dump(
                            {
                                k: {
                                    "total": b["total"],
                                    "hard_acc": b["hard"] / max(b["total"], 1),
                                }
                                for k, b in final_buckets.items()
                            },
                            f, indent=2, ensure_ascii=False,
                        )
            except Exception as _e:  # noqa: BLE001
                final_test_hard = None
                final_test_soft = None
                print(f"\n  [final skill test FAILED: {_e!r}] "
                      "— will be filled by post-hoc eval")

            # Comparison
            delta_hard = (test_hard or 0) - (baseline_test_hard or 0)
            print(f"\n  === Improvement vs baseline (init S_0) ===")
            print(
                f"    [2] best-on-val hard: {baseline_test_hard:.4f} -> {test_hard:.4f}  "
                f"(delta={delta_hard:+.4f})"
            )
            if final_test_hard is not None:
                final_delta_hard = (final_test_hard or 0) - (baseline_test_hard or 0)
                print(
                    f"    [3] final/last  hard: {baseline_test_hard:.4f} -> {final_test_hard:.4f}  "
                    f"(delta={final_delta_hard:+.4f})"
                )

        # ── Global summary ───────────────────────────────────────────────
        total_wall = time.time() - t_loop_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(1 for h in history if h.get("action") == "reject")
        n_skip = sum(1 for h in history if h.get("action") == "skip_no_patches")

        token_summary = get_token_summary()

        # Epoch-level statistics
        epoch_stats = []
        for e in range(1, num_epochs + 1):
            epoch_records = [h for h in history if h.get("epoch") == e]
            if epoch_records:
                epoch_stats.append({
                    "epoch": e,
                    "steps": [h["step"] for h in epoch_records],
                    "accepts": sum(1 for h in epoch_records if "accept" in h.get("action", "")),
                    "rejects": sum(1 for h in epoch_records if h.get("action") == "reject"),
                    "skips": sum(1 for h in epoch_records if h.get("action") == "skip_no_patches"),
                    "best_score_at_epoch_end": epoch_records[-1].get("best_score", 0.0),
                    "current_score_at_epoch_end": epoch_records[-1].get("current_score", 0.0),
                })

        summary = {
            "version": "skillopt-0.1.0",
            "config": _redact_cfg(cfg),
            "baseline_selection_hard": sel_cache.get(
                skill_hash(skill_init), (None, None),
            )[0],
            "best_selection_hard": best_score,
            "final_selection_hard": final_selection_hard,
            "final_selection_soft": final_selection_soft,
            "best_step": best_step,
            "current_origin": current_origin,
            "best_origin": best_origin,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "epoch_stats": epoch_stats,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "final_test_hard": final_test_hard,
            "final_test_soft": final_test_soft,
            "test_delta_hard": (
                (test_hard or 0) - (baseline_test_hard or 0)
                if test_hard is not None
                else None
            ),
            "final_test_delta_hard": (
                (final_test_hard or 0) - (baseline_test_hard or 0)
                if final_test_hard is not None
                else None
            ),
            "total_wall_time_s": round(total_wall, 1),
            "token_summary": token_summary,
        }
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("  Final Summary")
        print(f"{'='*60}")
        print(
            f"  steps={len(history)} accept={n_accept} "
            f"reject={n_reject} skip={n_skip}"
        )
        print(f"  best_score={best_score:.4f} (step {best_step})  wall={total_wall:.0f}s")
        if epoch_stats:
            for es in epoch_stats:
                print(
                    f"    epoch {es['epoch']}: accept={es['accepts']} reject={es['rejects']} "
                    f"best={es['best_score_at_epoch_end']:.4f}"
                )
        if baseline_test_hard is not None:
            print("\n  === TEST scores (3 skills, split=valid_unseen) ===")
            print(
                f"    [1] init/baseline (S_0)          : "
                f"test_hard={baseline_test_hard:.4f}"
            )
        if test_hard is not None:
            print(
                f"    [2] best-on-val (step {best_step})".ljust(37)
                + f": test_hard={test_hard:.4f} test_soft={test_soft:.4f}"
            )
        if final_test_hard is not None:
            print(
                f"    [3] final/last skill             : "
                f"test_hard={final_test_hard:.4f} test_soft={final_test_soft:.4f}"
            )
        if token_summary.get("_total"):
            t = token_summary["_total"]
            print(
                f"  total tokens: {t['total_tokens']:,} "
                f"(prompt={t['prompt_tokens']:,} "
                f"completion={t['completion_tokens']:,} "
                f"calls={t['calls']})"
            )

        return summary
