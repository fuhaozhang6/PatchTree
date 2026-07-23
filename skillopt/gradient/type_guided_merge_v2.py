"""Per-question PatchRecord pipeline for Type-Guided Merge Tree V2.

V2 changes the evidence unit from minibatch analyst edits to one PatchRecord
per unstable/failed question:

    repeated rollouts -> PatchRecords -> typed leaves -> root patch

The module keeps the public patch schema compatible with the existing trainer
so update, validation, and leaf fallback can reuse the V1 path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skillopt.gradient.type_guided_merge import (
    build_patchtree,
)
from skillopt.model import chat_optimizer
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json, skill_hash


def _trajectory_text(value: Any) -> str:
    return "" if value is None else str(value)


def fmt_trajectory(conversation: list[dict]) -> str:
    """Format one rollout conversation for the PatchRecord analyst."""
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_trajectory_text(item)}")
        elif item.get("type") == "tool_call":
            lines.append(f"[action] {_trajectory_text(item.get('cmd'))}")
            lines.append(f"[obs]    {_trajectory_text(item.get('obs'))}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _trajectory_text(item.get("reasoning"))
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {_trajectory_text(item.get('action'))}")
            lines.append(f"[step {step} obs]    {_trajectory_text(item.get('env_feedback'))}")
        elif item.get("role") == "system":
            lines.append(f"[verification] {_trajectory_text(item.get('content'))}")
        else:
            lines.append(
                f"[{item.get('role', 'agent')}] {_trajectory_text(item.get('content'))}"
            )
    return "\n".join(lines)


def _safe_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:96] or "sample"


def _safe_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:64] or "other"


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _compact_result(result: dict) -> dict:
    keep_keys = [
        "id",
        "hard",
        "soft",
        "task_description",
        "instruction",
        "question",
        "query",
        "task_type",
        "instruction_type",
        "subtype",
        "fail_reason",
        "error",
        "answer",
        "prediction",
        "pred",
        "output",
        "final_answer",
        "evaluator_feedback",
        "reference_text",
    ]
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in result and result.get(key) not in (None, ""):
            value = result.get(key)
            text = str(value)
            compact[key] = text[:4000] if len(text) > 4000 else value
    return compact


def _read_trajectory(prediction_dir: str, sample_id: str) -> str:
    if not prediction_dir:
        return ""
    path = os.path.join(prediction_dir, str(sample_id), "conversation.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            conversation = json.load(f)
        if isinstance(conversation, list):
            return fmt_trajectory(conversation)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _group_repeated_rollouts(
    repeated_rollouts: list[dict],
    *,
    include_trajectories: bool = True,
) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for repeat in repeated_rollouts:
        repeat_id = int(repeat.get("repeat_id", 0) or 0)
        prediction_dir = str(repeat.get("prediction_dir") or "")
        for result in repeat.get("results", []) or []:
            if not isinstance(result, dict) or result.get("id") is None:
                continue
            sample_id = str(result.get("source_id") or result.get("original_id") or result.get("id"))
            prediction_id = str(result.get("_prediction_id") or result.get("id"))
            row = grouped.setdefault(sample_id, {"sample_id": sample_id, "rollouts": []})
            entry = {
                "repeat_id": repeat_id,
                "result": _compact_result(result),
            }
            if include_trajectories:
                traj = _read_trajectory(prediction_dir, prediction_id)
                if traj:
                    entry["trajectory"] = traj
            row["rollouts"].append(entry)
    for row in grouped.values():
        row["rollouts"].sort(key=lambda item: int(item.get("repeat_id", 0) or 0))
    return grouped


def _success_rate(sample_group: dict) -> float:
    rollouts = sample_group.get("rollouts", []) or []
    if not rollouts:
        return 0.0
    hard = 0
    for rollout in rollouts:
        result = rollout.get("result", {}) if isinstance(rollout, dict) else {}
        try:
            hard += 1 if float(result.get("hard", 0) or 0) > 0 else 0
        except (TypeError, ValueError):
            hard += 1 if result.get("hard") else 0
    return hard / max(len(rollouts), 1)


def _cache_path(
    *,
    cache_dir: str,
    sample_group: dict,
    skill_content: str,
    prompt_version: str,
    optimizer_model: str,
) -> str:
    payload = {
        "sample_id": sample_group.get("sample_id"),
        "rollouts": sample_group.get("rollouts", []),
        "skill_hash": skill_hash(skill_content),
        "prompt_version": prompt_version,
        "optimizer_model": optimizer_model,
    }
    return os.path.join(cache_dir, f"patch_record_{_json_hash(payload)}.json")


def _validate_patch_record(
    result: dict | None,
) -> dict | None:
    if not isinstance(result, dict):
        return None
    if result.get("no_patch") is True:
        return {
            "no_patch": True,
            "reasoning": str(result.get("reasoning") or ""),
        }
    patch = result.get("patch")
    if not isinstance(patch, dict):
        return None
    op = str(patch.get("op") or "").strip()
    if op not in {"append", "insert_after", "replace", "delete"}:
        return None
    if op != "delete" and not str(patch.get("content") or "").strip():
        return None
    if op in {"insert_after", "replace", "delete"} and not str(patch.get("target") or "").strip():
        return None
    question_type = _safe_slug(result.get("question_type"))
    revision_type = _safe_slug(result.get("revision_type"))
    repair_signature = str(result.get("repair_signature") or "").strip()
    condition = str(result.get("condition") or result.get("applicability") or "").strip()
    boundary = str(result.get("boundary") or "").strip()
    if not repair_signature or not condition:
        return None
    record = {
        "question_type": question_type,
        "revision_type": revision_type,
        "repair_signature": repair_signature,
        "condition": condition,
        "boundary": boundary,
        "patch": dict(patch),
    }
    return record


def _call_patch_record_analyst(
    *,
    skill_content: str,
    sample_group: dict,
    q_i: float,
    status: str,
    optimizer_model: str,
    cache_dir: str,
    step_cache_dir: str,
    env_name: str = "",
) -> tuple[dict | None, dict]:
    sample_id = str(sample_group.get("sample_id"))
    prompt_version = f"type_guided_patch_record:v3_compact:{env_name or 'generic'}"
    cache_paths: list[str] = []
    if cache_dir:
        cache_paths.append(_cache_path(
            cache_dir=cache_dir,
            sample_group=sample_group,
            skill_content=skill_content,
            prompt_version=prompt_version,
            optimizer_model=optimizer_model,
        ))
    if step_cache_dir:
        cache_paths.append(os.path.join(step_cache_dir, f"{_safe_name(sample_id)}.json"))

    for path in cache_paths:
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    cached = json.load(f)
                record = _validate_patch_record(
                    cached.get("record"),
                )
                if record is not None:
                    return record, {
                        "sample_id": sample_id,
                        "status": "cache_hit",
                        "cache_hit": True,
                        "cache_path": path,
                    }
            except Exception:  # noqa: BLE001
                pass

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Sample Outcome\n"
        f"sample_id: {sample_id}\n"
        f"q_i: {q_i:.4f}\n"
        f"status: {status}\n\n"
        f"## Repeated Rollouts\n"
        f"{json.dumps(sample_group.get('rollouts', []), ensure_ascii=False, indent=2)}"
    )
    raw_result: dict | None = None
    error = ""
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_patch_record", env_name or None),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="type_guided_patch_record",
        )
        raw_result = extract_json(response)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    record = _validate_patch_record(
        raw_result,
    )
    report = {
        "sample_id": sample_id,
        "status": "ok" if record is not None else "invalid_or_no_json",
        "cache_hit": False,
    }
    if error:
        report["error"] = error
    if record and record.get("no_patch"):
        report["status"] = "no_patch"

    cache_payload = {
        "sample_id": sample_id,
        "record": record,
        "raw_result": raw_result,
        "report": report,
    }
    if record is not None:
        for path in cache_paths:
            if path:
                _atomic_write_json(path, cache_payload)
    return record, report


def generate_patch_records(
    *,
    skill_content: str,
    repeated_rollouts: list[dict],
    tau_succ: float = 1.0,
    max_patch_records: int = 24,
    workers: int = 16,
    optimizer_model: str = "",
    cache_dir: str = "",
    step_cache_dir: str = "",
    include_trajectories: bool = True,
    env_name: str = "",
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Generate one PatchRecord per unstable/failed sample when warranted."""
    t0 = time.time()
    grouped = _group_repeated_rollouts(
        repeated_rollouts,
        include_trajectories=include_trajectories,
    )
    candidates: list[tuple[str, dict, float, str]] = []
    stable_successes: list[dict] = []
    for sample_id, group in sorted(grouped.items()):
        q_i = _success_rate(group)
        if q_i >= tau_succ:
            stable_successes.append({
                "sample_id": sample_id,
                "q_i": q_i,
                "n_rollouts": len(group.get("rollouts", []) or []),
            })
            continue
        status = "failure" if q_i <= 0 else "unstable"
        candidates.append((sample_id, group, q_i, status))

    n_candidates_before_limit = len(candidates)
    if max_patch_records > 0 and len(candidates) > max_patch_records:
        # Prefer the most reliable failure evidence when analyst capacity is
        # bounded: consistent failures first, then progressively less severe
        # unstable samples.  More repeated observations break ties in favour
        # of better-supported empirical success rates.
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item[2]),
                -len(item[1].get("rollouts", []) or []),
                item[0],
            ),
        )[:max_patch_records]
    n_candidates_dropped_by_limit = n_candidates_before_limit - len(candidates)
    selected_candidate_ids = [sample_id for sample_id, _group, _q_i, _status in candidates]

    records: list[dict] = []
    reports: list[dict] = []
    no_patch_count = 0
    pending = list(candidates)
    if pending:
        with ThreadPoolExecutor(max_workers=max(int(workers or 1), 1)) as ex:
            futs = {
                ex.submit(
                    _call_patch_record_analyst,
                    skill_content=skill_content,
                    sample_group=group,
                    q_i=q_i,
                    status=status,
                    optimizer_model=optimizer_model,
                    cache_dir=cache_dir,
                    step_cache_dir=step_cache_dir,
                    env_name=env_name,
                ): (sample_id, q_i, status)
                for sample_id, group, q_i, status in pending
            }
            for fut in as_completed(futs):
                sample_id, _q_i, _status = futs[fut]
                record, report = fut.result()
                reports.append(report)
                if record and not record.get("no_patch"):
                    record["_source_order"] = sample_id
                    records.append(record)
                elif record and record.get("no_patch"):
                    no_patch_count += 1

    records.sort(key=lambda rec: str(rec.get("_source_order") or ""))
    for idx, record in enumerate(records, start=1):
        record.pop("_source_order", None)
        record["record_id"] = f"R{idx:04d}"

    if verbose:
        print(
            f"    [type-guided v2 records] samples={len(grouped)} "
            f"stable={len(stable_successes)} "
            f"candidates={len(candidates)}/{n_candidates_before_limit} "
            f"truncated={n_candidates_dropped_by_limit} "
            f"records={len(records)} no_patch={no_patch_count}"
        )

    artifact = {
        "version": "v2",
        "n_samples": len(grouped),
        "n_stable_success": len(stable_successes),
        "n_candidates": len(candidates),
        "n_candidates_before_limit": n_candidates_before_limit,
        "n_candidates_dropped_by_limit": n_candidates_dropped_by_limit,
        "selected_candidate_ids": selected_candidate_ids,
        "n_records": len(records),
        "n_no_patch": no_patch_count,
        "stable_successes": stable_successes,
        "analyst_reports": sorted(reports, key=lambda row: str(row.get("sample_id", ""))),
        "settings": {
            "tau_succ": tau_succ,
            "max_patch_records": max_patch_records,
            "workers": workers,
            "cache_dir": cache_dir,
            "include_trajectories": include_trajectories,
            "env_name": env_name,
        },
        "timing_s": round(time.time() - t0, 1),
    }
    return records, artifact


def _record_to_failure_patch(record: dict) -> dict:
    edit = dict(record.get("patch") or {})
    edit["question_type"] = str(record.get("question_type") or "other")
    edit["revision_type"] = str(record.get("revision_type") or "other")
    edit["source_type"] = "failure"
    edit["support_count"] = 1
    edit["record_ids"] = [str(record.get("record_id"))]
    edit["repair_signature"] = record.get("repair_signature", "")
    edit["condition"] = record.get("condition", "")
    edit["applicability"] = record.get("condition", "")
    edit["boundary"] = record.get("boundary", "")
    if record.get("cluster_id"):
        edit["cluster_id"] = record.get("cluster_id")
        edit["cluster_label"] = record.get("cluster_label", "")
        edit["cluster_question_type"] = record.get("cluster_question_type", "")
        edit["cluster_revision_type"] = record.get("cluster_revision_type", "")
        edit["cluster_source"] = record.get("cluster_source", "")
        edit["cluster_merge_rationale"] = record.get("cluster_merge_rationale", "")
        edit["cluster_boundary"] = record.get("cluster_boundary", "")
    return {
        "reasoning": record.get("repair_signature", ""),
        "edits": [edit],
    }


def _cluster_type_pair(record: dict) -> tuple[str, str]:
    return (
        str(record.get("question_type") or "other"),
        str(record.get("revision_type") or "other"),
    )


def _cluster_card(record: dict) -> dict:
    return {
        "record_id": str(record.get("record_id") or ""),
        "question_type": _cluster_type_pair(record)[0],
        "revision_type": _cluster_type_pair(record)[1],
        "repair_signature": str(record.get("repair_signature") or ""),
        "condition": str(record.get("condition") or "")[:600],
        "boundary": str(record.get("boundary") or "")[:600],
    }


def _chunk_records(records: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        return [records]
    return [records[i:i + size] for i in range(0, len(records), size)]


def _fallback_cluster_bucket(
    records: list[dict],
    *,
    prefix: str,
    max_cluster_size: int,
    source: str = "fallback_signature",
    rationale: str = "deterministic fallback by repair_signature",
) -> list[dict]:
    by_sig: dict[str, list[dict]] = {}
    for record in records:
        sig = _safe_slug(record.get("repair_signature") or "unspecified")
        by_sig.setdefault(sig, []).append(record)
    clusters: list[dict] = []
    counter = 1
    for sig, sig_records in sorted(by_sig.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for chunk in _chunk_records(sig_records, max_cluster_size):
            q_type, r_type = _cluster_type_pair(chunk[0])
            clusters.append({
                "cluster_id": f"{prefix}C{counter:03d}",
                "cluster_label": sig,
                "question_type": q_type,
                "revision_type": r_type,
                "repair_signature": sig if sig != "unspecified" else "",
                "record_ids": [str(row.get("record_id")) for row in chunk],
                "merge_rationale": rationale,
                "boundary": "",
                "source": source,
            })
            counter += 1
    return clusters


def _validate_cluster_plan(
    *,
    result: dict | None,
    records: list[dict],
    prefix: str,
    max_cluster_size: int,
) -> tuple[list[dict], list[str]]:
    record_by_id = {str(record.get("record_id") or ""): record for record in records}
    assigned: set[str] = set()
    used_cluster_ids: set[str] = set()
    clusters: list[dict] = []
    errors: list[str] = []

    def unique_cluster_id(raw_id: Any, fallback_id: str) -> str:
        base_cid = _safe_slug(raw_id or fallback_id)
        if not base_cid.startswith(_safe_slug(prefix)):
            base_cid = f"{_safe_slug(prefix)}_{base_cid}"
        cid = base_cid
        suffix = 2
        while cid in used_cluster_ids:
            cid = f"{base_cid}_{suffix}"
            suffix += 1
        used_cluster_ids.add(cid)
        return cid

    raw_clusters = result.get("clusters") if isinstance(result, dict) else None
    if not isinstance(raw_clusters, list):
        return _fallback_cluster_bucket(
            records, prefix=prefix, max_cluster_size=max_cluster_size,
        ), ["invalid_schema"]

    counter = 1
    for raw in raw_clusters:
        if not isinstance(raw, dict):
            errors.append("cluster_not_dict")
            continue
        raw_ids = raw.get("record_ids")
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            errors.append("record_ids_not_list")
            continue
        ids: list[str] = []
        for rid_raw in raw_ids:
            rid = str(rid_raw or "").strip()
            if not rid or rid not in record_by_id or rid in assigned:
                continue
            assigned.add(rid)
            ids.append(rid)
        if not ids:
            continue
        for chunk in _chunk_records(ids, max_cluster_size):
            first = record_by_id[chunk[0]]
            q_type, r_type = _cluster_type_pair(first)
            cid = unique_cluster_id(raw.get("cluster_id"), f"C{counter:03d}")
            clusters.append({
                "cluster_id": cid,
                "cluster_label": str(raw.get("cluster_label") or raw.get("repair_signature") or cid),
                "question_type": str(raw.get("question_type") or q_type),
                "revision_type": str(raw.get("revision_type") or r_type),
                "repair_signature": str(raw.get("repair_signature") or ""),
                "record_ids": chunk,
                "merge_rationale": str(raw.get("merge_rationale") or ""),
                "boundary": str(raw.get("boundary") or ""),
                "source": "llm",
            })
            counter += 1

    raw_singletons = result.get("singletons") if isinstance(result, dict) else None
    if raw_singletons is not None and not isinstance(raw_singletons, list):
        errors.append("singletons_not_list")
    if isinstance(raw_singletons, list):
        for raw in raw_singletons:
            if not isinstance(raw, dict):
                errors.append("singleton_not_dict")
                continue
            rid = str(raw.get("record_id") or "").strip()
            if not rid or rid not in record_by_id or rid in assigned:
                continue
            assigned.add(rid)
            record = record_by_id[rid]
            q_type, r_type = _cluster_type_pair(record)
            cid = unique_cluster_id(raw.get("cluster_id"), f"S{counter:03d}")
            clusters.append({
                "cluster_id": cid,
                "cluster_label": str(
                    raw.get("cluster_label")
                    or raw.get("repair_signature")
                    or record.get("repair_signature")
                    or cid
                ),
                "question_type": str(raw.get("question_type") or q_type),
                "revision_type": str(raw.get("revision_type") or r_type),
                "repair_signature": str(
                    raw.get("repair_signature") or record.get("repair_signature") or "",
                ),
                "record_ids": [rid],
                "merge_rationale": str(raw.get("reason") or "LLM marked this record as a singleton"),
                "boundary": str(raw.get("boundary") or record.get("boundary") or ""),
                "source": "llm_singleton",
            })
            counter += 1

    missing = [rid for rid in record_by_id if rid not in assigned]
    if missing:
        errors.append(f"unassigned={len(missing)}")
        missing_records = [record_by_id[rid] for rid in missing]
        clusters.extend(_fallback_cluster_bucket(
            missing_records,
            prefix=f"{prefix}M",
            max_cluster_size=max_cluster_size,
        ))
    return clusters, errors


def _call_cluster_planner(
    *,
    records: list[dict],
    target_cluster_size: int,
    max_cluster_size: int,
    cache_dir: str,
    skill_content: str,
    optimizer_model: str,
) -> tuple[list[dict], dict]:
    prefix = "G01_"
    cards = [_cluster_card(record) for record in records]
    cache_path = ""
    if cache_dir:
        cache_payload = {
            "prompt_version": "type_guided_cluster:v2.4_global",
            "cards": cards,
            "target_cluster_size": target_cluster_size,
            "max_cluster_size": max_cluster_size,
            "skill_hash": skill_hash(skill_content),
            "optimizer_model": optimizer_model,
        }
        cache_path = os.path.join(
            cache_dir, "cluster", f"{_json_hash(cache_payload)}.json",
        )
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                clusters = cached.get("clusters")
                report = cached.get("report")
                if isinstance(clusters, list) and isinstance(report, dict):
                    report["cache_hit"] = True
                    report["cache_path"] = cache_path
                    return clusters, report
            except Exception:  # noqa: BLE001
                pass

    payload = {
        "global_clustering": {
            "n_records": len(records),
            "target_cluster_size": target_cluster_size,
            "max_cluster_size": max_cluster_size,
        },
        "records": cards,
    }
    user = "## PatchRecords To Cluster\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw_result: dict | None = None
    status = "ok"
    error = ""
    try:
        response, _ = chat_optimizer(
            system=load_prompt("type_guided_cluster"),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="type_guided_cluster",
        )
        raw_result = extract_json(response)
    except Exception as exc:  # noqa: BLE001
        status = "fallback"
        error = repr(exc)

    clusters, errors = _validate_cluster_plan(
        result=raw_result,
        records=records,
        prefix=prefix,
        max_cluster_size=max_cluster_size,
    )
    if errors:
        status = "fallback" if status != "ok" else "partial_fallback"
    report = {
        "n_records": len(records),
        "n_clusters": len(clusters),
        "status": status,
        "errors": errors,
        "cache_hit": False,
    }
    if error:
        report["error"] = error
    if cache_path:
        _atomic_write_json(cache_path, {
            "clusters": clusters,
            "report": {**report, "cache_path": cache_path},
        })
        report["cache_path"] = cache_path
    return clusters, report


def cluster_patch_records(
    records: list[dict],
    *,
    enabled: bool = False,
    target_cluster_size: int = 6,
    max_cluster_size: int = 10,
    cache_dir: str = "",
    skill_content: str = "",
    optimizer_model: str = "",
) -> tuple[list[dict], dict]:
    if not records:
        return [], {"enabled": bool(enabled), "status": "empty", "clusters": []}
    target_cluster_size = max(int(target_cluster_size or 6), 1)
    max_cluster_size = max(int(max_cluster_size or target_cluster_size), target_cluster_size)
    if not enabled:
        clusters = _fallback_cluster_bucket(
            records,
            prefix="D_",
            max_cluster_size=max_cluster_size,
        )
        return _apply_clusters_to_records(records, clusters), {
            "enabled": False,
            "status": "default_signature",
            "clusters": clusters,
            "settings": {
                "target_cluster_size": target_cluster_size,
                "max_cluster_size": max_cluster_size,
            },
        }

    all_clusters, report = _call_cluster_planner(
        records=records,
        target_cluster_size=target_cluster_size,
        max_cluster_size=max_cluster_size,
        cache_dir=cache_dir,
        skill_content=skill_content,
        optimizer_model=optimizer_model,
    )
    all_clusters.sort(key=lambda row: (str(row.get("cluster_id", "")), str(row.get("cluster_label", ""))))
    clustered = _apply_clusters_to_records(records, all_clusters)
    return clustered, {
        "enabled": True,
        "status": report.get("status", "ok"),
        "clusters": all_clusters,
        "cluster_report": report,
        "bucket_reports": [report],
        "settings": {
            "target_cluster_size": target_cluster_size,
            "max_cluster_size": max_cluster_size,
            "planner": "global_llm",
        },
    }


def _apply_clusters_to_records(records: list[dict], clusters: list[dict]) -> list[dict]:
    by_id = {str(record.get("record_id") or ""): record for record in records}
    cluster_by_record: dict[str, dict] = {}
    for cluster in clusters:
        for rid in cluster.get("record_ids") or []:
            rid = str(rid)
            if rid in by_id and rid not in cluster_by_record:
                cluster_by_record[rid] = cluster
    out: list[dict] = []
    for record in records:
        rid = str(record.get("record_id") or "")
        cluster = cluster_by_record.get(rid)
        copied = dict(record)
        if cluster:
            copied["cluster_id"] = str(cluster.get("cluster_id") or "")
            copied["cluster_label"] = str(cluster.get("cluster_label") or "")
            copied["cluster_question_type"] = str(cluster.get("question_type") or "")
            copied["cluster_revision_type"] = str(cluster.get("revision_type") or "")
            copied["cluster_source"] = str(cluster.get("source") or "")
            copied["cluster_merge_rationale"] = str(cluster.get("merge_rationale") or "")
            copied["cluster_boundary"] = str(cluster.get("boundary") or "")
            if cluster.get("repair_signature") and not copied.get("repair_signature"):
                copied["repair_signature"] = str(cluster.get("repair_signature") or "")
        out.append(copied)
    return out


def merge_type_guided_v2_records(
    *,
    skill_content: str,
    patch_records: list[dict],
    min_support: int = 2,
    max_leaf_groups: int = 8,
    tree_depth: int = 2,
    tree_builder: str = "fixed",
    max_tree_depth: int = 4,
    merge_target_children: int = 3,
    merge_max_children: int = 4,
    top_mode: str = "auto",
    clustering_enabled: bool = False,
    cluster_target_size: int = 6,
    cluster_max_size: int = 10,
    leaf_merge_workers: int = 1,
    mid_merge_workers: int = 1,
    cache_dir: str = "",
    optimizer_model: str = "",
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Merge PatchRecords into typed leaves and a root candidate."""
    patch_records, clustering = cluster_patch_records(
        patch_records,
        enabled=clustering_enabled,
        target_cluster_size=cluster_target_size,
        max_cluster_size=cluster_max_size,
        cache_dir=cache_dir,
        skill_content=skill_content,
        optimizer_model=optimizer_model,
    )
    failure_patches = [_record_to_failure_patch(record) for record in patch_records]
    root_patch, artifact = build_patchtree(
        skill_content,
        failure_patches,
        min_support=min_support,
        max_leaf_groups=max_leaf_groups,
        allow_open_types=True,
        group_by_cluster=clustering_enabled,
        low_support_fallback=not clustering_enabled,
        tree_depth=tree_depth,
        tree_builder=tree_builder,
        max_tree_depth=max_tree_depth,
        merge_target_children=merge_target_children,
        merge_max_children=merge_max_children,
        top_mode=top_mode,
        leaf_merge_workers=leaf_merge_workers,
        mid_merge_workers=mid_merge_workers,
        verbose=verbose,
    )
    artifact["version"] = "v2"
    artifact["patch_records"] = patch_records
    artifact["clustering"] = clustering
    artifact["settings"] = {
        **artifact.get("settings", {}),
        "version": "v2",
        "tree_depth": int(artifact.get("tree_depth", max(int(tree_depth or 1), 1))),
        "tree_builder": str(tree_builder or "fixed"),
        "max_tree_depth": int(max_tree_depth),
        "merge_target_children": int(merge_target_children),
        "merge_max_children": int(merge_max_children),
        "top_mode": str(top_mode or "auto"),
        "clustering_enabled": bool(clustering_enabled),
        "cluster_target_size": int(cluster_target_size),
        "cluster_max_size": int(cluster_max_size),
        "leaf_merge_workers": int(leaf_merge_workers),
        "mid_merge_workers": int(mid_merge_workers),
        "cache_dir": cache_dir,
    }
    return root_patch, artifact
