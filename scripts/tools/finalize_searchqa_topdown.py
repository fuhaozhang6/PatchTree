#!/usr/bin/env python3
"""Finalize the SearchQA top-down candidate from phase-1 full-val results.

This tool is deliberately offline: it never configures or calls a target model.
It reconstructs one shared validation subset from the complete result IDs,
selects direct Root children that improve the parent on that subset, and builds
the deterministic reconciled candidate used by phase 2.

Example
-------
python scripts/tools/finalize_searchqa_topdown.py \
  --merge-artifact /path/to/type_guided_v2_merge_artifact.json \
  --parent-skill /path/to/skill_v0005.md \
  --root-skill /path/to/candidate_skill.md \
  --parent-results /path/to/parent_val/results.jsonl \
  --root-results /path/to/root_val/results.jsonl \
  --child-result M0001=/path/to/M0001_val/results.jsonl \
  --child-result M0002=/path/to/M0002_val/results.jsonl \
  --child-result M0003=/path/to/M0003_val/results.jsonl \
  --base-seed 42 \
  --global-step 6 \
  --out-dir /path/to/topdown_finalize
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from skillopt.engine.trainer import _reconcile_fallback_edits
from skillopt.evaluation.gate import select_gate_score
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.utils.scoring import compute_score, skill_hash


REPORT_SCHEMA = "searchqa_topdown_finalize_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a result JSONL, rejecting malformed rows and duplicate IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            sample_id = row.get("id")
            if sample_id is None:
                sample_id = row.get("source_id")
            if sample_id is None:
                raise ValueError(f"{path}:{lineno}: missing id/source_id")
            sample_id = str(sample_id)
            if sample_id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate result id {sample_id!r}")
            seen.add(sample_id)
            normalized = dict(row)
            normalized["id"] = sample_id
            rows.append(normalized)
    if not rows:
        raise ValueError(f"results file is empty: {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fallback_subset_seed(base_seed: int, global_step: int) -> int:
    """Use the exact fallback seed formula from PatchTreeTrainer."""
    return int(base_seed) + int(global_step) * 1_000_003 + 97


def select_subset_ids(
    full_val_ids: Iterable[str],
    *,
    base_seed: int,
    global_step: int,
    subset_size: int,
    ordered_ids: Iterable[str] | None = None,
) -> tuple[list[str], int]:
    """Select a reproducible shared subset from complete validation IDs.

    SearchQA writes concurrent results in completion order, so JSONL line order
    is not a stable dataset order.  Sorting IDs provides a canonical full-val
    order before applying the trainer's ``random.Random(seed).shuffle`` rule.
    """
    full_ids = {str(sample_id) for sample_id in full_val_ids}
    ids = (
        [str(sample_id) for sample_id in ordered_ids]
        if ordered_ids is not None
        else sorted(full_ids)
    )
    if len(ids) != len(set(ids)):
        raise ValueError("full validation IDs must be unique")
    if set(ids) != full_ids:
        raise ValueError("ordered validation IDs do not match full result IDs")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size > len(ids):
        raise ValueError(
            f"subset_size={subset_size} exceeds full validation size={len(ids)}"
        )
    seed = fallback_subset_seed(base_seed, global_step)
    random.Random(seed).shuffle(ids)
    return ids[:subset_size], seed


def rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def slice_rows(
    by_id: dict[str, dict[str, Any]],
    sample_ids: list[str],
    *,
    label: str,
) -> list[dict[str, Any]]:
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{label} is missing {len(missing)} subset IDs: {preview}")
    return [by_id[sample_id] for sample_id in sample_ids]


def score_summary(
    rows: list[dict[str, Any]],
    *,
    mixed_weight: float,
) -> dict[str, float | int]:
    hard, soft = compute_score(rows)
    return {
        "n": len(rows),
        "hard": hard,
        "soft": soft,
        "mixed": select_gate_score(hard, soft, "mixed", mixed_weight),
        "hard_correct": sum(float(row.get("hard", 0) or 0) > 0 for row in rows),
    }


def repaired_ids(
    parent_by_id: dict[str, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    sample_ids: list[str],
) -> set[str]:
    return {
        sample_id
        for sample_id in sample_ids
        if float(parent_by_id[sample_id].get("hard", 0) or 0) <= 0
        and float(candidate_by_id[sample_id].get("hard", 0) or 0) > 0
    }


def regressed_ids(
    parent_by_id: dict[str, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    sample_ids: list[str],
) -> set[str]:
    return {
        sample_id
        for sample_id in sample_ids
        if float(parent_by_id[sample_id].get("hard", 0) or 0) > 0
        and float(candidate_by_id[sample_id].get("hard", 0) or 0) <= 0
    }


def child_id(child: dict[str, Any], index: int) -> str:
    value = child.get("mid_id") or child.get("leaf_id") or f"C{index:04d}"
    return str(value)


def parse_child_result_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"invalid --child-result {value!r}; expected CHILD_ID=/path/results.jsonl"
            )
        identifier, raw_path = value.split("=", 1)
        identifier = identifier.strip()
        if not identifier or not raw_path.strip():
            raise ValueError(
                f"invalid --child-result {value!r}; child ID and path are required"
            )
        if identifier in result:
            raise ValueError(f"duplicate --child-result for {identifier!r}")
        result[identifier] = Path(raw_path).expanduser().resolve()
    return result


def _input_manifest(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def finalize_topdown(
    *,
    merge_artifact_path: Path,
    parent_skill_path: Path,
    root_skill_path: Path,
    parent_results_path: Path,
    root_results_path: Path,
    child_result_paths: dict[str, Path],
    out_dir: Path,
    base_seed: int = 42,
    global_step: int = 6,
    subset_size: int = 40,
    mixed_weight: float = 0.5,
    val_protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic top-down combination and its audit report."""
    if not 0.0 <= mixed_weight <= 1.0:
        raise ValueError("mixed_weight must be in [0, 1]")

    merge_artifact_path = merge_artifact_path.resolve()
    parent_skill_path = parent_skill_path.resolve()
    root_skill_path = root_skill_path.resolve()
    parent_results_path = parent_results_path.resolve()
    root_results_path = root_results_path.resolve()
    out_dir = out_dir.resolve()

    artifact = json.loads(merge_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("merge artifact must be a JSON object")
    level = str(artifact.get("root_children_level") or "")
    if level != "mid":
        raise ValueError(
            f"top-down evidence requires direct Mid children, got level={level!r}"
        )
    raw_children = artifact.get("root_child_patches")
    if not isinstance(raw_children, list) or not raw_children:
        raise ValueError("merge artifact has no root_child_patches")
    children = [child for child in raw_children if isinstance(child, dict)]
    if len(children) != len(raw_children):
        raise ValueError("every root_child_patches entry must be a JSON object")
    child_by_id: dict[str, dict[str, Any]] = {}
    for index, child in enumerate(children, 1):
        identifier = child_id(child, index)
        if identifier in child_by_id:
            raise ValueError(f"duplicate direct child ID in artifact: {identifier!r}")
        child_by_id[identifier] = child

    expected_ids = set(child_by_id)
    supplied_ids = set(child_result_paths)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        extra = sorted(supplied_ids - expected_ids)
        raise ValueError(
            f"child result mapping does not match artifact; missing={missing}, extra={extra}"
        )

    parent_skill = parent_skill_path.read_text(encoding="utf-8")
    root_skill = root_skill_path.read_text(encoding="utf-8")
    parent_rows = read_jsonl(parent_results_path)
    root_rows = read_jsonl(root_results_path)
    child_rows_all = {
        identifier: read_jsonl(path)
        for identifier, path in child_result_paths.items()
    }
    parent_map = rows_by_id(parent_rows)
    full_ids = set(parent_map)
    result_sets = {"root": set(rows_by_id(root_rows))}
    result_sets.update({
        f"child:{identifier}": set(rows_by_id(rows))
        for identifier, rows in child_rows_all.items()
    })
    mismatches = {
        label: {
            "missing": sorted(full_ids - ids),
            "extra": sorted(ids - full_ids),
        }
        for label, ids in result_sets.items()
        if ids != full_ids
    }
    if mismatches:
        compact = {
            label: {
                "missing_count": len(value["missing"]),
                "extra_count": len(value["extra"]),
                "missing_preview": value["missing"][:5],
                "extra_preview": value["extra"][:5],
            }
            for label, value in mismatches.items()
        }
        raise ValueError(f"full-val result ID sets differ: {compact}")

    ordered_val_ids = None
    if val_protocol_path is not None:
        val_protocol_path = val_protocol_path.resolve()
        protocol = json.loads(val_protocol_path.read_text(encoding="utf-8"))
        ordered_val_ids = protocol.get("item_ids")
        if not isinstance(ordered_val_ids, list):
            raise ValueError(f"{val_protocol_path} has no item_ids list")
    subset_ids, sample_seed = select_subset_ids(
        full_ids,
        base_seed=base_seed,
        global_step=global_step,
        subset_size=subset_size,
        ordered_ids=ordered_val_ids,
    )
    root_map = rows_by_id(root_rows)
    parent_subset = slice_rows(parent_map, subset_ids, label="parent")
    parent_subset_score = score_summary(parent_subset, mixed_weight=mixed_weight)
    parent_full_score = score_summary(parent_rows, mixed_weight=mixed_weight)
    root_subset_score = score_summary(
        slice_rows(root_map, subset_ids, label="root"),
        mixed_weight=mixed_weight,
    )
    root_full_score = score_summary(root_rows, mixed_weight=mixed_weight)

    child_reports: list[dict[str, Any]] = []
    kept_children: list[dict[str, Any]] = []
    kept_child_rows: list[dict[str, Any]] = []
    repairs_by_kept_child: dict[str, set[str]] = {}
    for index, child in enumerate(children, 1):
        identifier = child_id(child, index)
        result_path = child_result_paths[identifier]
        rows = child_rows_all[identifier]
        result_map = rows_by_id(rows)
        subset_rows = slice_rows(result_map, subset_ids, label=f"child {identifier}")
        subset_score = score_summary(subset_rows, mixed_weight=mixed_weight)
        full_score = score_summary(rows, mixed_weight=mixed_weight)
        improvement = (
            float(subset_score["mixed"]) - float(parent_subset_score["mixed"])
        )
        kept = improvement > 0.0
        candidate_skill, apply_report = apply_patch_with_report(parent_skill, child)
        repaired = repaired_ids(parent_map, result_map, subset_ids)
        regressed = regressed_ids(parent_map, result_map, subset_ids)
        row = {
            "child_id": identifier,
            "child_level": level,
            "mid_id": str(child.get("mid_id") or ""),
            "leaf_ids": child.get("leaf_ids") or [],
            "support_count": int(child.get("support_count", 0) or 0),
            "result_input": _input_manifest(result_path),
            "candidate_skill_hash": skill_hash(candidate_skill),
            "apply_report": apply_report,
            "full_val": full_score,
            "subset": subset_score,
            "parent_subset_mixed": parent_subset_score["mixed"],
            "mixed_improvement": improvement,
            "kept": kept,
            "keep_rule": "mixed_improvement > 0",
            "repaired_ids": sorted(repaired),
            "regressed_ids": sorted(regressed),
            "net_hard_repairs": len(repaired) - len(regressed),
        }
        child_reports.append(row)
        if kept:
            kept_children.append(child)
            kept_child_rows.append({
                "child_id": identifier,
                "child_level": level,
                "leaf_ids": child.get("leaf_ids") or [],
                "support_count": child.get("support_count", 0),
                "hard": subset_score["hard"],
                "soft": subset_score["soft"],
                "gate_score": subset_score["mixed"],
                "improvement": improvement,
            })
            repairs_by_kept_child[identifier] = repaired

    kept_ids = [str(row["child_id"]) for row in kept_child_rows]
    unique_repairs: dict[str, list[str]] = {}
    for identifier in kept_ids:
        other_repairs: set[str] = set()
        for other_id, repairs in repairs_by_kept_child.items():
            if other_id != identifier:
                other_repairs.update(repairs)
        unique_repairs[identifier] = sorted(
            repairs_by_kept_child[identifier] - other_repairs
        )
    pairwise: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(kept_ids):
        for right_id in kept_ids[left_index + 1:]:
            left = repairs_by_kept_child[left_id]
            right = repairs_by_kept_child[right_id]
            pairwise.append({
                "left_child_id": left_id,
                "right_child_id": right_id,
                "shared_repair_ids": sorted(left & right),
                "left_only_repair_ids": sorted(left - right),
                "right_only_repair_ids": sorted(right - left),
            })

    if kept_children:
        combined_edits, reconcile_report = _reconcile_fallback_edits(
            child_patches=kept_children,
            child_rows=kept_child_rows,
            update_mode="patch",
            mode="deterministic",
            min_children=1,
        )
    else:
        combined_edits = []
        reconcile_report = {
            "mode": "deterministic",
            "status": "no_kept_children",
            "n_children": 0,
            "n_input_edits": 0,
            "n_output_edits": 0,
            "dropped_edits": [],
            "llm_used": False,
        }
    combination_patch = {
        "reasoning": (
            "SearchQA top-down offline finalization: combine direct Mid patches "
            "whose mixed score strictly improved over the parent on the shared "
            f"subset; deterministic reconcile kept {len(combined_edits)} edits."
        ),
        "edits": combined_edits,
    }
    combination_skill, combination_apply_report = apply_patch_with_report(
        parent_skill, combination_patch,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    combination_patch_path = out_dir / "topdown_combination_patch.json"
    combination_skill_path = out_dir / "topdown_combination_candidate.md"
    apply_report_path = out_dir / "topdown_combination_apply_report.json"
    report_path = out_dir / "topdown_finalize_report.json"
    phase2_manifest_path = out_dir / "phase2_combo_manifest.tsv"
    combination_patch_path.write_text(
        json.dumps(combination_patch, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    combination_skill_path.write_text(combination_skill, encoding="utf-8")
    apply_report_path.write_text(
        json.dumps(combination_apply_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with phase2_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["run_name", "skill_path"],
        )
        writer.writeheader()
        writer.writerow({
            "run_name": "td_combo",
            "skill_path": combination_skill_path.name,
        })

    all_kept_have_unique = bool(kept_ids) and all(
        unique_repairs[identifier] for identifier in kept_ids
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "offline_only": True,
        "inputs": {
            "merge_artifact": _input_manifest(merge_artifact_path),
            "parent_skill": _input_manifest(parent_skill_path),
            "root_skill": _input_manifest(root_skill_path),
            "parent_results": _input_manifest(parent_results_path),
            "root_results": _input_manifest(root_results_path),
        },
        "protocol": {
            "metric": "mixed",
            "mixed_weight": mixed_weight,
            "child_tau": 0.0,
            "child_keep_rule": "mixed_improvement > 0",
            "base_seed": base_seed,
            "global_step": global_step,
            "fallback_sample_seed_formula": (
                "base_seed + global_step * 1_000_003 + 97"
            ),
            "fallback_sample_seed": sample_seed,
            "subset_size": subset_size,
            "sampling": (
                "evaluator_item_order_then_python_random_shuffle"
                if val_protocol_path is not None
                else "sorted_full_val_ids_then_python_random_shuffle"
            ),
            "val_protocol": (
                _input_manifest(val_protocol_path)
                if val_protocol_path is not None else None
            ),
            "subset_ids": subset_ids,
            "reconcile": "deterministic_exact_dedup",
        },
        "tree": {
            "root_children_level": level,
            "n_direct_children": len(children),
            "direct_child_ids": list(child_by_id),
        },
        "skills": {
            "parent_hash": skill_hash(parent_skill),
            "root_hash": skill_hash(root_skill),
            "combination_hash": skill_hash(combination_skill),
        },
        "scores": {
            "parent_full_val": parent_full_score,
            "root_full_val": root_full_score,
            "root_full_val_mixed_improvement": (
                float(root_full_score["mixed"]) - float(parent_full_score["mixed"])
            ),
            "parent_subset": parent_subset_score,
            "root_subset": root_subset_score,
        },
        "children": child_reports,
        "kept_child_ids": kept_ids,
        "complementarity": {
            "repair_definition": "parent hard=0 and child hard=1 on shared subset",
            "unique_repair_ids_by_child": unique_repairs,
            "pairwise": pairwise,
            "at_least_two_kept_children": len(kept_ids) >= 2,
            "every_kept_child_has_unique_repair": all_kept_have_unique,
            "topdown_complementarity_pass": (
                len(kept_ids) >= 2 and all_kept_have_unique
            ),
        },
        "reconcile": reconcile_report,
        "combination": {
            "n_edits": len(combined_edits),
            "patch_path": str(combination_patch_path),
            "candidate_skill_path": str(combination_skill_path),
            "apply_report_path": str(apply_report_path),
            "phase2_val_manifest": str(phase2_manifest_path),
            "apply_report": combination_apply_report,
            "full_val_status": "pending_phase2",
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline SearchQA top-down child selection and reconcile"
    )
    parser.add_argument("--merge-artifact", type=Path, required=True)
    parser.add_argument("--parent-skill", type=Path, required=True)
    parser.add_argument("--root-skill", type=Path, required=True)
    parser.add_argument("--parent-results", type=Path, required=True)
    parser.add_argument("--root-results", type=Path, required=True)
    parser.add_argument(
        "--child-result",
        action="append",
        default=[],
        metavar="CHILD_ID=PATH",
        help="Full-val results.jsonl for one direct Root child; repeat per child",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--global-step", type=int, default=6)
    parser.add_argument("--subset-size", type=int, default=40)
    parser.add_argument("--mixed-weight", type=float, default=0.5)
    parser.add_argument(
        "--val-protocol",
        type=Path,
        help="Evaluator protocol.json providing canonical full-val item_ids order.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    child_result_paths = parse_child_result_specs(args.child_result)
    report = finalize_topdown(
        merge_artifact_path=args.merge_artifact,
        parent_skill_path=args.parent_skill,
        root_skill_path=args.root_skill,
        parent_results_path=args.parent_results,
        root_results_path=args.root_results,
        child_result_paths=child_result_paths,
        out_dir=args.out_dir,
        base_seed=args.base_seed,
        global_step=args.global_step,
        subset_size=args.subset_size,
        mixed_weight=args.mixed_weight,
        val_protocol_path=args.val_protocol,
    )
    scores = report["scores"]
    print(
        "Top-down offline finalization complete: "
        f"kept={len(report['kept_child_ids'])}/{report['tree']['n_direct_children']} "
        f"parent_full_mixed={scores['parent_full_val']['mixed']:.4f} "
        f"root_full_mixed={scores['root_full_val']['mixed']:.4f}"
    )
    print(f"Report: {args.out_dir.resolve() / 'topdown_finalize_report.json'}")


if __name__ == "__main__":
    main()
