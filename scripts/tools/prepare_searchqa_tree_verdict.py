#!/usr/bin/env python3
"""Prepare the fixed-evidence SearchQA PatchTree verdict candidates.

This tool never runs target-model rollouts and never regenerates PatchRecords.
It audits one completed SearchQA P6 run, selects structurally valid accepted and
rejected steps, then prepares:

* G0: the parent skill entering the accepted step;
* G1: a new flat Record -> Root candidate;
* G2: a new Leaf -> Root candidate built from the *saved* G3 leaves;
* G3: the saved Full Tree candidate;
* G4 phase 1: the parent, rejected Root, and saved direct-child candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from skillopt.gradient.type_guided_merge import _build_root_patch
from skillopt.gradient.type_guided_merge_v2 import merge_type_guided_v2_records
from skillopt.model import (
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
)
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.update_modes import get_payload_items
from skillopt.utils import skill_hash


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _clean_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ids_from_node(node: dict, keys: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for key in keys:
        found.update(_clean_ids(node.get(key)))
    for edit in get_payload_items(node, "patch"):
        if not isinstance(edit, dict):
            continue
        for key in keys:
            found.update(_clean_ids(edit.get(key)))
    return found


def _mid_groups(artifact: dict) -> list[dict]:
    groups = artifact.get("mid_groups")
    if isinstance(groups, list):
        return [row for row in groups if isinstance(row, dict)]
    plan_groups = (artifact.get("mid_plan") or {}).get("groups")
    if isinstance(plan_groups, list):
        return [row for row in plan_groups if isinstance(row, dict)]
    return []


def _step_number(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[-1])


def audit_step(step_dir: Path) -> dict:
    step = _step_number(step_dir)
    records_path = step_dir / "type_guided_v2_patch_records.json"
    artifact_path = step_dir / "type_guided_v2_merge_artifact.json"
    record_rows = _read_json(records_path) if records_path.exists() else []
    artifact = _read_json(artifact_path) if artifact_path.exists() else {}
    step_record_path = step_dir / "step_record.json"
    step_record = _read_json(step_record_path) if step_record_path.exists() else {}

    records = [row for row in record_rows if isinstance(row, dict)]
    record_ids = {
        str(row.get("record_id") or "").strip()
        for row in records
        if str(row.get("record_id") or "").strip()
    }
    leaves = [
        row for row in artifact.get("leaf_patches", [])
        if isinstance(row, dict)
    ]
    leaf_ids = {
        str(row.get("leaf_id") or "").strip()
        for row in leaves
        if str(row.get("leaf_id") or "").strip()
    }
    groups = _mid_groups(artifact)
    mid_patches = [
        row for row in artifact.get("mid_patches", [])
        if isinstance(row, dict)
    ]
    multi_groups = [
        row for row in groups
        if len(set(_clean_ids(row.get("leaf_ids")))) >= 2
    ]
    singleton_groups = [
        row for row in groups
        if len(set(_clean_ids(row.get("leaf_ids")))) == 1
    ]
    multi_leaf_ids = {
        leaf_id
        for group in multi_groups
        for leaf_id in _clean_ids(group.get("leaf_ids"))
    }
    assigned_leaf_ids = {
        leaf_id
        for group in groups
        for leaf_id in _clean_ids(group.get("leaf_ids"))
    }
    leaf_record_ids = {
        record_id
        for leaf in leaves
        for record_id in _ids_from_node(leaf, ("record_ids",))
    }

    root_level = str(artifact.get("root_children_level") or "leaf")
    root_patch = artifact.get("root_patch")
    if not isinstance(root_patch, dict):
        root_patch = {}
    if root_level == "mid":
        expected_root_ids = {
            str(row.get("mid_id") or "").strip()
            for row in mid_patches
            if str(row.get("mid_id") or "").strip()
        }
        root_source_ids = _ids_from_node(root_patch, ("mid_ids", "source_child_ids"))
    elif root_level == "leaf":
        expected_root_ids = leaf_ids
        root_source_ids = _ids_from_node(root_patch, ("leaf_ids", "source_child_ids"))
    else:
        expected_root_ids = record_ids
        root_source_ids = _ids_from_node(root_patch, ("record_ids", "source_child_ids"))

    root_coverage = (
        len(expected_root_ids & root_source_ids) / len(expected_root_ids)
        if expected_root_ids else 0.0
    )
    return {
        "step": step,
        "action": str(step_record.get("action") or ""),
        "edit_budget": int(step_record.get("edit_budget", 0) or 0),
        "n_records": len(records),
        "n_record_ids": len(record_ids),
        "n_leaves": len(leaves),
        "n_leaf_ids": len(leaf_ids),
        "n_mid_nodes": len(mid_patches),
        "n_mid_groups": len(groups),
        "n_multi_leaf_mids": len(multi_groups),
        "n_singleton_mids": len(singleton_groups),
        "multi_leaf_coverage": (
            len(multi_leaf_ids & leaf_ids) / len(leaf_ids) if leaf_ids else 0.0
        ),
        "assigned_leaf_coverage": (
            len(assigned_leaf_ids & leaf_ids) / len(leaf_ids) if leaf_ids else 0.0
        ),
        "leaf_record_coverage": (
            len(leaf_record_ids & record_ids) / len(record_ids) if record_ids else 0.0
        ),
        "root_children_level": root_level,
        "root_child_coverage": root_coverage,
        "unknown_mid_leaf_ids": sorted(assigned_leaf_ids - leaf_ids),
        "unknown_root_child_ids": sorted(root_source_ids - expected_root_ids),
        "step_dir": str(step_dir),
    }


def audit_run(source_run: Path) -> list[dict]:
    step_dirs = sorted(
        (source_run / "steps").glob("step_[0-9][0-9][0-9][0-9]"),
        key=_step_number,
    )
    if not step_dirs:
        raise FileNotFoundError(f"no step directories under {source_run / 'steps'}")
    return [audit_step(path) for path in step_dirs]


def _valid_tree(row: dict) -> bool:
    return (
        row["n_multi_leaf_mids"] >= 2
        and row["multi_leaf_coverage"] >= 0.5
        and not row["unknown_mid_leaf_ids"]
    )


def choose_main_step(rows: list[dict], requested: str) -> int:
    if requested != "auto":
        step = int(requested)
        row = next((item for item in rows if item["step"] == step), None)
        if row is None:
            raise ValueError(f"main step {step} is absent")
        if not str(row["action"]).startswith("accept"):
            raise ValueError(f"main step {step} is not accepted: {row['action']}")
        if not _valid_tree(row):
            raise ValueError(
                f"main step {step} lacks a valid multi-Leaf tree: {row}"
            )
        return step
    candidates = [
        row for row in rows
        if str(row["action"]).startswith("accept") and _valid_tree(row)
    ]
    if not candidates:
        raise ValueError("no accepted step satisfies the registered tree preconditions")
    candidates.sort(key=lambda row: (-row["multi_leaf_coverage"], row["step"]))
    return int(candidates[0]["step"])


def choose_fallback_step(rows: list[dict], requested: str) -> int:
    if requested != "auto":
        step = int(requested)
        row = next((item for item in rows if item["step"] == step), None)
        if row is None:
            raise ValueError(f"fallback step {step} is absent")
        if "reject" not in str(row["action"]):
            raise ValueError(f"fallback step {step} is not rejected: {row['action']}")
        if not _valid_tree(row):
            raise ValueError(
                f"fallback step {step} lacks a valid multi-Leaf tree: {row}"
            )
        return step
    candidates = [
        row for row in rows
        if "reject" in str(row["action"]) and _valid_tree(row)
    ]
    if not candidates:
        raise ValueError("no rejected step satisfies the registered tree preconditions")
    candidates.sort(
        key=lambda row: (
            -(row["n_leaves"] / max(row["n_mid_nodes"], 1)),
            row["step"],
        )
    )
    return int(candidates[0]["step"])


def _configure_optimizer(model: str) -> None:
    endpoint = (
        os.environ.get("OPTIMIZER_AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or ""
    ).strip()
    api_key = (
        os.environ.get("OPTIMIZER_AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or ""
    ).strip()
    auth_mode = (
        os.environ.get("OPTIMIZER_AZURE_OPENAI_AUTH_MODE")
        or os.environ.get("AZURE_OPENAI_AUTH_MODE")
        or "openai_compatible"
    ).strip()
    api_version = (
        os.environ.get("OPTIMIZER_AZURE_OPENAI_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or "2024-12-01-preview"
    ).strip()
    if not endpoint or not api_key:
        raise RuntimeError(
            "optimizer endpoint/key missing; export DeepSeek/Ark optimizer environment first"
        )
    configure_azure_openai(
        optimizer_endpoint=endpoint,
        optimizer_api_key=api_key,
        optimizer_auth_mode=auth_mode,
        optimizer_api_version=api_version,
    )
    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(model)
    set_reasoning_effort(None)


def _copy_json(source: Path, target: Path) -> Any:
    value = _read_json(source)
    _write_json(target, value)
    return value


def _rank_apply(
    *,
    parent_skill: str,
    merged_patch: dict,
    edit_budget: int,
    out_dir: Path,
) -> tuple[dict, str, list[dict]]:
    _write_json(out_dir / "merged_patch.json", merged_patch)
    ranked = rank_and_select(
        parent_skill,
        merged_patch,
        max_edits=edit_budget,
        update_mode="patch",
    )
    _write_json(out_dir / "ranked_edits.json", ranked)
    candidate, apply_report = apply_patch_with_report(parent_skill, ranked)
    _write_text(out_dir / "candidate_skill.md", candidate)
    _write_json(out_dir / "apply_report.json", apply_report)
    return ranked, candidate, apply_report


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["run_name", "skill_path"],
        )
        writer.writeheader()
        for row in rows:
            skill_path = Path(row["skill_path"]).resolve()
            writer.writerow({
                "run_name": row["run_name"],
                "skill_path": os.path.relpath(skill_path, path.parent.resolve()),
            })


def _prepare_topdown(
    *,
    source_run: Path,
    step: int,
    out_root: Path,
) -> dict:
    step_dir = source_run / "steps" / f"step_{step:04d}"
    parent_path = source_run / "skills" / f"skill_v{step - 1:04d}.md"
    artifact_path = step_dir / "type_guided_v2_merge_artifact.json"
    root_path = step_dir / "candidate_skill.md"
    for path in (parent_path, artifact_path, root_path):
        if not path.exists():
            raise FileNotFoundError(path)
    parent = parent_path.read_text(encoding="utf-8")
    artifact = _read_json(artifact_path)
    children = [
        row for row in artifact.get("root_child_patches", [])
        if isinstance(row, dict) and get_payload_items(row, "patch")
    ]
    target_dir = out_root / "topdown"
    target_dir.mkdir(parents=True, exist_ok=True)
    parent_target = target_dir / "parent_skill.md"
    root_target = target_dir / "rejected_root_skill.md"
    _write_text(parent_target, parent)
    shutil.copyfile(root_path, root_target)
    _write_json(target_dir / "source_tree_artifact.json", artifact)

    manifest_rows = [
        {"run_name": "td_parent", "skill_path": str(parent_target)},
        {"run_name": "td_root", "skill_path": str(root_target)},
    ]
    child_manifest: list[dict] = []
    child_tsv_rows: list[dict[str, str]] = []
    for index, child in enumerate(children, start=1):
        child_id = str(
            child.get("mid_id") or child.get("leaf_id") or f"C{index}"
        )
        child_dir = target_dir / "children" / child_id
        child_candidate, apply_report = apply_patch_with_report(parent, child)
        _write_json(child_dir / "patch.json", child)
        _write_text(child_dir / "candidate_skill.md", child_candidate)
        _write_json(child_dir / "apply_report.json", apply_report)
        manifest_rows.append({
            "run_name": f"td_child_{child_id}",
            "skill_path": str(child_dir / "candidate_skill.md"),
        })
        child_manifest.append({
            "child_id": child_id,
            "patch_path": str(child_dir / "patch.json"),
            "skill_path": str(child_dir / "candidate_skill.md"),
            "skill_hash": skill_hash(child_candidate),
            "support_count": int(child.get("support_count", 0) or 0),
            "leaf_ids": _clean_ids(child.get("leaf_ids")),
        })
        child_tsv_rows.append({
            "child_id": child_id,
            "run_name": f"td_child_{child_id}",
        })
    _write_tsv(target_dir / "phase1_val_manifest.tsv", manifest_rows)
    child_manifest_path = target_dir / "child_manifest.tsv"
    with child_manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=["child_id", "run_name"],
        )
        writer.writeheader()
        writer.writerows(child_tsv_rows)
    report = {
        "source_step": step,
        "parent_source": str(parent_path),
        "parent_hash": skill_hash(parent),
        "root_source": str(root_path),
        "root_hash": skill_hash(root_target.read_text(encoding="utf-8")),
        "root_children_level": artifact.get("root_children_level"),
        "n_children": len(child_manifest),
        "children": child_manifest,
        "child_manifest": str(child_manifest_path),
        "phase1_val_manifest": str(target_dir / "phase1_val_manifest.tsv"),
    }
    _write_json(target_dir / "topdown_prepare_report.json", report)
    _write_text(target_dir / "global_step.txt", f"{step}\n")
    return report


def prepare(args: argparse.Namespace) -> dict:
    source_run = Path(args.source_run).resolve()
    out_root = Path(args.out_dir).resolve()
    rows = audit_run(source_run)
    main_step = choose_main_step(rows, args.main_step)
    fallback_step = choose_fallback_step(rows, args.fallback_step)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "all_steps_structure_audit.json", rows)

    main_dir = source_run / "steps" / f"step_{main_step:04d}"
    parent_path = source_run / "skills" / f"skill_v{main_step - 1:04d}.md"
    required = [
        parent_path,
        main_dir / "type_guided_v2_patch_records.json",
        main_dir / "type_guided_v2_merge_artifact.json",
        main_dir / "ranked_edits.json",
        main_dir / "candidate_skill.md",
        main_dir / "step_record.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    parent_skill = parent_path.read_text(encoding="utf-8")
    records = _read_json(main_dir / "type_guided_v2_patch_records.json")
    tree_artifact = _read_json(main_dir / "type_guided_v2_merge_artifact.json")
    step_record = _read_json(main_dir / "step_record.json")
    edit_budget = int(step_record.get("edit_budget", 0) or 0)
    if edit_budget < 1:
        raise ValueError(f"invalid edit budget in {main_dir / 'step_record.json'}")

    main_out = out_root / "main"
    parent_target = main_out / "g0_parent" / "candidate_skill.md"
    _write_text(parent_target, parent_skill)
    _write_json(main_out / "fixed_patch_records.json", records)

    tree_dir = main_out / "g3_tree"
    _copy_json(
        main_dir / "type_guided_v2_merge_artifact.json",
        tree_dir / "merge_artifact.json",
    )
    tree_ranked = _copy_json(
        main_dir / "ranked_edits.json",
        tree_dir / "ranked_edits.json",
    )
    tree_candidate = (main_dir / "candidate_skill.md").read_text(encoding="utf-8")
    _write_text(tree_dir / "candidate_skill.md", tree_candidate)
    replayed_tree, tree_apply_report = apply_patch_with_report(parent_skill, tree_ranked)
    _write_json(tree_dir / "apply_report_replayed.json", tree_apply_report)
    if replayed_tree != tree_candidate:
        raise ValueError(
            "saved G3 candidate is not reproducible from its parent + ranked_edits"
        )

    if not args.audit_only:
        _configure_optimizer(args.optimizer_model)
        flat_dir = main_out / "g1_flat"
        flat_patch, flat_artifact = merge_type_guided_v2_records(
            skill_content=parent_skill,
            patch_records=records,
            min_support=1,
            max_leaf_groups=40,
            tree_depth=1,
            clustering_enabled=False,
            leaf_merge_workers=1,
            mid_merge_workers=1,
            optimizer_model=args.optimizer_model,
            verbose=True,
        )
        _write_json(flat_dir / "merge_artifact.json", flat_artifact)
        _, flat_candidate, _ = _rank_apply(
            parent_skill=parent_skill,
            merged_patch=flat_patch,
            edit_budget=edit_budget,
            out_dir=flat_dir,
        )

        frozen_leaves = [
            row for row in tree_artifact.get("leaf_patches", [])
            if isinstance(row, dict)
        ]
        if not frozen_leaves:
            raise ValueError("saved Tree artifact has no frozen Leaf patches")
        leaf_dir = main_out / "g2_leaf"
        leaf_patch = _build_root_patch(
            skill_content=parent_skill,
            child_patches=frozen_leaves,
            allow_open_types=True,
            child_level="leaf",
        )
        leaf_artifact = {
            "version": "fixed_leaf_root_replay_v1",
            "tree_depth": 2,
            "root_children_level": "leaf",
            "source_tree_artifact": str(
                main_dir / "type_guided_v2_merge_artifact.json"
            ),
            "frozen_leaf_patches": frozen_leaves,
            "root_patch": leaf_patch,
        }
        _write_json(leaf_dir / "merge_artifact.json", leaf_artifact)
        _, leaf_candidate, _ = _rank_apply(
            parent_skill=parent_skill,
            merged_patch=leaf_patch,
            edit_budget=edit_budget,
            out_dir=leaf_dir,
        )
        manifest_rows = [
            {"run_name": "g0_parent", "skill_path": str(parent_target)},
            {
                "run_name": "g1_flat",
                "skill_path": str(flat_dir / "candidate_skill.md"),
            },
            {
                "run_name": "g2_leaf",
                "skill_path": str(leaf_dir / "candidate_skill.md"),
            },
            {
                "run_name": "g3_tree",
                "skill_path": str(tree_dir / "candidate_skill.md"),
            },
        ]
        _write_tsv(main_out / "skill_manifest.tsv", manifest_rows)
    else:
        flat_candidate = ""
        leaf_candidate = ""
        manifest_rows = []

    topdown_report = _prepare_topdown(
        source_run=source_run,
        step=fallback_step,
        out_root=out_root,
    )
    report = {
        "source_run": str(source_run),
        "source_run_hashes": {
            "parent_skill": _sha256_file(parent_path),
            "patch_records": _sha256_file(
                main_dir / "type_guided_v2_patch_records.json"
            ),
            "tree_artifact": _sha256_file(
                main_dir / "type_guided_v2_merge_artifact.json"
            ),
        },
        "main_step": main_step,
        "fallback_step": fallback_step,
        "edit_budget": edit_budget,
        "optimizer_model": args.optimizer_model,
        "audit_only": bool(args.audit_only),
        "tree_replay_matches_saved_candidate": replayed_tree == tree_candidate,
        "candidate_hashes": {
            "g0_parent": skill_hash(parent_skill),
            "g1_flat": skill_hash(flat_candidate) if flat_candidate else None,
            "g2_leaf": skill_hash(leaf_candidate) if leaf_candidate else None,
            "g3_tree": skill_hash(tree_candidate),
        },
        "main_manifest": (
            str(main_out / "skill_manifest.tsv") if manifest_rows else None
        ),
        "topdown": topdown_report,
    }
    _write_json(out_root / "replay_manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--main-step", default="auto")
    parser.add_argument("--fallback-step", default="auto")
    parser.add_argument("--optimizer-model", default="deepseek-v4-pro")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit/select/copy saved artifacts without calling the optimizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
