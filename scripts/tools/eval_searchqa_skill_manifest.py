#!/usr/bin/env python3
"""Evaluate an absolute-path SearchQA skill manifest with safe resume checks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from skillopt.envs.searchqa.adapter import SearchQAAdapter
from skillopt.model import (
    configure_qwen_chat,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from skillopt.utils import compute_score, skill_hash


def _read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    bad: list[str] = []
    if not path.exists():
        return rows, bad
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            bad.append(f"line {line_no}: invalid json")
            continue
        if not isinstance(row, dict) or row.get("id") is None:
            bad.append(f"line {line_no}: missing object/id")
            continue
        rows.append(row)
    return rows, bad


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows or set(rows[0]) != {"run_name", "skill_path"}:
        raise ValueError(
            f"{path} must have exactly two TSV columns: run_name, skill_path"
        )
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        name = str(row.get("run_name") or "").strip()
        skill = str(row.get("skill_path") or "").strip()
        if not name or not skill or name in seen:
            raise ValueError(f"invalid/duplicate manifest row: {row}")
        seen.add(name)
        normalized.append({"run_name": name, "skill_path": skill})
    return normalized


def _validate_results(rows: list[dict], expected_ids: list[str]) -> dict:
    ids = [str(row.get("id")) for row in rows]
    counts: dict[str, int] = {}
    for item_id in ids:
        counts[item_id] = counts.get(item_id, 0) + 1
    duplicates = sorted(item_id for item_id, count in counts.items() if count > 1)
    expected = set(expected_ids)
    actual = set(ids)
    failed = [
        str(row.get("id"))
        for row in rows
        if row.get("agent_ok") is not True
    ]
    return {
        "n_rows": len(rows),
        "n_unique": len(actual),
        "duplicates": duplicates,
        "missing_ids": sorted(expected - actual),
        "unexpected_ids": sorted(actual - expected),
        "agent_failed_ids": failed,
        "valid": (
            len(rows) == len(expected_ids)
            and len(actual) == len(expected_ids)
            and not duplicates
            and actual == expected
            and not failed
        ),
    }


def _filter_items(items: list[dict], ids_path: str) -> list[dict]:
    if not ids_path:
        return items
    raw_ids = [
        line.strip()
        for line in Path(ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError(f"duplicate IDs in {ids_path}")
    by_id = {str(item["id"]): item for item in items}
    missing = [item_id for item_id in raw_ids if item_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} requested IDs are absent: {missing[:5]}")
    return [by_id[item_id] for item_id in raw_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--item-ids-file", default="")
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_root = Path(args.out_root).resolve()
    rows = _manifest_rows(manifest_path)

    set_target_backend("qwen_chat")
    set_target_deployment(args.target_model)
    configure_qwen_chat(
        target_base_url=args.base_url,
        target_api_key=args.api_key,
        target_temperature=args.temperature,
        target_timeout_seconds=args.timeout,
        target_max_tokens=args.max_tokens,
        target_enable_thinking=False,
    )
    set_reasoning_effort(None)
    adapter = SearchQAAdapter(
        split_dir=str(Path(args.split_dir).resolve()),
        split_mode="split_dir",
        max_turns=args.max_turns,
        exec_timeout=args.timeout,
        workers=args.workers,
        seed=args.seed,
        limit=0,
        max_completion_tokens=args.max_tokens,
    )
    adapter.setup({
        "env": "searchqa",
        "split_mode": "split_dir",
        "split_dir": str(Path(args.split_dir).resolve()),
        "out_root": str(out_root),
    })
    split_name = "valid_seen" if args.split == "val" else "valid_unseen"
    items = adapter.build_eval_env(0, split_name, args.seed)
    items = _filter_items(items, args.item_ids_file)
    item_ids = [str(item["id"]) for item in items]

    suite_rows: list[dict] = []
    for manifest_row in rows:
        name = manifest_row["run_name"]
        skill_path = Path(manifest_row["skill_path"]).resolve()
        if not skill_path.is_file():
            raise FileNotFoundError(skill_path)
        skill_content = skill_path.read_text(encoding="utf-8")
        run_dir = out_root / name / args.split
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol = {
            "version": "searchqa_tree_verdict_eval_v1",
            "run_name": name,
            "split": args.split,
            "skill_path": str(skill_path),
            "skill_hash": skill_hash(skill_content),
            "item_ids": item_ids,
            "item_ids_hash": _sha256_json(item_ids),
            "target_model": args.target_model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "workers": args.workers,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "git_commit": _git_commit(),
        }
        protocol_path = run_dir / "protocol.json"
        if protocol_path.exists():
            old = json.loads(protocol_path.read_text(encoding="utf-8"))
            if old != protocol:
                raise RuntimeError(
                    f"unsafe resume blocked for {run_dir}: protocol changed"
                )
        else:
            _atomic_json(protocol_path, protocol)

        results_path = run_dir / "results.jsonl"
        existing, bad_lines = _read_jsonl(results_path)
        if bad_lines:
            raise RuntimeError(f"malformed results in {results_path}: {bad_lines[:3]}")
        if args.retry_failed and existing:
            kept = [row for row in existing if row.get("agent_ok") is True]
            if len(kept) != len(existing):
                tmp = results_path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as file:
                    for row in kept:
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                os.replace(tmp, results_path)

        started = time.time()
        print(f"[eval] {name} split={args.split} n={len(items)} skill={skill_path}")
        results = adapter.rollout(items, skill_content, str(run_dir))
        elapsed = time.time() - started
        validation = _validate_results(results, item_ids)
        hard, soft = compute_score(results)
        summary = {
            "run_name": name,
            "split": args.split,
            "skill_path": str(skill_path),
            "skill_hash": skill_hash(skill_content),
            "n_items": len(results),
            "hard": hard,
            "soft": soft,
            "elapsed_seconds": round(elapsed, 3),
            "validation": validation,
        }
        _atomic_json(run_dir / "eval_summary.json", summary)
        suite_rows.append(summary)
        if not validation["valid"]:
            raise RuntimeError(
                f"invalid verdict results for {name}/{args.split}: {validation}"
            )

    _atomic_json(
        out_root / f"{args.split}_suite_summary.json",
        {
            "manifest": str(manifest_path),
            "split": args.split,
            "item_ids_hash": _sha256_json(item_ids),
            "runs": suite_rows,
        },
    )
    print(f"[done] {args.split} results: {out_root}")


if __name__ == "__main__":
    main()
