#!/usr/bin/env python3
"""Validate blind SearchQA type patches on held-out members and boundaries."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli.train import get_adapter  # noqa: E402
from scripts.tools.audit_observed_type_taxonomy import item_id, repeated_rollout  # noqa: E402
from scripts.tools.searchqa_blind_common import (  # noqa: E402
    load_unique_cards,
    write_json,
)
from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.gradient.type_guided_merge_v2 import (  # noqa: E402
    _group_repeated_rollouts,
    _success_rate,
)
from skillopt.model import (  # noqa: E402
    configure_qwen_chat,
    set_target_backend,
    set_target_deployment,
)
from skillopt.optimizer.skill import apply_edit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--cards", action="append", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/searchqa/default.yaml")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--target-base-url", default="http://127.0.0.1:59317/v1")
    parser.add_argument("--target-api-key", default="dummy")
    parser.add_argument("--target-temperature", type=float, default=0.2)
    parser.add_argument("--target-timeout-seconds", type=float, default=300)
    parser.add_argument("--target-max-tokens", type=int, default=4096)
    parser.add_argument("--target-workers", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-holdout-per-type", type=int, default=40)
    parser.add_argument("--max-boundary-per-type", type=int, default=8)
    parser.add_argument("--min-delta-in", type=float, default=0.05)
    parser.add_argument("--max-boundary-drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.target_workers, args.batch_size, args.repeats) < 1:
        parser.error("worker, batch, and repeat counts must be positive")
    return args


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    taxonomy = json.loads(args.taxonomy.expanduser().resolve().read_text(encoding="utf-8"))
    types = list(taxonomy.get("types") or [])
    cards = load_unique_cards(args.cards)
    cards_by_key = {str(card["sample_key"]): card for card in cards}
    planned = sum(
        min(len(item.get("holdout_member_keys") or []), args.max_holdout_per_type)
        + min(len(item.get("boundary_member_keys") or []), args.max_boundary_per_type)
        for item in types
    )
    print("============================================================")
    print("  SearchQA blind taxonomy transfer validation")
    print("============================================================")
    print(f"  types:        {len(types)}")
    print(f"  sample slots: {planned} before cross-type overlap")
    print(f"  repeats:      {args.repeats}")
    print(f"  target:       {args.target_model} @ {args.target_base_url}")
    print(f"  output:       {args.output_dir}")
    print("============================================================")
    if args.dry_run:
        return

    cfg_raw = load_config(str(args.config.expanduser().resolve()))
    cfg = flatten_config(cfg_raw) if is_structured(cfg_raw) else dict(cfg_raw)
    cfg.update({
        "split_mode": "split_dir",
        "limit": 0,
        "workers": args.target_workers,
        "max_api_workers": args.target_workers,
        "target_backend": "qwen_chat",
        "target_model": args.target_model,
        "target_qwen_chat_base_url": args.target_base_url,
        "target_qwen_chat_api_key": args.target_api_key,
        "target_qwen_chat_temperature": args.target_temperature,
        "target_qwen_chat_timeout_seconds": args.target_timeout_seconds,
        "target_qwen_chat_max_tokens": args.target_max_tokens,
        "target_qwen_chat_enable_thinking": False,
        "out_root": str(args.output_dir.expanduser().resolve()),
    })
    set_target_backend("qwen_chat")
    set_target_deployment(args.target_model)
    configure_qwen_chat(
        target_base_url=args.target_base_url,
        target_api_key=args.target_api_key,
        target_temperature=args.target_temperature,
        target_timeout_seconds=args.target_timeout_seconds,
        target_max_tokens=args.target_max_tokens,
        target_enable_thinking=False,
    )
    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    dataloader = adapter.get_dataloader()
    if dataloader is None or not hasattr(dataloader, "get_split_items"):
        raise SystemExit("SearchQA dataloader does not expose split items")
    items_by_key: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        for index, item in enumerate(dataloader.get_split_items(split)):
            items_by_key[f"{split}::{item_id(item, index)}"] = item
    skill_path = args.skill or Path(str(cfg["skill_init"]))
    if not skill_path.is_absolute():
        skill_path = PROJECT_ROOT / skill_path
    initial_skill = skill_path.resolve().read_text(encoding="utf-8")
    output_dir = args.output_dir.expanduser().resolve()
    validation_rows = []

    for type_index, item in enumerate(types, 1):
        type_id = str(item["type_id"])
        result_path = output_dir / type_id / "transfer_result.json"
        if result_path.is_file():
            validation_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            print(f"[resume] {type_id}")
            continue
        holdout = [
            key for key in (item.get("holdout_member_keys") or [])[:args.max_holdout_per_type]
            if key in items_by_key and key in cards_by_key
        ]
        boundary = [
            key for key in (item.get("boundary_member_keys") or [])[:args.max_boundary_per_type]
            if key in items_by_key and key in cards_by_key and key not in holdout
        ]
        selected = set(holdout + boundary)
        patched_skill = apply_edit(initial_skill, item["shared_patch"])
        patched_q: dict[str, float] = {}
        grouped_by_split: dict[str, list[str]] = defaultdict(list)
        for key in selected:
            grouped_by_split[key.split("::", 1)[0]].append(key)
        for split, keys in sorted(grouped_by_split.items()):
            keys.sort()
            for start in range(0, len(keys), args.batch_size):
                batch_keys = keys[start:start + args.batch_size]
                chunk_dir = output_dir / type_id / "rollouts" / split / f"chunk_{start // args.batch_size:04d}"
                repeated = repeated_rollout(
                    adapter=adapter,
                    dataloader=dataloader,
                    items=[items_by_key[key] for key in batch_keys],
                    split=split,
                    seed=args.seed + type_index * 1009 + start,
                    repeats=args.repeats,
                    skill_content=patched_skill,
                    chunk_dir=chunk_dir,
                )
                groups = _group_repeated_rollouts(repeated, include_trajectories=False)
                for key in batch_keys:
                    sample_id = key.split("::", 1)[1]
                    if sample_id not in groups:
                        raise RuntimeError(f"missing patched rollout group: {key}")
                    patched_q[key] = _success_rate(groups[sample_id])
        base_holdout = [float(cards_by_key[key]["q_i"]) for key in holdout]
        patch_holdout = [patched_q[key] for key in holdout]
        base_boundary = [float(cards_by_key[key]["q_i"]) for key in boundary]
        patch_boundary = [patched_q[key] for key in boundary]
        delta_in = mean(patch_holdout) - mean(base_holdout)
        delta_boundary = mean(patch_boundary) - mean(base_boundary)
        accepted = (
            len(holdout) >= 2
            and delta_in >= args.min_delta_in
            and delta_boundary >= -args.max_boundary_drop
        )
        result = {
            "type_id": type_id,
            "revision_type": item["revision_type"],
            "accepted": accepted,
            "n_holdout": len(holdout),
            "n_boundary": len(boundary),
            "base_holdout_accuracy": mean(base_holdout),
            "patched_holdout_accuracy": mean(patch_holdout),
            "delta_in": delta_in,
            "base_boundary_accuracy": mean(base_boundary),
            "patched_boundary_accuracy": mean(patch_boundary),
            "delta_boundary": delta_boundary,
            "thresholds": {
                "min_delta_in": args.min_delta_in,
                "max_boundary_drop": args.max_boundary_drop,
            },
            "holdout_keys": holdout,
            "boundary_keys": boundary,
            "patched_q_i": patched_q,
        }
        write_json(result_path, result)
        validation_rows.append(result)
        print(
            f"[validate] {type_id} {type_index}/{len(types)} "
            f"delta_in={delta_in:+.4f} delta_boundary={delta_boundary:+.4f} "
            f"accepted={accepted}"
        )

    by_id = {row["type_id"]: row for row in validation_rows}
    validated = dict(taxonomy)
    for item in validated["types"]:
        item["transfer_validation"] = by_id.get(item["type_id"])
    write_json(output_dir / "validated_blind_revision_taxonomy.json", validated)
    lines = [
        "# SearchQA blind taxonomy transfer validation",
        "",
        "| Type | Name | Holdout | Δ in-cluster | Boundary | Δ boundary | Accepted |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['type_id']} | {row['revision_type']} | {row['n_holdout']} | "
            f"{row['delta_in']:+.4f} | {row['n_boundary']} | "
            f"{row['delta_boundary']:+.4f} | {row['accepted']} |"
        )
    (output_dir / "transfer_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[done] accepted={sum(row['accepted'] for row in validation_rows)}/{len(validation_rows)}")


if __name__ == "__main__":
    main()
