#!/usr/bin/env python3
"""Validate blind SearchQA type patches with per-sample paired A/B cohorts.

The rollout ``seed`` controls the repository's repeat/batch bookkeeping.  The
SearchQA Qwen chat backend does not currently forward it as a generation seed,
so baseline and patched results are paired by sample, not by identical model
randomness.
"""
from __future__ import annotations

import argparse
import hashlib
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


VALIDATION_SCHEMA_VERSION = "searchqa_blind_transfer_paired_v3"
PAIR_EPSILON = 1e-12


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
    parser.add_argument("--min-holdout-samples", type=int, default=10)
    parser.add_argument("--min-boundary-samples", type=int, default=4)
    parser.add_argument("--min-delta-in", type=float, default=0.05)
    parser.add_argument("--max-boundary-drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.target_workers, args.batch_size, args.repeats) < 1:
        parser.error("worker, batch, and repeat counts must be positive")
    if min(args.min_holdout_samples, args.min_boundary_samples) < 0:
        parser.error("minimum validation sample counts must be non-negative")
    return args


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stratified_sample_keys(
    candidate_keys: list[str],
    cards_by_key: dict[str, dict[str, Any]],
    maximum: int,
) -> list[str]:
    """Select a deterministic split x outcome-stratified sample of card keys.

    When a cap applies, every available stratum is represented if the cap
    permits it.  Remaining slots are apportioned by stratum size, and midpoint
    selection avoids taking a lexicographic prefix within a stratum.
    """
    candidates = sorted({
        str(key)
        for key in candidate_keys
        if str(key) in cards_by_key
    })
    if maximum <= 0:
        return []
    if len(candidates) <= maximum:
        return candidates

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in candidates:
        card = cards_by_key[key]
        split = str(card.get("split") or key.split("::", 1)[0] or "unknown")
        outcome = str(card.get("outcome_status") or "unknown")
        buckets[(split, outcome)].append(key)

    strata = sorted(buckets, key=lambda stratum: (-len(buckets[stratum]), stratum))
    selected_strata = strata[:maximum]
    allocations = {stratum: 1 for stratum in selected_strata}
    remaining = maximum - len(selected_strata)
    while remaining:
        eligible = [
            stratum
            for stratum in selected_strata
            if allocations[stratum] < len(buckets[stratum])
        ]
        if not eligible:
            break
        # D'Hondt-style apportionment is deterministic and tracks the observed
        # stratum proportions while retaining the one-per-stratum guarantee.
        best = min(
            eligible,
            key=lambda stratum: (
                -len(buckets[stratum]) / (allocations[stratum] + 1),
                stratum,
            ),
        )
        allocations[best] += 1
        remaining -= 1

    selected: list[str] = []
    for stratum in sorted(selected_strata):
        bucket = buckets[stratum]
        count = allocations[stratum]
        positions = [
            min(int((index + 0.5) * len(bucket) / count), len(bucket) - 1)
            for index in range(count)
        ]
        selected.extend(bucket[index] for index in positions)
    return selected


def paired_summary(
    keys: list[str],
    baseline_new_q: dict[str, float],
    patched_q: dict[str, float],
) -> dict[str, Any]:
    """Summarize paired per-sample changes for one validation cohort."""
    pairs: list[dict[str, Any]] = []
    counts = {"improved": 0, "unchanged": 0, "regressed": 0}
    for key in keys:
        baseline = float(baseline_new_q[key])
        patched = float(patched_q[key])
        delta = patched - baseline
        if delta > PAIR_EPSILON:
            outcome = "improved"
        elif delta < -PAIR_EPSILON:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        counts[outcome] += 1
        pairs.append({
            "sample_key": key,
            "baseline_new_q_i": baseline,
            "patched_q_i": patched,
            "paired_delta": delta,
            "paired_outcome": outcome,
        })
    baseline_accuracy = mean([row["baseline_new_q_i"] for row in pairs])
    patched_accuracy = mean([row["patched_q_i"] for row in pairs])
    return {
        "n": len(pairs),
        "baseline_new_accuracy": baseline_accuracy,
        "patched_accuracy": patched_accuracy,
        "paired_mean_delta": mean([row["paired_delta"] for row in pairs]),
        "accuracy_delta": patched_accuracy - baseline_accuracy,
        "improved": counts["improved"],
        "unchanged": counts["unchanged"],
        "regressed": counts["regressed"],
        "pairs": pairs,
    }


def validation_protocol(repeats: int) -> dict[str, Any]:
    """Describe pairing without overstating backend generation determinism."""
    return {
        "paired_ab": True,
        "paired_by_sample": True,
        "baseline_skill": "initial_skill",
        "repeats": repeats,
        "same_rollout_batch_seed_argument": True,
        "generation_randomness_seed_paired": False,
        "seed_note": (
            "SearchQA/Qwen does not forward the rollout batch seed as "
            "a generation seed; pairing is by sample only."
        ),
        "stored_card_q_i_used_for_acceptance": False,
    }


def validation_fingerprint(
    *,
    type_item: dict[str, Any],
    initial_skill: str,
    selected_items: dict[str, dict[str, Any]],
    holdout: list[str],
    boundary: list[str],
    type_index: int,
    runtime_config_sha256: str,
    args: argparse.Namespace,
) -> str:
    """Fingerprint every input that can affect a cached validation result."""
    return stable_hash({
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "type_id": type_item.get("type_id"),
        "revision_type": type_item.get("revision_type"),
        "shared_patch": type_item.get("shared_patch"),
        "initial_skill_sha256": stable_hash(initial_skill),
        "runtime_config_sha256": runtime_config_sha256,
        "holdout_keys": holdout,
        "boundary_keys": boundary,
        "selected_items": {
            key: selected_items[key]
            for key in sorted(selected_items)
        },
        "target": {
            "model": args.target_model,
            "base_url": args.target_base_url,
            "temperature": args.target_temperature,
            "timeout_seconds": args.target_timeout_seconds,
            "max_tokens": args.target_max_tokens,
            "enable_thinking": False,
        },
        "sampling": {
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "type_seed_offset": type_index * 1009,
        },
        "acceptance": {
            "min_holdout_samples": args.min_holdout_samples,
            "min_boundary_samples": args.min_boundary_samples,
            "min_delta_in": args.min_delta_in,
            "max_boundary_drop": args.max_boundary_drop,
        },
    })


def reusable_cached_result(
    cached: Any,
    expected_fingerprint: str,
) -> bool:
    return (
        isinstance(cached, dict)
        and cached.get("schema_version") == VALIDATION_SCHEMA_VERSION
        and cached.get("validation_fingerprint") == expected_fingerprint
    )


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
    print("  pairing:      same samples; generation randomness is not seed-paired")
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
    runtime_config_sha256 = stable_hash(cfg)
    validation_rows = []

    for type_index, item in enumerate(types, 1):
        type_id = str(item["type_id"])
        result_path = output_dir / type_id / "transfer_result.json"
        eligible_holdout = [
            key for key in (item.get("holdout_member_keys") or [])
            if key in items_by_key and key in cards_by_key
        ]
        holdout = stratified_sample_keys(
            eligible_holdout,
            cards_by_key,
            args.max_holdout_per_type,
        )
        boundary = [
            key for key in (item.get("boundary_member_keys") or [])[:args.max_boundary_per_type]
            if key in items_by_key and key in cards_by_key and key not in holdout
        ]
        selected = set(holdout + boundary)
        fingerprint = validation_fingerprint(
            type_item=item,
            initial_skill=initial_skill,
            selected_items={key: items_by_key[key] for key in selected},
            holdout=holdout,
            boundary=boundary,
            type_index=type_index,
            runtime_config_sha256=runtime_config_sha256,
            args=args,
        )
        if result_path.is_file():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if reusable_cached_result(cached, fingerprint):
                validation_rows.append(cached)
                print(f"[resume] {type_id} paired cache")
                continue
            print(f"[stale] {type_id} cache does not match paired validation inputs")

        patched_skill = apply_edit(initial_skill, item["shared_patch"])
        baseline_new_q: dict[str, float] = {}
        patched_q: dict[str, float] = {}
        rollout_batch_seed_by_key: dict[str, int] = {}
        grouped_by_split: dict[str, list[str]] = defaultdict(list)
        for key in selected:
            grouped_by_split[key.split("::", 1)[0]].append(key)
        for split, keys in sorted(grouped_by_split.items()):
            keys.sort()
            for start in range(0, len(keys), args.batch_size):
                batch_keys = keys[start:start + args.batch_size]
                rollout_batch_seed = args.seed + type_index * 1009 + start
                chunk_root = (
                    output_dir / type_id / "paired_rollouts" / fingerprint[:16]
                    / split / f"chunk_{start // args.batch_size:04d}"
                )
                baseline_repeated = repeated_rollout(
                    adapter=adapter,
                    dataloader=dataloader,
                    items=[items_by_key[key] for key in batch_keys],
                    split=split,
                    seed=rollout_batch_seed,
                    repeats=args.repeats,
                    skill_content=initial_skill,
                    chunk_dir=chunk_root / "baseline_new",
                )
                patched_repeated = repeated_rollout(
                    adapter=adapter,
                    dataloader=dataloader,
                    items=[items_by_key[key] for key in batch_keys],
                    split=split,
                    seed=rollout_batch_seed,
                    repeats=args.repeats,
                    skill_content=patched_skill,
                    chunk_dir=chunk_root / "patched",
                )
                baseline_groups = _group_repeated_rollouts(
                    baseline_repeated, include_trajectories=False
                )
                patched_groups = _group_repeated_rollouts(
                    patched_repeated, include_trajectories=False
                )
                for key in batch_keys:
                    sample_id = key.split("::", 1)[1]
                    if sample_id not in baseline_groups:
                        raise RuntimeError(f"missing baseline_new rollout group: {key}")
                    if sample_id not in patched_groups:
                        raise RuntimeError(f"missing patched rollout group: {key}")
                    baseline_new_q[key] = _success_rate(baseline_groups[sample_id])
                    patched_q[key] = _success_rate(patched_groups[sample_id])
                    rollout_batch_seed_by_key[key] = rollout_batch_seed

        holdout_summary = paired_summary(holdout, baseline_new_q, patched_q)
        boundary_summary = paired_summary(boundary, baseline_new_q, patched_q)
        delta_in = float(holdout_summary["paired_mean_delta"])
        delta_boundary = float(boundary_summary["paired_mean_delta"])
        accepted = (
            len(holdout) >= args.min_holdout_samples
            and len(boundary) >= args.min_boundary_samples
            and delta_in >= args.min_delta_in
            and delta_boundary >= -args.max_boundary_drop
        )
        result = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_fingerprint": fingerprint,
            "type_id": type_id,
            "revision_type": item["revision_type"],
            "accepted": accepted,
            "n_holdout": len(holdout),
            "n_boundary": len(boundary),
            "baseline_new_holdout_accuracy": holdout_summary["baseline_new_accuracy"],
            "patched_holdout_accuracy": holdout_summary["patched_accuracy"],
            "delta_in": delta_in,
            "baseline_new_boundary_accuracy": boundary_summary["baseline_new_accuracy"],
            "patched_boundary_accuracy": boundary_summary["patched_accuracy"],
            "delta_boundary": delta_boundary,
            "paired_holdout": holdout_summary,
            "paired_boundary": boundary_summary,
            "thresholds": {
                "min_holdout_samples": args.min_holdout_samples,
                "min_boundary_samples": args.min_boundary_samples,
                "min_delta_in": args.min_delta_in,
                "max_boundary_drop": args.max_boundary_drop,
            },
            "holdout_keys": holdout,
            "boundary_keys": boundary,
            "baseline_new_q_i": baseline_new_q,
            "patched_q_i": patched_q,
            "rollout_batch_seed_by_key": rollout_batch_seed_by_key,
            "validation_protocol": validation_protocol(args.repeats),
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
        (
            "Baseline and patch are paired by sample. The rollout batch seed "
            "is not forwarded as a Qwen generation seed, so model randomness "
            "is not seed-paired."
        ),
        "",
        "| Type | Name | Holdout (↑/= /↓) | Δ in-cluster | Boundary (↑/= /↓) | Δ boundary | Accepted |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in validation_rows:
        holdout_counts = row["paired_holdout"]
        boundary_counts = row["paired_boundary"]
        lines.append(
            f"| {row['type_id']} | {row['revision_type']} | "
            f"{row['n_holdout']} ({holdout_counts['improved']}/"
            f"{holdout_counts['unchanged']}/{holdout_counts['regressed']}) | "
            f"{row['delta_in']:+.4f} | {row['n_boundary']} "
            f"({boundary_counts['improved']}/{boundary_counts['unchanged']}/"
            f"{boundary_counts['regressed']}) | "
            f"{row['delta_boundary']:+.4f} | {row['accepted']} |"
        )
    (output_dir / "transfer_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[done] accepted={sum(row['accepted'] for row in validation_rows)}/{len(validation_rows)}")


if __name__ == "__main__":
    main()
