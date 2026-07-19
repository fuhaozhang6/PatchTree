#!/usr/bin/env python3
"""Compare parent and child SearchQA patches on matched hierarchy cohorts.

The protocol is paired by sample: every skill variant is evaluated on exactly
the same selected examples.  Qwen chat does not currently expose a generation
seed, so the ``seed`` passed through BatchSpec must not be interpreted as
common-random-number or same-generation-seed pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli.train import get_adapter  # noqa: E402
from scripts.tools.audit_observed_type_taxonomy import item_id, repeated_rollout  # noqa: E402
from scripts.tools.searchqa_blind_common import load_unique_cards, write_json  # noqa: E402
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


HIERARCHY_SCHEMA_VERSION = "searchqa_blind_hierarchy_paired_by_sample_v1"
DEFAULT_PAIR_SPECS = (
    "R_SEARCH_001:R_SEARCH_002:R_SEARCH_004",
    "R_SEARCH_001:R_SEARCH_004:R_SEARCH_002",
)
PAIR_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--cards", action="append", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/searchqa/default.yaml")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pair",
        action="append",
        help=(
            "PARENT_TYPE_ID:CHILD_TYPE_ID[:CONTROL_TYPE_ID]. May be repeated. "
            "Defaults to R1/R2 with R4 control and R1/R4 with R2 control."
        ),
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--target-base-url", default="http://127.0.0.1:59317/v1")
    parser.add_argument("--target-api-key", default="dummy")
    parser.add_argument("--target-temperature", type=float, default=0.2)
    parser.add_argument("--target-timeout-seconds", type=float, default=300)
    parser.add_argument("--target-max-tokens", type=int, default=4096)
    parser.add_argument("--target-workers", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-child-holdout", type=int, default=40)
    parser.add_argument("--max-parent-reference", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.target_workers, args.batch_size, args.repeats) < 1:
        parser.error("worker, batch, and repeat counts must be positive")
    if min(args.max_child_holdout, args.max_parent_reference) < 0:
        parser.error("cohort limits must be non-negative")
    try:
        args.pair_specs = parse_pair_specs(args.pair or list(DEFAULT_PAIR_SPECS))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_pair_specs(raw_specs: list[str]) -> list[dict[str, str | None]]:
    """Parse and validate PARENT:CHILD[:CONTROL] pair specifications."""
    parsed: list[dict[str, str | None]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for raw in raw_specs:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) not in (2, 3) or any(not part for part in parts):
            raise ValueError(f"invalid --pair {raw!r}; expected PARENT:CHILD[:CONTROL]")
        parent_id, child_id = parts[:2]
        control_id = parts[2] if len(parts) == 3 else None
        if parent_id == child_id:
            raise ValueError(f"parent and child must differ in --pair {raw!r}")
        if control_id in {parent_id, child_id}:
            raise ValueError(f"control must differ from parent and child in --pair {raw!r}")
        identity = (parent_id, child_id, control_id)
        if identity in seen:
            raise ValueError(f"duplicate --pair {raw!r}")
        seen.add(identity)
        parsed.append({
            "parent_type_id": parent_id,
            "child_type_id": child_id,
            "control_type_id": control_id,
        })
    return parsed


def pair_slug(pair: dict[str, str | None]) -> str:
    slug = f"{pair['parent_type_id']}__{pair['child_type_id']}"
    if pair.get("control_type_id"):
        slug += f"__control_{pair['control_type_id']}"
    return slug


def card_stratum(key: str, cards_by_key: dict[str, dict[str, Any]]) -> tuple[str, str]:
    split = key.split("::", 1)[0]
    status = str(cards_by_key.get(key, {}).get("outcome_status") or "unknown")
    return split, status


def stratified_select(
    keys: list[str],
    *,
    cards_by_key: dict[str, dict[str, Any]],
    limit: int,
    seed: int,
) -> list[str]:
    """Deterministically sample across split × outcome strata."""
    unique_keys = sorted(set(str(key) for key in keys))
    if limit <= 0 or not unique_keys:
        return []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in unique_keys:
        groups[card_stratum(key, cards_by_key)].append(key)
    strata = sorted(groups)
    for stratum in strata:
        groups[stratum].sort(
            key=lambda key: stable_hash({"seed": seed, "stratum": stratum, "key": key})
        )
    selected: list[str] = []
    cursor = 0
    while len(selected) < min(limit, len(unique_keys)):
        emitted = False
        for stratum in strata:
            values = groups[stratum]
            if cursor < len(values):
                selected.append(values[cursor])
                emitted = True
                if len(selected) >= limit:
                    break
        if not emitted:
            break
        cursor += 1
    return selected


def build_variants(
    initial_skill: str,
    parent: dict[str, Any],
    child: dict[str, Any],
    control: dict[str, Any] | None,
) -> dict[str, str]:
    """Compile all hierarchy comparison skills from the same initial skill."""
    parent_skill = apply_edit(initial_skill, parent["shared_patch"])
    variants = {
        "initial": initial_skill,
        "parent": parent_skill,
        "child": apply_edit(initial_skill, child["shared_patch"]),
        "parent_plus_child": apply_edit(parent_skill, child["shared_patch"]),
    }
    if control is not None:
        variants["unrelated_control"] = apply_edit(initial_skill, control["shared_patch"])
    return variants


def deterministic_variant_order(
    variant_names: list[str],
    *,
    seed: int,
    pair_name: str,
    cohort_name: str,
    split: str,
    chunk_index: int,
) -> list[str]:
    """Shuffle execution order reproducibly to reduce service-time ordering bias."""
    material = {
        "schema": HIERARCHY_SCHEMA_VERSION,
        "seed": seed,
        "pair": pair_name,
        "cohort": cohort_name,
        "split": split,
        "chunk_index": chunk_index,
    }
    rng = random.Random(int(stable_hash(material)[:16], 16))
    ordered = list(variant_names)
    rng.shuffle(ordered)
    return ordered


def summarize_cohort(
    keys: list[str],
    q_by_variant: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Summarize raw variant accuracy and sample-paired deltas vs initial."""
    if "initial" not in q_by_variant:
        raise ValueError("initial variant is required")
    rows: dict[str, dict[str, Any]] = {}
    initial_q = q_by_variant["initial"]
    for variant, q_values in q_by_variant.items():
        deltas = [float(q_values[key]) - float(initial_q[key]) for key in keys]
        improved = sum(delta > PAIR_EPSILON for delta in deltas)
        regressed = sum(delta < -PAIR_EPSILON for delta in deltas)
        accuracy = sum(float(q_values[key]) for key in keys) / len(keys) if keys else 0.0
        rows[variant] = {
            "n": len(keys),
            "accuracy": accuracy,
            "delta_vs_initial": sum(deltas) / len(deltas) if deltas else 0.0,
            "improved_vs_initial": improved,
            "unchanged_vs_initial": len(deltas) - improved - regressed,
            "regressed_vs_initial": regressed,
        }
    best_accuracy = max((row["accuracy"] for row in rows.values()), default=0.0)
    metric_winners = [
        variant
        for variant, row in rows.items()
        if abs(float(row["accuracy"]) - best_accuracy) <= PAIR_EPSILON
    ]
    child_control = None
    if "child" in rows and "unrelated_control" in rows:
        child_control = {
            "child_accuracy": rows["child"]["accuracy"],
            "unrelated_control_accuracy": rows["unrelated_control"]["accuracy"],
            "child_minus_unrelated_control": (
                rows["child"]["accuracy"] - rows["unrelated_control"]["accuracy"]
            ),
        }
    return {
        "n": len(keys),
        "variants": rows,
        "metric_winners": metric_winners,
        "best_accuracy": best_accuracy,
        "winner_interpretation": "descriptive_raw_accuracy_only",
        "child_vs_unrelated_control": child_control,
        "sample_rates": {
            key: {variant: float(q_by_variant[variant][key]) for variant in q_by_variant}
            for key in keys
        },
    }


def hierarchy_decision_metrics(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    """Return the raw contrasts used to reason about hierarchy specialization."""
    child_rows = summaries["child_holdout"]["variants"]
    parent_rows = summaries["parent_reference"]["variants"]

    def difference(
        rows: dict[str, dict[str, Any]],
        left: str,
        right: str,
    ) -> float | None:
        if left not in rows or right not in rows:
            return None
        return float(rows[left]["accuracy"]) - float(rows[right]["accuracy"])

    return {
        "child_holdout_child_minus_parent": difference(child_rows, "child", "parent"),
        "child_holdout_combined_minus_parent": difference(
            child_rows, "parent_plus_child", "parent"
        ),
        "child_holdout_combined_minus_child": difference(
            child_rows, "parent_plus_child", "child"
        ),
        "child_holdout_child_minus_unrelated_control": difference(
            child_rows, "child", "unrelated_control"
        ),
        "parent_reference_child_minus_initial": difference(
            parent_rows, "child", "initial"
        ),
        "parent_reference_combined_minus_parent": difference(
            parent_rows, "parent_plus_child", "parent"
        ),
    }


def hierarchy_fingerprint(
    *,
    pair: dict[str, str | None],
    type_items: dict[str, dict[str, Any]],
    initial_skill: str,
    cohorts: dict[str, list[str]],
    selected_items: dict[str, dict[str, Any]],
    selected_cards: dict[str, dict[str, Any]],
    runtime_config_sha256: str,
    args: argparse.Namespace,
) -> str:
    """Fingerprint all inputs affecting selection, skills, rollout, or summary."""
    relevant_ids = [
        pair["parent_type_id"],
        pair["child_type_id"],
        pair.get("control_type_id"),
    ]
    return stable_hash({
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "pair": pair,
        "types": {type_id: type_items[type_id] for type_id in relevant_ids if type_id},
        "initial_skill_sha256": stable_hash(initial_skill),
        "cohorts": cohorts,
        "selected_items": {key: selected_items[key] for key in sorted(selected_items)},
        "selected_cards": {key: selected_cards[key] for key in sorted(selected_cards)},
        "runtime_config_sha256": runtime_config_sha256,
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
            "max_child_holdout": args.max_child_holdout,
            "max_parent_reference": args.max_parent_reference,
            "variant_order": "deterministic_shuffle_v1",
        },
        "protocol": "paired_by_sample_not_same_generation_seed",
    })


def reusable_cached_result(cached: Any, expected_fingerprint: str) -> bool:
    return (
        isinstance(cached, dict)
        and cached.get("schema_version") == HIERARCHY_SCHEMA_VERSION
        and cached.get("validation_fingerprint") == expected_fingerprint
    )


def _taxonomy_types_by_id(taxonomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    types = list(taxonomy.get("types") or [])
    by_id = {str(item.get("type_id")): item for item in types if item.get("type_id")}
    if len(by_id) != len(types):
        raise SystemExit("taxonomy contains missing or duplicate type_id values")
    return by_id


def _validate_pair_ids(
    pairs: list[dict[str, str | None]],
    type_items: dict[str, dict[str, Any]],
) -> None:
    for pair in pairs:
        for role in ("parent_type_id", "child_type_id", "control_type_id"):
            type_id = pair.get(role)
            if type_id and type_id not in type_items:
                raise SystemExit(f"{role} {type_id!r} is not present in taxonomy")


def _load_runtime(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any, Any, dict[str, dict[str, Any]], str]:
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
    return cfg, adapter, dataloader, items_by_key, initial_skill


def _cohorts_for_pair(
    *,
    pair: dict[str, str | None],
    type_items: dict[str, dict[str, Any]],
    cards_by_key: dict[str, dict[str, Any]],
    items_by_key: dict[str, dict[str, Any]],
    seed: int,
    max_child_holdout: int,
    max_parent_reference: int,
) -> dict[str, list[str]]:
    parent = type_items[str(pair["parent_type_id"])]
    child = type_items[str(pair["child_type_id"])]
    child_candidates = [
        key for key in (child.get("holdout_member_keys") or [])
        if key in cards_by_key and key in items_by_key
    ]
    child_all_keys = set(child.get("holdout_member_keys") or [])
    child_all_keys.update(child.get("fit_member_keys") or [])
    child_all_keys.update(child.get("member_keys") or [])
    parent_candidates = [
        key for key in (parent.get("holdout_member_keys") or [])
        if key not in child_all_keys and key in cards_by_key and key in items_by_key
    ]
    pair_seed = seed + int(stable_hash(pair)[:8], 16)
    return {
        "child_holdout": stratified_select(
            child_candidates,
            cards_by_key=cards_by_key,
            limit=max_child_holdout,
            seed=pair_seed,
        ),
        "parent_reference": stratified_select(
            parent_candidates,
            cards_by_key=cards_by_key,
            limit=max_parent_reference,
            seed=pair_seed + 1,
        ),
    }


def _run_pair(
    *,
    pair: dict[str, str | None],
    type_items: dict[str, dict[str, Any]],
    cards_by_key: dict[str, dict[str, Any]],
    items_by_key: dict[str, dict[str, Any]],
    initial_skill: str,
    runtime_config_sha256: str,
    adapter: Any,
    dataloader: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parent = type_items[str(pair["parent_type_id"])]
    child = type_items[str(pair["child_type_id"])]
    control_id = pair.get("control_type_id")
    control = type_items[str(control_id)] if control_id else None
    cohorts = _cohorts_for_pair(
        pair=pair,
        type_items=type_items,
        cards_by_key=cards_by_key,
        items_by_key=items_by_key,
        seed=args.seed,
        max_child_holdout=args.max_child_holdout,
        max_parent_reference=args.max_parent_reference,
    )
    selected_keys = sorted(set(key for keys in cohorts.values() for key in keys))
    fingerprint = hierarchy_fingerprint(
        pair=pair,
        type_items=type_items,
        initial_skill=initial_skill,
        cohorts=cohorts,
        selected_items={key: items_by_key[key] for key in selected_keys},
        selected_cards={key: cards_by_key[key] for key in selected_keys},
        runtime_config_sha256=runtime_config_sha256,
        args=args,
    )
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / pair_slug(pair) / "hierarchy_result.json"
    if result_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if reusable_cached_result(cached, fingerprint):
            print(f"[resume] {pair_slug(pair)}")
            return cached
        print(f"[stale] {pair_slug(pair)} cache does not match current inputs")

    variants = build_variants(initial_skill, parent, child, control)
    q_by_cohort: dict[str, dict[str, dict[str, float]]] = {}
    execution_orders: list[dict[str, Any]] = []
    for cohort_name, cohort_keys in cohorts.items():
        q_by_variant: dict[str, dict[str, float]] = {
            variant: {} for variant in variants
        }
        grouped_by_split: dict[str, list[str]] = defaultdict(list)
        for key in cohort_keys:
            grouped_by_split[key.split("::", 1)[0]].append(key)
        for split, split_keys in sorted(grouped_by_split.items()):
            split_keys.sort()
            for start in range(0, len(split_keys), args.batch_size):
                chunk_index = start // args.batch_size
                batch_keys = split_keys[start:start + args.batch_size]
                batch_seed = args.seed + (
                    int(stable_hash({
                        "pair": pair,
                        "cohort": cohort_name,
                        "split": split,
                        "chunk_index": chunk_index,
                    })[:8], 16)
                    % 1_000_000_000
                )
                order = deterministic_variant_order(
                    list(variants),
                    seed=args.seed,
                    pair_name=pair_slug(pair),
                    cohort_name=cohort_name,
                    split=split,
                    chunk_index=chunk_index,
                )
                execution_orders.append({
                    "cohort": cohort_name,
                    "split": split,
                    "chunk_index": chunk_index,
                    "sample_keys": batch_keys,
                    "batch_spec_seed": batch_seed,
                    "variant_order": order,
                })
                chunk_root = (
                    output_dir / pair_slug(pair) / "paired_rollouts"
                    / fingerprint[:16] / cohort_name / split / f"chunk_{chunk_index:04d}"
                )
                for variant in order:
                    repeated = repeated_rollout(
                        adapter=adapter,
                        dataloader=dataloader,
                        items=[items_by_key[key] for key in batch_keys],
                        split=split,
                        seed=batch_seed,
                        repeats=args.repeats,
                        skill_content=variants[variant],
                        chunk_dir=chunk_root / variant,
                    )
                    groups = _group_repeated_rollouts(
                        repeated, include_trajectories=False
                    )
                    for key in batch_keys:
                        sample_id = key.split("::", 1)[1]
                        if sample_id not in groups:
                            raise RuntimeError(
                                f"missing {variant} rollout group for {key}"
                            )
                        q_by_variant[variant][key] = _success_rate(groups[sample_id])
        q_by_cohort[cohort_name] = q_by_variant

    summaries = {
        cohort: summarize_cohort(cohorts[cohort], q_by_cohort[cohort])
        for cohort in cohorts
    }
    decision_metrics = hierarchy_decision_metrics(summaries)
    result = {
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "validation_fingerprint": fingerprint,
        "pair": pair,
        "parent_revision_type": parent.get("revision_type"),
        "child_revision_type": child.get("revision_type"),
        "control_revision_type": control.get("revision_type") if control else None,
        "cohorts": cohorts,
        "cohort_strata": {
            cohort: {
                f"{split}|{status}": sum(
                    card_stratum(key, cards_by_key) == (split, status)
                    for key in keys
                )
                for split, status in sorted({
                    card_stratum(key, cards_by_key) for key in keys
                })
            }
            for cohort, keys in cohorts.items()
        },
        "summaries": summaries,
        "decision_metrics": decision_metrics,
        "decision_reference_thresholds": {
            "minimum_in_scope_advantage": 0.05,
            "maximum_outside_scope_drop": 0.02,
            "note": (
                "Reference values only. No automatic relation verdict is emitted; "
                "all raw contrasts are retained."
            ),
        },
        "execution_orders": execution_orders,
        "variant_skill_sha256": {
            variant: stable_hash(skill) for variant, skill in variants.items()
        },
        "validation_protocol": {
            "paired_by_sample": True,
            "same_selected_samples_for_all_variants": True,
            "common_generation_random_numbers": False,
            "generation_seed_controlled": False,
            "batch_spec_seed_recorded_but_qwen_chat_does_not_honor_generation_seed": True,
            "variant_execution_order": "deterministically_shuffled_per_pair_cohort_split_chunk",
            "winner_is_descriptive_raw_accuracy_only": True,
        },
    }
    write_json(result_path, result)
    print(
        f"[validate] {pair_slug(pair)} "
        f"child_n={len(cohorts['child_holdout'])} "
        f"parent_ref_n={len(cohorts['parent_reference'])}"
    )
    return result


def _write_reports(output_dir: Path, results: list[dict[str, Any]]) -> None:
    aggregate = {
        "schema_version": HIERARCHY_SCHEMA_VERSION,
        "pairs": results,
        "interpretation": (
            "Metric winners are descriptive raw accuracies on selected cohorts; "
            "they are not statistical or causal acceptance decisions."
        ),
    }
    write_json(output_dir / "hierarchy_validation.json", aggregate)
    lines = [
        "# SearchQA blind hierarchy validation",
        "",
        "All comparisons are paired by sample, not by a controlled Qwen generation seed.",
        "Metric winners below are descriptive raw accuracies only.",
        "",
    ]
    for result in results:
        pair = result["pair"]
        lines.extend([
            f"## {pair['parent_type_id']} → {pair['child_type_id']}",
            "",
            f"- Parent: `{result['parent_revision_type']}`",
            f"- Child: `{result['child_revision_type']}`",
            f"- Unrelated control: `{result['control_revision_type'] or 'none'}`",
            "",
            "| Cohort | Variant | N | Accuracy | Δ vs initial | ↑/= /↓ vs initial |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for cohort_name in ("child_holdout", "parent_reference"):
            summary = result["summaries"][cohort_name]
            for variant, row in summary["variants"].items():
                lines.append(
                    f"| {cohort_name} | {variant} | {row['n']} | "
                    f"{row['accuracy']:.4f} | {row['delta_vs_initial']:+.4f} | "
                    f"{row['improved_vs_initial']}/{row['unchanged_vs_initial']}/"
                    f"{row['regressed_vs_initial']} |"
                )
            lines.append(
                f"| {cohort_name} | **raw metric winner(s)** | {summary['n']} | "
                f"**{summary['best_accuracy']:.4f}** | — | "
                f"{', '.join(summary['metric_winners'])} |"
            )
        child_control = result["summaries"]["child_holdout"].get(
            "child_vs_unrelated_control"
        )
        if child_control:
            lines.extend([
                "",
                (
                    "Child minus unrelated-control accuracy on child holdout: "
                    f"`{child_control['child_minus_unrelated_control']:+.4f}`."
                ),
            ])
        lines.extend([
            "",
            "| Raw hierarchy contrast | Value |",
            "|---|---:|",
        ])
        for name, value in result["decision_metrics"].items():
            rendered = "n/a" if value is None else f"{value:+.4f}"
            lines.append(f"| {name} | {rendered} |")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hierarchy_validation.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    taxonomy = json.loads(args.taxonomy.expanduser().resolve().read_text(encoding="utf-8"))
    type_items = _taxonomy_types_by_id(taxonomy)
    _validate_pair_ids(args.pair_specs, type_items)
    cards = load_unique_cards(args.cards)
    cards_by_key = {str(card["sample_key"]): card for card in cards}
    print("============================================================")
    print("  SearchQA blind hierarchy validation")
    print("============================================================")
    print(f"  pairs:                {len(args.pair_specs)}")
    print(f"  cards:                {len(cards)}")
    print(f"  repeats:              {args.repeats}")
    print(f"  max child holdout:    {args.max_child_holdout}")
    print(f"  max parent reference: {args.max_parent_reference}")
    print(f"  target:               {args.target_model} @ {args.target_base_url}")
    print(f"  output:               {args.output_dir}")
    for pair in args.pair_specs:
        child = type_items[str(pair["child_type_id"])]
        parent = type_items[str(pair["parent_type_id"])]
        print(
            f"  - {pair_slug(pair)}: "
            f"child candidates={len(child.get('holdout_member_keys') or [])}, "
            f"parent candidates={len(parent.get('holdout_member_keys') or [])}"
        )
    print("============================================================")
    if args.dry_run:
        return

    cfg, adapter, dataloader, items_by_key, initial_skill = _load_runtime(args)
    runtime_config_sha256 = stable_hash(cfg)
    results = [
        _run_pair(
            pair=pair,
            type_items=type_items,
            cards_by_key=cards_by_key,
            items_by_key=items_by_key,
            initial_skill=initial_skill,
            runtime_config_sha256=runtime_config_sha256,
            adapter=adapter,
            dataloader=dataloader,
            args=args,
        )
        for pair in args.pair_specs
    ]
    output_dir = args.output_dir.expanduser().resolve()
    _write_reports(output_dir, results)
    print(f"[done] pairs={len(results)} report={output_dir / 'hierarchy_validation.md'}")


if __name__ == "__main__":
    main()
