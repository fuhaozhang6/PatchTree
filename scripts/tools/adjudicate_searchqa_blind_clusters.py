#!/usr/bin/env python3
"""Adjudicate, merge, and name blind SearchQA mechanism clusters."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.model import (  # noqa: E402
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from skillopt.utils import extract_json  # noqa: E402
from scripts.tools.searchqa_blind_common import (  # noqa: E402
    load_unique_cards,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)

CLUSTER_SYSTEM = """\
You are validating one mechanically proposed cluster of type-free repair
mechanism cards. No predefined taxonomy is available.

Decide whether the supplied fit cards support one shared operational repair.
Do not group by topic, named entity, answer, or wording alone. A valid cluster
must share an interchangeable missing operation, applicability condition, and
boundary. Be conservative when all evidence comes from all-failed attempts.

Only after deciding coherence, assign a short snake_case revision_type name.
The name summarizes the discovered cluster; it must not determine membership.
The shared patch must be generic and must not contain row-specific entities,
answers, quotations, dates, or examples.

Return one JSON object:
{
  "accepted": true|false,
  "reasoning": "...",
  "revision_type": "short_snake_case; empty when rejected",
  "definition": "shared failure and missing operation",
  "repair_signature": "shared operational mechanism",
  "condition": "when the repair applies",
  "boundary": "when it must not apply",
  "shared_patch": {"op":"append","content":"one generic Skill rule"},
  "suspected_submechanisms": ["..."],
  "confidence": "high|medium|low"
}
"""

GLOBAL_SYSTEM = """\
You are reconciling independently discovered repair clusters. You receive only
blind cluster summaries and measured support, not the old seeded taxonomy.

Merge clusters only when the same operational patch, trigger condition, and
boundary can safely apply to both. Similar topics or similar wording are not
enough. Keep distinct abstractions separate when their executable corrections
differ. Reject incoherent candidates instead of forcing coverage.

Return one JSON object:
{
  "types": [
    {
      "source_cluster_ids": ["C001"],
      "revision_type": "short_snake_case",
      "definition": "...",
      "repair_signature": "...",
      "condition": "...",
      "boundary": "...",
      "shared_patch": {"op":"append","content":"one generic Skill rule"},
      "merge_reason": "..."
    }
  ],
  "rejected_cluster_ids": ["..."],
  "taxonomy_rationale": "..."
}
"""

ADJUDICATION_CALL_CONFIG = {
    "max_completion_tokens": 8000,
    "retries": 3,
}
ADJUDICATION_CACHE_VERSION = "searchqa_blind_adjudication_v2"
UNSTABLE_EVIDENCE_WEIGHT = 1.5
UNSTABLE_EVIDENCE_MAX_FRACTION = 2 / 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", action="append", required=True, type=Path)
    parser.add_argument(
        "--seeded-labels",
        action="append",
        type=Path,
        default=[],
        help="Optional post-hoc-only mapping; read after all discovery LLM calls.",
    )
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--optimizer-model",
        default=os.environ.get("OPTIMIZER_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-fit-cards", type=int, default=36)
    parser.add_argument("--skip-global-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.drafts < 1 or args.workers < 1 or args.max_fit_cards < 3:
        parser.error("--drafts/--workers must be >=1 and --max-fit-cards >=3")
    return args


def configure_optimizer(model: str) -> None:
    configure_azure_openai(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or None,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION") or None,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY") or None,
        auth_mode=os.environ.get("AZURE_OPENAI_AUTH_MODE") or None,
        optimizer_endpoint=os.environ.get("OPTIMIZER_AZURE_OPENAI_ENDPOINT") or None,
        optimizer_api_version=os.environ.get("OPTIMIZER_AZURE_OPENAI_API_VERSION") or None,
        optimizer_api_key=os.environ.get("OPTIMIZER_AZURE_OPENAI_API_KEY") or None,
        optimizer_auth_mode=os.environ.get("OPTIMIZER_AZURE_OPENAI_AUTH_MODE") or None,
    )
    set_optimizer_backend("openai_chat")
    set_optimizer_deployment(model)


def blind_card(card: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields that were produced without the seeded taxonomy."""
    return {
        key: card.get(key)
        for key in (
            "sample_key", "split", "outcome_status", "q_i", "observed_failure",
            "successful_contrast", "missing_operation", "repair_signature",
            "mechanism_keywords", "condition", "boundary", "candidate_patch",
            "confidence",
        )
    }


def select_fit_cards(
    cluster: dict[str, Any],
    cards_by_key: dict[str, dict[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    candidates = [
        cards_by_key[key]
        for key in cluster.get("fit_member_keys", [])
        if key in cards_by_key
    ]
    candidates.sort(key=lambda card: str(card.get("sample_key") or ""))
    if len(candidates) <= maximum:
        return [blind_card(card) for card in candidates]

    # The adjudicator must see the evidence composition it is deciding for.
    # First cover as many outcome_status x split strata as possible, then use
    # deterministic weighted apportionment. Unstable cards receive a modest
    # contrast-evidence boost, but cannot consume more than two thirds of the
    # sample while any non-unstable evidence remains available.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for card in candidates:
        key = (
            str(card.get("outcome_status") or "unknown"),
            str(card.get("split") or "unknown"),
        )
        buckets.setdefault(key, []).append(card)
    strata = sorted(
        buckets,
        key=lambda key: (
            0 if key[0] == "unstable" else 1,
            -len(buckets[key]),
            key,
        ),
    )
    selected_strata = strata[:maximum]
    allocations = {key: 1 for key in selected_strata}
    unstable_cap = maximum
    if any(key[0] == "unstable" for key in selected_strata) and any(
        key[0] != "unstable" for key in selected_strata
    ):
        unstable_cap = max(
            sum(key[0] == "unstable" for key in selected_strata),
            int(maximum * UNSTABLE_EVIDENCE_MAX_FRACTION),
        )

    def can_allocate(key: tuple[str, str]) -> bool:
        if allocations[key] >= len(buckets[key]):
            return False
        if key[0] != "unstable":
            return True
        return sum(
            count
            for bucket_key, count in allocations.items()
            if bucket_key[0] == "unstable"
        ) < unstable_cap

    while sum(allocations.values()) < maximum:
        available = [key for key in selected_strata if can_allocate(key)]
        if not available:
            # The cap prevents avoidable dominance; it must not under-fill the
            # prompt when the other strata simply contain too few cards.
            available = [
                key
                for key in selected_strata
                if allocations[key] < len(buckets[key])
            ]
        if not available:
            break
        # D'Hondt-style apportionment is deterministic, proportional to stratum
        # size, and gives unstable contrast evidence a bounded 1.5x weight.
        best = min(
            available,
            key=lambda key: (
                -(
                    len(buckets[key])
                    * (UNSTABLE_EVIDENCE_WEIGHT if key[0] == "unstable" else 1.0)
                    / (allocations[key] + 1)
                ),
                0 if key[0] == "unstable" else 1,
                key,
            ),
        )
        allocations[best] += 1

    chosen: list[dict[str, Any]] = []
    for key in selected_strata:
        bucket = buckets[key]
        count = allocations[key]
        # Midpoint sampling spreads representatives across the stable key order
        # rather than taking a lexicographic prefix from each stratum.
        positions = [
            min(int((index + 0.5) * len(bucket) / count), len(bucket) - 1)
            for index in range(count)
        ]
        chosen.extend(bucket[index] for index in positions)
    chosen.sort(key=lambda card: (
        0 if card.get("outcome_status") == "unstable" else 1,
        str(card.get("outcome_status") or ""),
        str(card.get("split") or ""),
        str(card.get("sample_key") or ""),
    ))
    return [blind_card(card) for card in chosen]


def call_json(system: str, user: str, stage: str) -> dict[str, Any]:
    response, _ = chat_optimizer(
        system=system,
        user=user,
        **ADJUDICATION_CALL_CONFIG,
        stage=stage,
    )
    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{stage}: expected a JSON object")
    return parsed


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text[:80]


def valid_patch(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("op") == "append"
        and bool(str(value.get("content") or "").strip())
    )


def validate_candidate(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("accepted") is not True:
        return {
            "accepted": False,
            "reasoning": str(value.get("reasoning") or ""),
            "suspected_submechanisms": value.get("suspected_submechanisms") or [],
            "confidence": str(value.get("confidence") or "low"),
        }
    required = ("definition", "repair_signature", "condition", "boundary")
    if any(not str(value.get(key) or "").strip() for key in required):
        raise ValueError("accepted candidate is missing required text")
    if not slug(value.get("revision_type")) or not valid_patch(value.get("shared_patch")):
        raise ValueError("accepted candidate has invalid name or patch")
    return {
        "accepted": True,
        "reasoning": str(value.get("reasoning") or ""),
        "revision_type": slug(value["revision_type"]),
        **{key: str(value[key]) for key in required},
        "shared_patch": {
            "op": "append",
            "content": str(value["shared_patch"]["content"]),
        },
        "suspected_submechanisms": value.get("suspected_submechanisms") or [],
        "confidence": str(value.get("confidence") or "low"),
    }


def adjudicate_cluster(
    cluster: dict[str, Any],
    evidence: list[dict[str, Any]],
    drafts_count: int,
    cache_dir: Path,
    optimizer_model: str,
) -> dict[str, Any]:
    cluster_id = str(cluster["cluster_id"])
    base_user = (
        f"Candidate cluster: {cluster_id}\n"
        f"Measured support: {cluster['support_count']}\n"
        f"Origin: {cluster['origin']}\n"
        f"Split counts: {json.dumps(cluster['split_counts'])}\n"
        f"Outcome counts: {json.dumps(cluster['outcome_counts'])}\n\n"
        f"## Fit cards only\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    draft_suffixes = [
        f"\n\nProduce independent draft {index + 1}."
        for index in range(drafts_count)
    ]
    reconcile_suffix = (
        "\n\n## Independent drafts\n{drafts}"
        "\n\nReconcile against the fit cards. Evidence overrides either draft."
    )
    cache_fingerprint = {
        "cache_version": ADJUDICATION_CACHE_VERSION,
        "optimizer_model": optimizer_model,
        "drafts_count": drafts_count,
        "call_config": ADJUDICATION_CALL_CONFIG,
        "system_prompt": CLUSTER_SYSTEM,
        "base_user_prompt": base_user,
        "draft_prompt_suffixes": draft_suffixes,
        "reconcile_prompt_template": reconcile_suffix,
        "evidence": evidence,
    }
    cache_path = cache_dir / f"{cluster_id}_{stable_hash(cache_fingerprint)}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    drafts = [
        call_json(
            CLUSTER_SYSTEM,
            base_user + suffix,
            f"searchqa_blind_cluster_{cluster_id}_draft{index + 1}",
        )
        for index, suffix in enumerate(draft_suffixes)
    ]
    reconciled = call_json(
        CLUSTER_SYSTEM,
        base_user
        + reconcile_suffix.format(
            drafts=json.dumps(drafts, ensure_ascii=False, indent=2)
        ),
        f"searchqa_blind_cluster_{cluster_id}_reconcile",
    )
    result = {
        "cluster_id": cluster_id,
        **validate_candidate(reconciled),
        "draft_count": drafts_count,
        "fit_evidence_keys": [str(card["sample_key"]) for card in evidence],
    }
    write_json(cache_path, result)
    return result


def fallback_types(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "types": [
            {
                "source_cluster_ids": [candidate["cluster_id"]],
                **{
                    key: candidate[key]
                    for key in (
                        "revision_type", "definition", "repair_signature",
                        "condition", "boundary", "shared_patch",
                    )
                },
                "merge_reason": "No global merge requested.",
            }
            for candidate in candidates if candidate.get("accepted")
        ],
        "rejected_cluster_ids": [
            candidate["cluster_id"] for candidate in candidates if not candidate.get("accepted")
        ],
        "taxonomy_rationale": "Per-cluster adjudication without cross-cluster merging.",
    }


def validate_global(
    value: dict[str, Any],
    candidate_ids: set[str],
) -> dict[str, Any]:
    types = value.get("types")
    if not isinstance(types, list):
        raise ValueError("global result has no types list")
    seen: set[str] = set()
    clean_types = []
    for item in types:
        sources = [str(value) for value in (item.get("source_cluster_ids") or [])]
        if not sources or any(source not in candidate_ids for source in sources):
            raise ValueError("global result cites unknown/empty source clusters")
        if seen.intersection(sources):
            raise ValueError("global result reuses a source cluster")
        seen.update(sources)
        if not slug(item.get("revision_type")) or not valid_patch(item.get("shared_patch")):
            raise ValueError("global type has invalid name/patch")
        clean_types.append({
            "source_cluster_ids": sources,
            "revision_type": slug(item["revision_type"]),
            "definition": str(item.get("definition") or ""),
            "repair_signature": str(item.get("repair_signature") or ""),
            "condition": str(item.get("condition") or ""),
            "boundary": str(item.get("boundary") or ""),
            "shared_patch": {
                "op": "append",
                "content": str(item["shared_patch"]["content"]),
            },
            "merge_reason": str(item.get("merge_reason") or ""),
        })
    rejected = {
        str(cluster_id)
        for cluster_id in (value.get("rejected_cluster_ids") or [])
        if str(cluster_id) in candidate_ids
    }
    rejected.update(candidate_ids - seen)
    return {
        "types": clean_types,
        "rejected_cluster_ids": sorted(rejected),
        "taxonomy_rationale": str(value.get("taxonomy_rationale") or ""),
    }


def main() -> None:
    args = parse_args()
    cards = load_unique_cards(args.cards)
    cards_by_key = {str(card["sample_key"]): card for card in cards}
    cluster_payload = json.loads(args.clusters.expanduser().resolve().read_text(encoding="utf-8"))
    clusters = list(cluster_payload.get("clusters") or [])
    if not clusters:
        raise SystemExit(
            "candidate cluster file contains zero clusters; rerun stage 2 with "
            "the auto-threshold implementation before adjudication"
        )
    print("============================================================")
    print("  SearchQA blind cluster adjudication")
    print("============================================================")
    print(f"  cards/clusters: {len(cards)}/{len(clusters)}")
    print(f"  drafts:         {args.drafts} + reconciliation per cluster")
    print(f"  cluster workers:{args.workers}")
    print(f"  output:         {args.output_dir}")
    print("  old labels passed to adjudicator: false")
    print("============================================================")
    if args.dry_run:
        return
    configure_optimizer(args.optimizer_model)
    output_dir = args.output_dir.expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    work: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for cluster in clusters:
        evidence = select_fit_cards(cluster, cards_by_key, args.max_fit_cards)
        if len(evidence) < 3:
            candidates.append({
                "cluster_id": cluster["cluster_id"],
                "accepted": False,
                "reasoning": "Fewer than three fit cards.",
            })
        else:
            work.append((cluster, evidence))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                adjudicate_cluster,
                cluster,
                evidence,
                args.drafts,
                output_dir / "cache",
                args.optimizer_model,
            ): str(cluster["cluster_id"])
            for cluster, evidence in work
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            candidates.append(future.result())
            print(f"[adjudicate] {index}/{len(work)} {futures[future]}")
    candidates.sort(key=lambda item: str(item["cluster_id"]))
    write_json(output_dir / "cluster_adjudications.json", {"candidates": candidates})

    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    global_input = [
        {
            key: candidate.get(key)
            for key in (
                "cluster_id", "revision_type", "definition", "repair_signature",
                "condition", "boundary", "shared_patch", "confidence",
            )
        }
        for candidate in accepted
    ]
    if args.skip_global_merge or len(global_input) <= 1:
        global_result = fallback_types(candidates)
    else:
        raw_global = call_json(
            GLOBAL_SYSTEM,
            "## Blind cluster summaries\n"
            + json.dumps(global_input, ensure_ascii=False, indent=2),
            "searchqa_blind_global_reconcile",
        )
        global_result = validate_global(
            raw_global, {candidate["cluster_id"] for candidate in accepted}
        )

    # The old label mapping is intentionally loaded only after extraction,
    # clustering, per-cluster adjudication, naming, and global merging finish.
    old_revision_by_key: dict[str, str] = {}
    for path in args.seeded_labels:
        for row in read_jsonl(path.expanduser().resolve()):
            key = str(row.get("sample_key") or "")
            label = str(row.get("seeded_revision_type") or "")
            if key and label:
                old_revision_by_key[key] = label

    clusters_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    final_types = []
    assignment_rows = []
    for index, item in enumerate(global_result["types"], 1):
        source_clusters = [clusters_by_id[source] for source in item["source_cluster_ids"]]
        members = sorted({
            key for cluster in source_clusters for key in cluster["member_keys"]
        })
        fit_members = sorted({
            key for cluster in source_clusters for key in cluster["fit_member_keys"]
        })
        holdout_members = sorted({
            key for cluster in source_clusters for key in cluster["holdout_member_keys"]
        })
        boundary_members = sorted({
            key for cluster in source_clusters for key in cluster["boundary_member_keys"]
            if key not in members
        })
        split_counts = Counter(cards_by_key[key]["split"] for key in members)
        outcome_counts = Counter(cards_by_key[key]["outcome_status"] for key in members)
        old_counts = Counter(
            old_revision_by_key[key] for key in members if key in old_revision_by_key
        )
        type_id = f"R_SEARCH_{index:03d}"
        final = {
            "type_id": type_id,
            **item,
            "support_count": len(members),
            "split_counts": dict(sorted(split_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "member_keys": members,
            "fit_member_keys": fit_members,
            "holdout_member_keys": holdout_members,
            "boundary_member_keys": boundary_members,
            "posthoc_seeded_revision_type_counts": dict(old_counts.most_common()),
            "old_labels_used_during_discovery": False,
        }
        final_types.append(final)
        assignment_rows.extend({
            "sample_key": key,
            "type_id": type_id,
            "revision_type": item["revision_type"],
        } for key in members)
    final_payload = {
        "schema_version": "searchqa_blind_taxonomy_v1",
        "optimizer_model": args.optimizer_model,
        "old_labels_used_during_discovery": False,
        "n_cards": len(cards),
        "n_candidate_clusters": len(clusters),
        "n_final_types": len(final_types),
        "rejected_cluster_ids": global_result["rejected_cluster_ids"],
        "taxonomy_rationale": global_result["taxonomy_rationale"],
        "types": final_types,
    }
    write_json(output_dir / "blind_revision_taxonomy.json", final_payload)
    write_jsonl(
        output_dir / "blind_type_assignments.jsonl",
        sorted(assignment_rows, key=lambda row: (row["sample_key"], row["type_id"])),
    )
    lines = [
        "# SearchQA blind revision taxonomy",
        "",
        "Old seeded labels were hidden until the post-hoc comparison columns below.",
        "",
        "| Type | Name | Support | Unstable | Failure | Old-label comparison |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in final_types:
        old = ", ".join(
            f"{key}:{value}"
            for key, value in item["posthoc_seeded_revision_type_counts"].items()
        )
        lines.append(
            f"| {item['type_id']} | {item['revision_type']} | {item['support_count']} | "
            f"{item['outcome_counts'].get('unstable', 0)} | "
            f"{item['outcome_counts'].get('failure', 0)} | {old} |"
        )
    (output_dir / "blind_revision_taxonomy.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[done] final_types={len(final_types)} output={output_dir}")


if __name__ == "__main__":
    main()
