#!/usr/bin/env python3
"""Cluster blind SearchQA mechanism cards without using seeded type labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.searchqa_blind_common import (  # noqa: E402
    card_feature_text,
    centroid,
    cosine_dbscan_from_similarities,
    deterministic_partition,
    hashed_tfidf,
    load_unique_cards,
    member_counts,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--similarity-threshold", default="auto")
    parser.add_argument("--residual-similarity-threshold", default="auto")
    parser.add_argument("--assignment-threshold", default="auto")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=6)
    parser.add_argument("--dimensions", type=int, default=8192)
    parser.add_argument("--fit-fraction", type=float, default=0.60)
    parser.add_argument("--boundary-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("similarity_threshold", "residual_similarity_threshold", "assignment_threshold"):
        raw = str(getattr(args, name)).strip().lower()
        if raw == "auto":
            setattr(args, name, None)
            continue
        try:
            value = float(raw)
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} must be 'auto' or a number")
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
        setattr(args, name, value)
    if args.min_samples < 2 or args.min_cluster_size < args.min_samples:
        parser.error("require min-cluster-size >= min-samples >= 2")
    if not 0.1 <= args.fit_fraction <= 0.9:
        parser.error("--fit-fraction must be in [0.1,0.9]")
    return args


def dbscan_groups(
    labels: list[int],
    source_indices: list[int],
    min_cluster_size: int,
) -> tuple[list[list[int]], list[int]]:
    grouped: dict[int, list[int]] = {}
    noise: list[int] = []
    for local_index, label in enumerate(labels):
        global_index = source_indices[local_index]
        if label < 0:
            noise.append(global_index)
        else:
            grouped.setdefault(label, []).append(global_index)
    accepted = []
    for members in grouped.values():
        if len(members) >= min_cluster_size:
            accepted.append(sorted(members))
        else:
            noise.extend(members)
    accepted.sort(key=lambda values: (-len(values), values[0]))
    return accepted, sorted(noise)


def silhouette_for_groups(
    similarities: np.ndarray,
    local_groups: list[list[int]],
) -> float:
    if len(local_groups) < 2:
        return -1.0
    values: list[float] = []
    for group_index, members in enumerate(local_groups):
        for point in members:
            own = [index for index in members if index != point]
            if not own:
                continue
            a = float(np.mean(1.0 - similarities[point, own]))
            b = min(
                float(np.mean(1.0 - similarities[point, other]))
                for other_index, other in enumerate(local_groups)
                if other_index != group_index and other
            )
            denominator = max(a, b)
            values.append((b - a) / denominator if denominator else 0.0)
    return float(np.mean(values)) if values else -1.0


def threshold_candidates(similarities: np.ndarray) -> list[float]:
    if similarities.shape[0] < 2:
        return [0.1]
    off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    positive = off_diagonal[off_diagonal > 0]
    if positive.size == 0:
        return [0.05]
    quantiles = np.quantile(positive, np.linspace(0.45, 0.995, 48))
    fixed = np.linspace(0.05, min(0.45, float(positive.max())), 25)
    return sorted({
        round(float(value), 5)
        for value in np.concatenate([quantiles, fixed])
        if 0.01 <= float(value) <= 0.95
    })


def evaluate_threshold(
    similarities: np.ndarray,
    source_indices: list[int],
    threshold: float,
    min_samples: int,
    min_cluster_size: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    labels = cosine_dbscan_from_similarities(similarities, threshold, min_samples)
    global_groups, _ = dbscan_groups(labels, source_indices, min_cluster_size)
    global_to_local = {global_index: local for local, global_index in enumerate(source_indices)}
    local_groups = [
        [global_to_local[index] for index in members]
        for members in global_groups
    ]
    assigned = sum(len(group) for group in local_groups)
    total = max(len(source_indices), 1)
    coverage = assigned / total
    largest_total_share = max((len(group) for group in local_groups), default=0) / total
    largest_assigned_share = (
        max((len(group) for group in local_groups), default=0) / assigned
        if assigned else 1.0
    )
    silhouette = silhouette_for_groups(similarities, local_groups)
    cluster_count = len(local_groups)
    valid_shape = (
        cluster_count >= 2
        and coverage >= 0.20
        and largest_total_share <= 0.60
        and largest_assigned_share <= 0.75
    )
    score = (
        silhouette
        + 0.35 * coverage
        + 0.02 * min(cluster_count, 10)
        - 0.40 * max(0.0, largest_assigned_share - 0.50)
    )
    if not valid_shape:
        score -= 2.0
    return global_groups, {
        "threshold": threshold,
        "n_clusters": cluster_count,
        "n_assigned": assigned,
        "coverage": coverage,
        "largest_total_share": largest_total_share,
        "largest_assigned_share": largest_assigned_share,
        "silhouette": silhouette,
        "valid_shape": valid_shape,
        "score": score,
    }


def choose_threshold(
    similarities: np.ndarray,
    source_indices: list[int],
    requested: float | None,
    min_samples: int,
    min_cluster_size: int,
) -> tuple[list[list[int]], float, list[dict[str, Any]], dict[str, Any]]:
    candidates = [requested] if requested is not None else threshold_candidates(similarities)
    evaluated = [
        evaluate_threshold(
            similarities,
            source_indices,
            float(threshold),
            min_samples,
            min_cluster_size,
        )
        for threshold in candidates
    ]
    best_groups, best_metrics = max(
        evaluated,
        key=lambda item: (
            float(item[1]["score"]),
            float(item[1]["silhouette"]),
            float(item[1]["coverage"]),
            float(item[1]["threshold"]),
        ),
    )
    # Neighbour diagnostics explain why a fixed threshold did or did not work.
    if similarities.shape[0] > 1:
        without_self = similarities.copy()
        np.fill_diagonal(without_self, -1.0)
        ordered = np.sort(without_self, axis=1)[:, ::-1]
        neighbor_index = min(max(min_samples - 2, 0), ordered.shape[1] - 1)
        kth = ordered[:, neighbor_index]
        neighbor_quantiles = {
            str(q): float(np.quantile(kth, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        }
    else:
        neighbor_quantiles = {}
    sweep = sorted(
        [metrics for _, metrics in evaluated],
        key=lambda item: float(item["threshold"]),
    )
    return (
        best_groups,
        float(best_metrics["threshold"]),
        sweep,
        {"kth_neighbor_similarity_quantiles": neighbor_quantiles},
    )


def main() -> None:
    args = parse_args()
    cards = load_unique_cards(args.input)
    if not cards:
        raise SystemExit("no usable mechanism cards")
    cards_by_key = {str(card["sample_key"]): card for card in cards}
    matrix = hashed_tfidf(
        [card_feature_text(card) for card in cards],
        dimensions=args.dimensions,
    )
    unstable = [
        index for index, card in enumerate(cards)
        if card.get("outcome_status") == "unstable"
    ]
    failures = [
        index for index, card in enumerate(cards)
        if card.get("outcome_status") == "failure"
    ]
    if len(unstable) < args.min_cluster_size:
        raise SystemExit(
            f"only {len(unstable)} unstable cards; extract more evidence or lower "
            "--min-cluster-size"
        )
    unstable_similarities = matrix[unstable] @ matrix[unstable].T
    (
        groups,
        selected_similarity_threshold,
        core_threshold_sweep,
        core_similarity_diagnostics,
    ) = choose_threshold(
        unstable_similarities,
        unstable,
        args.similarity_threshold,
        args.min_samples,
        args.min_cluster_size,
    )

    # All-failure rows may expand contrast-derived clusters, but do not define
    # their initial centers.
    centers = [centroid(matrix, members) for members in groups]
    assignments: dict[int, int] = {}
    for group_index, members in enumerate(groups):
        assignments.update({index: group_index for index in members})
    residual: list[int] = []
    nearest_failure_similarities: list[float] = []
    if centers:
        nearest_failure_similarities = [
            max(float(matrix[index] @ center) for center in centers)
            for index in failures
        ]
    if args.assignment_threshold is None:
        selected_assignment_threshold = (
            float(np.quantile(nearest_failure_similarities, 0.70))
            if nearest_failure_similarities else 1.0
        )
        selected_assignment_threshold = min(max(selected_assignment_threshold, 0.08), 0.45)
    else:
        selected_assignment_threshold = args.assignment_threshold
    for index in failures:
        if not centers:
            residual.append(index)
            continue
        similarities = np.asarray([float(matrix[index] @ center) for center in centers])
        best = int(similarities.argmax())
        if float(similarities[best]) >= selected_assignment_threshold:
            groups[best].append(index)
            assignments[index] = best
        else:
            residual.append(index)

    # Repeated mechanisms seen only in all-failed rows become lower-evidence
    # residual clusters instead of being forced into a contrast cluster.
    residual_threshold_sweep: list[dict[str, Any]] = []
    residual_similarity_diagnostics: dict[str, Any] = {}
    selected_residual_threshold = (
        args.residual_similarity_threshold
        if args.residual_similarity_threshold is not None else 0.0
    )
    if len(residual) >= args.min_cluster_size:
        residual_similarities = matrix[residual] @ matrix[residual].T
        (
            residual_groups,
            selected_residual_threshold,
            residual_threshold_sweep,
            residual_similarity_diagnostics,
        ) = choose_threshold(
            residual_similarities,
            residual,
            args.residual_similarity_threshold,
            args.min_samples,
            args.min_cluster_size,
        )
        for members in residual_groups:
            group_index = len(groups)
            groups.append(members)
            assignments.update({index: group_index for index in members})

    unassigned = sorted(set(range(len(cards))) - set(assignments))
    clusters: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for group_index, raw_members in enumerate(groups):
        member_indices = sorted(set(raw_members))
        member_keys = sorted(str(cards[index]["sample_key"]) for index in member_indices)
        cluster_id = f"C{group_index + 1:03d}"
        fit_keys, holdout_keys = deterministic_partition(
            member_keys, args.fit_fraction, args.seed + group_index
        )
        center = centroid(matrix, member_indices)
        member_set = set(member_indices)
        outside = sorted(
            (
                (float(matrix[index] @ center), str(cards[index]["sample_key"]))
                for index in range(len(cards))
                if index not in member_set
            ),
            key=lambda item: (-item[0], item[1]),
        )
        origin = (
            "contrast_core"
            if any(cards[index].get("outcome_status") == "unstable" for index in member_indices)
            else "failure_residual"
        )
        clusters.append({
            "cluster_id": cluster_id,
            "origin": origin,
            **member_counts(member_keys, cards_by_key),
            "member_keys": member_keys,
            "fit_member_keys": fit_keys,
            "holdout_member_keys": holdout_keys,
            "boundary_member_keys": [key for _, key in outside[:args.boundary_size]],
            "seeded_labels_used_for_clustering": False,
        })
        assignment_rows.extend({
            "sample_key": key,
            "cluster_id": cluster_id,
            "assignment": "member",
            "origin": origin,
        } for key in member_keys)
    assignment_rows.extend({
        "sample_key": str(cards[index]["sample_key"]),
        "cluster_id": None,
        "assignment": "unassigned",
        "origin": "noise",
    } for index in unassigned)
    assignment_rows.sort(key=lambda row: row["sample_key"])

    output_dir = args.output_dir.expanduser().resolve()
    payload = {
        "schema_version": "searchqa_blind_clusters_v1",
        "method": {
            "vectorizer": "hashed_word_1_2gram_tfidf",
            "clusterer": "cosine_dbscan",
            "similarity_threshold_requested": (
                "auto" if args.similarity_threshold is None else args.similarity_threshold
            ),
            "similarity_threshold_selected": selected_similarity_threshold,
            "residual_similarity_threshold_requested": (
                "auto"
                if args.residual_similarity_threshold is None
                else args.residual_similarity_threshold
            ),
            "residual_similarity_threshold_selected": selected_residual_threshold,
            "assignment_threshold_requested": (
                "auto" if args.assignment_threshold is None else args.assignment_threshold
            ),
            "assignment_threshold_selected": selected_assignment_threshold,
            "min_samples": args.min_samples,
            "min_cluster_size": args.min_cluster_size,
            "fit_fraction": args.fit_fraction,
            "seed": args.seed,
            "seeded_labels_used": False,
        },
        "n_cards": len(cards),
        "n_unstable_cards": len(unstable),
        "n_failure_cards": len(failures),
        "n_clusters": len(clusters),
        "n_unassigned": len(unassigned),
        "threshold_diagnostics": {
            "contrast_core": {
                **core_similarity_diagnostics,
                "sweep": core_threshold_sweep,
            },
            "failure_assignment": {
                "nearest_centroid_quantiles": (
                    {
                        str(q): float(np.quantile(nearest_failure_similarities, q))
                        for q in (0.10, 0.25, 0.50, 0.70, 0.90)
                    }
                    if nearest_failure_similarities else {}
                ),
            },
            "failure_residual": {
                **residual_similarity_diagnostics,
                "sweep": residual_threshold_sweep,
            },
        },
        "clusters": clusters,
    }
    write_json(output_dir / "candidate_clusters.json", payload)
    write_jsonl(output_dir / "assignments.jsonl", assignment_rows)
    lines = [
        "# SearchQA blind candidate clusters",
        "",
        "Seeded question/revision labels were not used in extraction or clustering.",
        "",
        f"- cards: {len(cards)}",
        f"- unstable contrast cards: {len(unstable)}",
        f"- all-failure cards: {len(failures)}",
        f"- selected contrast threshold: {selected_similarity_threshold:.5f}",
        f"- selected failure assignment threshold: {selected_assignment_threshold:.5f}",
        f"- selected residual threshold: {selected_residual_threshold:.5f}",
        f"- clusters: {len(clusters)}",
        f"- unassigned/noise: {len(unassigned)}",
        "",
        "| Cluster | Origin | Support | Unstable | Failure | Splits |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cluster in clusters:
        outcomes = cluster["outcome_counts"]
        splits = ", ".join(f"{key}:{value}" for key, value in cluster["split_counts"].items())
        lines.append(
            f"| {cluster['cluster_id']} | {cluster['origin']} | "
            f"{cluster['support_count']} | {outcomes.get('unstable', 0)} | "
            f"{outcomes.get('failure', 0)} | {splits} |"
        )
    (output_dir / "clustering_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"[done] cards={len(cards)} unstable={len(unstable)} "
        f"threshold={selected_similarity_threshold:.5f} "
        f"clusters={len(clusters)} unassigned={len(unassigned)}"
    )


if __name__ == "__main__":
    main()
