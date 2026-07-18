#!/usr/bin/env python3
"""Merge sharded observed-taxonomy audits and rank few-shot evidence.

Only root ``sample_taxonomy.jsonl`` files are consumed; chunk-level copies are
ignored. Duplicate dataset/split/sample keys must be byte-for-byte equivalent,
otherwise the program stops rather than mixing incompatible runs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Audit directory or root sample_taxonomy.jsonl; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-datasets",
        default="",
        help="Whitespace/comma-separated env names that must all be present.",
    )
    parser.add_argument("--max-few-shots", type=int, default=8)
    parser.add_argument("--max-evidence-per-pair", type=int, default=3)
    args = parser.parse_args()
    if args.max_few_shots < 1 or args.max_evidence_per_pair < 1:
        parser.error("few-shot limits must be positive")
    return args


def taxonomy_files(inputs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.name != "sample_taxonomy.jsonl":
                raise ValueError(f"expected sample_taxonomy.jsonl, got: {path}")
            files.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in path.rglob("sample_taxonomy.jsonl"):
            if "chunks" not in candidate.parts:
                files.add(candidate.resolve())
    if not files:
        raise FileNotFoundError("no root sample_taxonomy.jsonl files found")
    return sorted(files)


def read_rows(files: Iterable[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    origins: dict[tuple[str, str, str], Path] = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row.get("env") or ""),
                    str(row.get("split") or ""),
                    str(row.get("id") or ""),
                )
                if not all(key):
                    raise ValueError(f"{path}:{line_no}: missing env/split/id")
                previous = by_key.get(key)
                if previous is not None and previous != row:
                    raise ValueError(
                        "conflicting duplicate sample "
                        f"{key}: {origins[key]} versus {path}"
                    )
                by_key[key] = row
                origins[key] = path
    return [by_key[key] for key in sorted(by_key)]


def validate_manifests(
    files: list[Path],
    rows: list[dict[str, Any]],
    expected_datasets: set[str],
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    shards: dict[tuple[str, int], set[int]] = defaultdict(set)
    for path in files:
        summary_path = path.parent / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing audit summary beside {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        env = str(summary.get("env") or "")
        shard_count = int(summary.get("shard_count", 1) or 1)
        shard_index = int(summary.get("shard_index", 0) or 0)
        line_count = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        expected = int(summary.get("expected_samples", -1))
        if not summary.get("complete") or expected < 0 or line_count != expected:
            raise ValueError(
                f"incomplete audit {path}: lines={line_count} "
                f"expected={expected} complete={summary.get('complete')}"
            )
        shard_key = (env, shard_count)
        if shard_index in shards[shard_key]:
            raise ValueError(f"duplicate shard {shard_index}/{shard_count} for {env}")
        shards[shard_key].add(shard_index)
        manifests.append({
            "path": str(path),
            "env": env,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "n_samples": line_count,
            "skill_hash": summary.get("skill_hash"),
            "repeats": summary.get("repeats"),
        })

    present = {manifest["env"] for manifest in manifests}
    missing = sorted(expected_datasets - present)
    if missing:
        raise ValueError(f"missing expected datasets: {', '.join(missing)}")
    for (env, shard_count), indices in shards.items():
        expected_indices = set(range(shard_count))
        if indices != expected_indices:
            raise ValueError(
                f"incomplete shard coverage for {env}: "
                f"got={sorted(indices)} expected={sorted(expected_indices)}"
            )
    for env in present:
        shard_counts = {
            shard_count
            for candidate_env, shard_count in shards
            if candidate_env == env
        }
        if len(shard_counts) != 1:
            raise ValueError(f"{env} mixes shard-count schemes: {sorted(shard_counts)}")

    # A fixed target, Skill, and repeat count are required within each dataset.
    for env in present:
        env_rows = [row for row in rows if str(row["env"]) == env]
        for field in (
            "target_model",
            "skill_hash",
            "repeats_requested",
            "taxonomy_schema_version",
        ):
            values = {str(row.get(field) or "unknown") for row in env_rows}
            if len(values) != 1 or "unknown" in values:
                raise ValueError(f"{env} has incompatible {field} values: {sorted(values)}")
    return manifests


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evidence_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Mixed success/failure repeats provide the cleanest contrastive evidence.
    return (
        0 if row.get("outcome_status") == "unstable" else 1,
        abs(float(row.get("q_i") or 0.0) - 0.5),
        -len(str(row.get("condition") or "")),
        str(row.get("split") or ""),
        str(row.get("id") or ""),
    )


def generic_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": row.get("question_type"),
        "revision_type": row.get("revision_type"),
        "repair_signature": row.get("repair_signature"),
        "condition": row.get("condition"),
        "boundary": row.get("boundary"),
        "patch": row.get("patch"),
    }


def pair_records(
    rows: list[dict[str, Any]],
    max_evidence: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("has_reusable_revision") and row.get("revision_type"):
            grouped[(str(row["question_type"]), str(row["revision_type"]))].append(row)

    records: list[dict[str, Any]] = []
    for (question_type, revision_type), group in grouped.items():
        ranked = sorted(group, key=evidence_sort_key)
        splits = sorted({str(row["split"]) for row in group})
        unstable = sum(row.get("outcome_status") == "unstable" for row in group)
        failure = sum(row.get("outcome_status") == "failure" for row in group)
        support = len(group)
        # Support is primary; contrastive and cross-split evidence break ties.
        score = support + 2.0 * unstable + 0.75 * len(splits)
        if support >= 3 and (len(splits) >= 2 or unstable >= 2):
            evidence_grade = "A"
        elif support >= 2:
            evidence_grade = "B"
        else:
            evidence_grade = "C"
        records.append({
            "question_type": question_type,
            "revision_type": revision_type,
            "support_count": support,
            "unstable_count": unstable,
            "failure_count": failure,
            "split_support": splits,
            "selection_score": score,
            "evidence_grade": evidence_grade,
            "evidence": [
                {
                    "split": row.get("split"),
                    "id": row.get("id"),
                    "q_i": row.get("q_i"),
                    "outcome_status": row.get("outcome_status"),
                    **generic_example(row),
                }
                for row in ranked[:max_evidence]
            ],
        })
    return sorted(
        records,
        key=lambda row: (
            -float(row["selection_score"]),
            -int(row["support_count"]),
            str(row["question_type"]),
            str(row["revision_type"]),
        ),
    )


def choose_shortlist(
    pairs: list[dict[str, Any]],
    max_few_shots: int,
) -> list[dict[str, Any]]:
    """Choose supported, mechanism-diverse pairs without fabricating examples."""
    selected: list[dict[str, Any]] = []
    selected_revisions: set[str] = set()

    # First cover distinct correction mechanisms with replicated evidence.
    for pair in pairs:
        revision = str(pair["revision_type"])
        if pair["support_count"] < 2 or revision in selected_revisions:
            continue
        selected.append(pair)
        selected_revisions.add(revision)
        if len(selected) >= max_few_shots:
            return selected

    # Then fill with strong repeated pairs, even when a revision label repeats.
    for pair in pairs:
        if pair in selected or pair["support_count"] < 2:
            continue
        selected.append(pair)
        if len(selected) >= max_few_shots:
            return selected

    # Singleton evidence is retained only as a clearly marked fallback.
    for pair in pairs:
        if pair in selected:
            continue
        selected.append(pair)
        if len(selected) >= max_few_shots:
            break
    return selected


def write_dataset_report(
    output_dir: Path,
    env: str,
    rows: list[dict[str, Any]],
    *,
    max_few_shots: int,
    max_evidence: int,
) -> dict[str, Any]:
    dataset_dir = output_dir / env
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(str(row["split"]) for row in rows)
    outcomes = Counter(str(row["outcome_status"]) for row in rows)
    question_counts = Counter(str(row["question_type"]) for row in rows)
    revision_counts = Counter(
        str(row["revision_type"])
        for row in rows
        if row.get("has_reusable_revision")
    )
    revisions = sum(bool(row.get("has_reusable_revision")) for row in rows)
    no_patch = sum(bool(row.get("no_patch")) for row in rows)
    pair_stats = pair_records(rows, max_evidence)
    shortlist_pairs = choose_shortlist(pair_stats, max_few_shots)
    shortlist = []
    for rank, pair in enumerate(shortlist_pairs, 1):
        best = pair["evidence"][0]
        shortlist.append({
            "rank": rank,
            "evidence_grade": pair["evidence_grade"],
            "support_count": pair["support_count"],
            "unstable_count": pair["unstable_count"],
            "split_support": pair["split_support"],
            "source_sample": {
                key: best.get(key)
                for key in ("split", "id", "q_i", "outcome_status")
            },
            "example": {
                key: best.get(key)
                for key in (
                    "question_type", "revision_type", "repair_signature",
                    "condition", "boundary", "patch",
                )
            },
        })

    summary = {
        "env": env,
        "n_samples": len(rows),
        "optimizer_sources": dict(
            Counter(str(row.get("optimizer_source") or "unknown") for row in rows)
        ),
        "optimizer_models": dict(Counter(str(row.get("optimizer_model") or "unknown") for row in rows)),
        "target_models": dict(Counter(str(row.get("target_model") or "unknown") for row in rows)),
        "skill_hashes": dict(Counter(str(row.get("skill_hash") or "unknown") for row in rows)),
        "repeats_requested": dict(
            Counter(str(row.get("repeats_requested") or "unknown") for row in rows)
        ),
        "split_counts": dict(split_counts),
        "outcome_counts": dict(outcomes),
        "stable_success_rate": outcomes["stable_success"] / max(len(rows), 1),
        "stable_success_rate_wilson_95": wilson(outcomes["stable_success"], len(rows)),
        "revision_rate": revisions / max(len(rows), 1),
        "revision_rate_wilson_95": wilson(revisions, len(rows)),
        "n_reusable_revisions": revisions,
        "n_no_patch": no_patch,
        "question_type_counts": dict(question_counts.most_common()),
        "revision_type_counts": dict(revision_counts.most_common()),
        "pair_stats": pair_stats,
        "shortlist_requires_semantic_label_adjudication": True,
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "few_shot_shortlist.json").write_text(
        json.dumps(shortlist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "sample_taxonomy.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    lines = [
        f"# {env}: observed taxonomy",
        "",
        f"- Samples: `{len(rows)}`",
        f"- Stable success / unstable / failure: "
        f"`{outcomes['stable_success']} / {outcomes['unstable']} / {outcomes['failure']}`",
        f"- Reusable revisions: `{revisions}`",
        f"- Analyst no-patch: `{no_patch}`",
        "",
        "## Few-shot evidence shortlist",
        "",
        "| Rank | Grade | question_type | revision_type | Support | Unstable | Splits |",
        "|---:|:---:|---|---|---:|---:|---|",
    ]
    for item in shortlist:
        example = item["example"]
        lines.append(
            f"| {item['rank']} | {item['evidence_grade']} | "
            f"`{example['question_type']}` | `{example['revision_type']}` | "
            f"{item['support_count']} | {item['unstable_count']} | "
            f"{', '.join(item['split_support'])} |"
        )
    lines.extend([
        "",
        "> Shortlist is evidence-ranked but not yet safe to paste into prompts. "
        "Semantically synonymous labels must be adjudicated first.",
        "",
    ])
    (dataset_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    files = taxonomy_files(args.input)
    rows = read_rows(files)
    expected_datasets = {
        token.strip()
        for token in args.expected_datasets.replace(",", " ").split()
        if token.strip()
    }
    manifests = validate_manifests(files, rows, expected_datasets)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_env[str(row["env"])].append(row)

    overall = {
        "source_files": [str(path) for path in files],
        "source_manifests": manifests,
        "n_samples": len(rows),
        "datasets": {},
    }
    for env in sorted(by_env):
        dataset_rows = sorted(
            by_env[env],
            key=lambda row: (str(row["split"]), str(row["id"])),
        )
        summary = write_dataset_report(
            output_dir,
            env,
            dataset_rows,
            max_few_shots=args.max_few_shots,
            max_evidence=args.max_evidence_per_pair,
        )
        overall["datasets"][env] = {
            "n_samples": summary["n_samples"],
            "n_reusable_revisions": summary["n_reusable_revisions"],
            "outcome_counts": summary["outcome_counts"],
        }
    (output_dir / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] files={len(files)} rows={len(rows)} output={output_dir}")


if __name__ == "__main__":
    main()
