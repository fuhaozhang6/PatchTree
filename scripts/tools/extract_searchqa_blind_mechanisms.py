#!/usr/bin/env python3
"""Extract type-free mechanism cards from saved SearchQA repeated trajectories."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.gradient.type_guided_merge_v2 import fmt_trajectory  # noqa: E402
from skillopt.model import (  # noqa: E402
    chat_optimizer,
    configure_azure_openai,
    set_optimizer_backend,
    set_optimizer_deployment,
)
from skillopt.utils import extract_json  # noqa: E402
from scripts.tools.searchqa_blind_common import (  # noqa: E402
    SCHEMA_VERSION,
    infrastructure_failure,
    outcome_from_rollouts,
    read_jsonl,
    sample_key,
    split_prediction_id,
    stable_hash,
    write_json,
    write_jsonl,
)

BLIND_SYSTEM = """\
Perform open coding of repeated model trajectories. Infer at most one reusable
repair mechanism from the evidence itself.

You are intentionally given no question-type catalog, revision-type catalog,
or taxonomy examples. Do not invent or return a category/type name. Describe
observable behavior, the missing operation, applicability, boundary, and one
generic repair. Never copy sample entities, answers, dates, quotations, or
row-specific facts into reusable fields.

For mixed successful/failed attempts, use successful attempts as the
within-question contrast. For all-failed attempts, be conservative. Return
reusable=false when evidence is insufficient, the issue is infrastructure,
or the current Skill already contains the complete rule.

Return one JSON object only:
{
  "reusable": true|false,
  "reasoning": "brief evidence assessment",
  "observed_failure": "generic observable failed behavior",
  "successful_contrast": "generic difference from successful attempts; empty if none",
  "missing_operation": "one missing or misapplied operation",
  "repair_signature": "generic operational mechanism, not a type label",
  "mechanism_keywords": ["2-6 generic action phrases"],
  "condition": "when the repair applies",
  "boundary": "when it must not be applied",
  "candidate_patch": {"op":"append","content":"one generic executable Skill rule"},
  "confidence": "high|medium|low"
}
"""

RESULT_KEYS = (
    "id", "hard", "soft", "predicted_answer", "response", "fail_reason",
    "agent_ok", "evaluator_feedback", "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skill",
        type=Path,
        default=PROJECT_ROOT / "skillopt/envs/searchqa/skills/initial.md",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data/searchqa_split",
        help="Restore question/context from data without running the target model.",
    )
    parser.add_argument("--splits", default="train val test")
    parser.add_argument("--statuses", default="unstable failure")
    parser.add_argument(
        "--optimizer-model",
        default=os.environ.get("OPTIMIZER_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument("--analyst-workers", type=int, default=64)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-trajectory-chars", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.analyst_workers < 1:
        parser.error("--analyst-workers must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= shard-index < shard-count")
    args.splits = set(args.splits.replace(",", " ").split())
    args.statuses = set(args.statuses.replace(",", " ").split())
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


def observed_dataset_dirs(inputs: list[Path]) -> list[Path]:
    directories: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if (path / "sample_taxonomy.jsonl").is_file() and path.name.startswith("searchqa"):
            directories.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in path.rglob("sample_taxonomy.jsonl"):
            parent = candidate.parent
            if parent.name.startswith("searchqa") and (parent / "chunks").is_dir():
                directories.add(parent)
    if not directories:
        raise FileNotFoundError("no SearchQA observed-taxonomy directories found")
    return sorted(directories)


def load_conversation(path: Path, max_chars: int) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return fmt_trajectory(value)[:max_chars] if isinstance(value, list) else ""


def evidence_from_chunk(
    chunk_dir: Path,
    taxonomy_rows: list[dict[str, Any]],
    max_trajectory_chars: int,
    tasks_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results_path = chunk_dir / "rollout_flattened" / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"saved trajectories missing: {results_path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in read_jsonl(results_path):
        original_id, repeat_id = split_prediction_id(str(result.get("id") or ""))
        prediction_id = str(result.get("id") or "")
        compact = {
            key: result.get(key)
            for key in RESULT_KEYS
            if result.get(key) not in (None, "", [])
        }
        grouped.setdefault(original_id, []).append({
            "repeat_id": repeat_id,
            "result": compact,
            "trajectory": load_conversation(
                chunk_dir / "rollout_flattened" / "predictions"
                / prediction_id / "conversation.json",
                max_trajectory_chars,
            ),
            "infra_failure": infrastructure_failure(compact),
        })
    output = []
    for taxonomy in taxonomy_rows:
        split = str(taxonomy.get("split") or chunk_dir.parent.name)
        sample_id = str(taxonomy.get("id") or "")
        rollouts = sorted(grouped.get(sample_id, []), key=lambda row: int(row["repeat_id"]))
        valid = [row for row in rollouts if not row["infra_failure"]]
        status, q_i = outcome_from_rollouts(valid)
        output.append({
            "split": split,
            "id": sample_id,
            "sample_key": sample_key(split, sample_id),
            "outcome_status": status,
            "q_i": q_i,
            "n_rollouts": len(rollouts),
            "n_valid_rollouts": len(valid),
            "infra_failures": [row["infra_failure"] for row in rollouts if row["infra_failure"]],
            "task_evidence": tasks_by_key.get(sample_key(split, sample_id), {}),
            "rollouts": valid,
            # Stored for post-hoc comparison only; never included in the LLM user message.
            "seeded_taxonomy": {
                "question_type": taxonomy.get("question_type"),
                "revision_type": taxonomy.get("revision_type"),
            },
        })
    return output


def collect_evidence(
    dataset_dirs: list[Path],
    splits: set[str],
    statuses: set[str],
    max_trajectory_chars: int,
    tasks_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_dir in dataset_dirs:
        pattern = dataset_dir / "chunks"
        for taxonomy_path in sorted(pattern.glob("*/*/sample_taxonomy.jsonl")):
            for row in evidence_from_chunk(
                taxonomy_path.parent,
                read_jsonl(taxonomy_path),
                max_trajectory_chars,
                tasks_by_key,
            ):
                if row["split"] not in splits or row["outcome_status"] not in statuses:
                    continue
                if row["sample_key"] in seen:
                    raise ValueError(f"duplicate sample: {row['sample_key']}")
                seen.add(row["sample_key"])
                rows.append(row)
    return sorted(rows, key=lambda row: row["sample_key"])


def load_task_evidence(split_dir: Path, splits: set[str]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for split in sorted(splits):
        path = split_dir.expanduser().resolve() / split / "items.json"
        if not path.is_file():
            raise FileNotFoundError(f"SearchQA split data missing: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"expected a JSON list: {path}")
        for item in values:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            key = sample_key(split, str(item["id"]))
            answers = item.get("answers")
            if isinstance(answers, str):
                try:
                    parsed_answers = ast.literal_eval(answers)
                    if isinstance(parsed_answers, list):
                        answers = parsed_answers
                except (SyntaxError, ValueError):
                    pass
            tasks[key] = {
                "question": item.get("question"),
                "context": str(item.get("context") or "")[:6500],
                "gold_answers": answers,
            }
    return tasks


def validate_card(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("reusable") is not True:
        return {"reusable": False, "reasoning": str(value.get("reasoning") or "")}
    required = ("observed_failure", "missing_operation", "repair_signature", "condition", "boundary")
    if any(not str(value.get(key) or "").strip() for key in required):
        return None
    patch = value.get("candidate_patch")
    if not isinstance(patch, dict) or patch.get("op") != "append":
        return None
    if not str(patch.get("content") or "").strip():
        return None
    keywords = value.get("mechanism_keywords") or []
    return {
        "reusable": True,
        "reasoning": str(value.get("reasoning") or "")[:2000],
        "observed_failure": str(value.get("observed_failure") or "")[:2000],
        "successful_contrast": str(value.get("successful_contrast") or "")[:2000],
        "missing_operation": str(value.get("missing_operation") or "")[:2000],
        "repair_signature": str(value.get("repair_signature") or "")[:2000],
        "mechanism_keywords": (
            [str(item)[:200] for item in keywords[:8]] if isinstance(keywords, list) else []
        ),
        "condition": str(value.get("condition") or "")[:2000],
        "boundary": str(value.get("boundary") or "")[:2000],
        "candidate_patch": {"op": "append", "content": str(patch["content"])[:4000]},
        "confidence": str(value.get("confidence") or "low").lower(),
    }


def extract_one(
    evidence: dict[str, Any],
    skill_content: str,
    optimizer_model: str,
    cache_dir: Path,
) -> dict[str, Any]:
    metadata_fields = (
        "split", "id", "sample_key", "outcome_status", "q_i", "n_rollouts",
        "n_valid_rollouts", "infra_failures", "task_evidence",
    )
    metadata = {field: evidence[field] for field in metadata_fields}
    cache_path = cache_dir / (
        stable_hash({
            "schema": SCHEMA_VERSION,
            "system_prompt": stable_hash(BLIND_SYSTEM, 32),
            "key": evidence["sample_key"],
            "skill": stable_hash(skill_content),
            "task": evidence["task_evidence"],
            "rollouts": evidence["rollouts"],
            "model": optimizer_model,
        }) + ".json"
    )
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("schema_version") == SCHEMA_VERSION:
            return cached
    if evidence["n_valid_rollouts"] < 2:
        output = {
            **metadata,
            "reusable": False,
            "status": "insufficient_valid_rollouts",
            "reasoning": "Fewer than two valid attempts.",
            "schema_version": SCHEMA_VERSION,
        }
        write_json(cache_path, output)
        return output
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Empirical outcome\nsample_key: {evidence['sample_key']}\n"
        f"valid_attempts: {evidence['n_valid_rollouts']}\n"
        f"success_rate: {evidence['q_i']:.4f}\n"
        f"status: {evidence['outcome_status']}\n\n"
        f"## Original task evidence\n"
        f"{json.dumps(evidence['task_evidence'], ensure_ascii=False, indent=2)}\n\n"
        f"## Repeated trajectory evidence\n"
        f"{json.dumps(evidence['rollouts'], ensure_ascii=False, indent=2)}"
    )
    parsed = None
    error = ""
    try:
        response, _ = chat_optimizer(
            system=BLIND_SYSTEM,
            user=user,
            max_completion_tokens=2500,
            retries=3,
            stage="searchqa_blind_mechanism",
        )
        parsed = extract_json(response)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    card = validate_card(parsed)
    if card is None:
        status = "invalid"
        card = {"reusable": False, "reasoning": error or "Invalid analyst JSON."}
    else:
        status = "ok" if card["reusable"] else "no_reusable_mechanism"
    output = {
        **metadata,
        **card,
        "status": status,
        "analyst_error": error,
        "optimizer_model": optimizer_model,
        "schema_version": SCHEMA_VERSION,
    }
    write_json(cache_path, output)
    return output


def main() -> None:
    args = parse_args()
    dataset_dirs = observed_dataset_dirs(args.input)
    tasks_by_key = load_task_evidence(args.split_dir, args.splits)
    evidence = collect_evidence(
        dataset_dirs,
        args.splits,
        args.statuses,
        args.max_trajectory_chars,
        tasks_by_key,
    )[args.shard_index::args.shard_count]
    if args.limit > 0:
        evidence = evidence[:args.limit]
    outcomes: dict[str, int] = {}
    for row in evidence:
        outcomes[row["outcome_status"]] = outcomes.get(row["outcome_status"], 0) + 1
    context_coverage = sum(bool(row.get("task_evidence", {}).get("context")) for row in evidence)
    conversation_coverage = sum(
        any(bool(rollout.get("trajectory")) for rollout in row.get("rollouts", []))
        for row in evidence
    )
    complete_attempts = sum(row.get("n_valid_rollouts") == 3 for row in evidence)
    print("============================================================")
    print("  SearchQA blind mechanism extraction (saved trajectories)")
    print("============================================================")
    print(f"  datasets:   {' '.join(str(path) for path in dataset_dirs)}")
    print(f"  samples:    {len(evidence)} {outcomes}")
    print(f"  3 valid attempts: {complete_attempts}/{len(evidence)}")
    print(f"  context restored: {context_coverage}/{len(evidence)}")
    print(f"  nonempty conversation files: {conversation_coverage}/{len(evidence)}")
    print(f"  shard:      {args.shard_index}/{args.shard_count}")
    print(f"  workers:    {args.analyst_workers}")
    print(f"  output:     {args.output_dir}")
    print("  target Qwen calls: 0")
    print("  old labels passed to analyst: false")
    print("============================================================")
    if args.dry_run:
        return
    if not evidence:
        raise SystemExit("no eligible saved trajectory evidence")
    skill_content = args.skill.expanduser().resolve().read_text(encoding="utf-8")
    configure_optimizer(args.optimizer_model)
    output_dir = args.output_dir.expanduser().resolve()
    cards: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.analyst_workers) as executor:
        futures = [
            executor.submit(
                extract_one, row, skill_content, args.optimizer_model, output_dir / "cache"
            )
            for row in evidence
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            cards.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"[cards] {index}/{len(futures)}")
    cards.sort(key=lambda row: row["sample_key"])
    usable = [row for row in cards if row.get("reusable") and row.get("status") == "ok"]
    write_jsonl(output_dir / "mechanism_cards.jsonl", cards)
    write_jsonl(output_dir / "usable_mechanism_cards.jsonl", usable)
    # Physically separate old labels from blind cards. Only the final reporting
    # stage may read this file after the new taxonomy has been frozen.
    write_jsonl(output_dir / "posthoc_seeded_labels.jsonl", [
        {
            "sample_key": row["sample_key"],
            "seeded_question_type": row["seeded_taxonomy"].get("question_type"),
            "seeded_revision_type": row["seeded_taxonomy"].get("revision_type"),
        }
        for row in evidence
    ])
    write_json(output_dir / "summary.json", {
        "schema_version": SCHEMA_VERSION,
        "n_evidence": len(evidence),
        "n_usable": len(usable),
        "status_counts": {
            status: sum(row.get("status") == status for row in cards)
            for status in sorted({str(row.get("status")) for row in cards})
        },
        "outcome_counts": outcomes,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "optimizer_model": args.optimizer_model,
        "target_qwen_calls": 0,
        "old_labels_passed_to_analyst": False,
    })
    print(f"[done] usable={len(usable)}/{len(cards)}")


if __name__ == "__main__":
    main()
