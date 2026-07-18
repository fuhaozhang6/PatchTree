#!/usr/bin/env python3
"""Adjudicate observed taxonomy evidence into dataset-specific few-shots.

Two independent drafts are generated and a third call reconciles them against
the measured support table. Outputs are review artifacts; prompts are not
modified automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from skillopt.prompts import load_prompt  # noqa: E402
from skillopt.utils import extract_json  # noqa: E402


SYSTEM = """\
You are adjudicating dataset-specific few-shot examples for a reusable
PatchRecord classifier. Every revision claim comes from repeated target-model
trajectories under one fixed Skill.

Selection rules, in priority order:
1. Never invent empirical support. Every final example must cite source_pairs
   present in the evidence.
2. Merge labels only when their correction mechanism and applicability are
   semantically equivalent; similar wording alone is insufficient.
3. Prefer unstable samples (both successful and failed repeats), replicated
   support, and support spanning multiple splits.
4. Cover distinct high-frequency correction mechanisms instead of repeating
   one mechanism under several question labels.
5. Use a singleton only for a clearly distinct important mechanism, and mark
   evidence_grade C.
6. The example must be generic: remove sample entities, answers, numbers,
   file names, paths, cell addresses, object names, and dataset-row wording.
7. condition says when to apply; boundary says when not to over-apply.
8. patch must be an executable append/insert_after/replace/delete edit and must
   express one reusable correction.

Return one JSON object only:
{
  "label_merges": [
    {"from": [["question_type","revision_type"]], "to": ["question_type","revision_type"], "reason": "..."}
  ],
  "few_shots": [
    {
      "question_type": "...",
      "revision_type": "...",
      "repair_signature": "...",
      "condition": "...",
      "boundary": "...",
      "patch": {"op": "append|insert_after|replace|delete", "target": "... optional", "content": "... optional"},
      "source_pairs": [["observed_question_type","observed_revision_type"]],
      "evidence_grade": "A|B|C",
      "selection_reason": "..."
    }
  ],
  "coverage_rationale": "...",
  "rejected_pairs": [
    {"pair": ["question_type","revision_type"], "reason": "..."}
  ]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--optimizer-model", default=os.environ.get("OPTIMIZER_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--max-few-shots", type=int, default=8)
    parser.add_argument("--max-evidence-pairs", type=int, default=50)
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_few_shots < 1 or args.max_evidence_pairs < 1 or args.drafts < 1:
        parser.error("limits and draft count must be positive")
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


def compact_evidence(summary: dict[str, Any], max_evidence_pairs: int) -> dict[str, Any]:
    pairs = summary.get("pair_stats", []) or []
    catalog = [
        {
            key: pair.get(key)
            for key in (
                "question_type", "revision_type", "support_count",
                "unstable_count", "failure_count", "split_support",
                "evidence_grade",
            )
        }
        for pair in pairs
    ]
    evidence = []
    for pair in pairs[:max_evidence_pairs]:
        evidence.append({
            **{
                key: pair.get(key)
                for key in (
                    "question_type", "revision_type", "support_count",
                    "unstable_count", "split_support", "evidence_grade",
                )
            },
            "evidence": pair.get("evidence", []),
        })
    return {
        "n_samples": summary.get("n_samples"),
        "outcome_counts": summary.get("outcome_counts"),
        "all_observed_pairs": catalog,
        "detailed_top_pairs": evidence,
    }


def call_json(user: str, stage: str) -> dict[str, Any]:
    response, _ = chat_optimizer(
        system=SYSTEM,
        user=user,
        max_completion_tokens=12000,
        retries=3,
        stage=stage,
    )
    parsed = extract_json(response)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{stage}: optimizer did not return a JSON object")
    return parsed


def validate_result(
    result: dict[str, Any],
    pair_stats_by_key: dict[tuple[str, str], dict[str, Any]],
    max_few_shots: int,
) -> list[dict[str, Any]]:
    few_shots = result.get("few_shots")
    if not isinstance(few_shots, list) or not 1 <= len(few_shots) <= max_few_shots:
        raise ValueError(f"expected 1..{max_few_shots} few_shots")
    output = []
    for index, item in enumerate(few_shots):
        if not isinstance(item, dict):
            raise ValueError(f"few_shots[{index}] is not an object")
        patch = item.get("patch")
        if not isinstance(patch, dict) or patch.get("op") not in {
            "append", "insert_after", "replace", "delete",
        }:
            raise ValueError(f"few_shots[{index}] has invalid patch")
        if patch["op"] != "delete" and not str(patch.get("content") or "").strip():
            raise ValueError(f"few_shots[{index}] patch content is empty")
        if patch["op"] in {"insert_after", "replace", "delete"} and not str(
            patch.get("target") or ""
        ).strip():
            raise ValueError(f"few_shots[{index}] patch target is empty")
        raw_sources = item.get("source_pairs")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"few_shots[{index}] has no source_pairs")
        sources = []
        for source in raw_sources:
            if not isinstance(source, list) or len(source) != 2:
                raise ValueError(f"few_shots[{index}] has malformed source pair")
            pair = (str(source[0]), str(source[1]))
            if pair not in pair_stats_by_key:
                raise ValueError(f"few_shots[{index}] cites unknown pair {pair}")
            sources.append(list(pair))
        cleaned = {
            key: item.get(key)
            for key in (
                "question_type", "revision_type", "repair_signature",
                "condition", "boundary", "patch", "evidence_grade",
                "selection_reason",
            )
        }
        source_stats = [
            pair_stats_by_key[(source[0], source[1])]
            for source in sources
        ]
        support = sum(int(stats.get("support_count") or 0) for stats in source_stats)
        unstable = sum(int(stats.get("unstable_count") or 0) for stats in source_stats)
        splits = sorted({
            str(split)
            for stats in source_stats
            for split in stats.get("split_support", []) or []
        })
        if support >= 3 and (len(splits) >= 2 or unstable >= 2):
            grade = "A"
        elif support >= 2:
            grade = "B"
        else:
            grade = "C"
        cleaned["source_pairs"] = sources
        cleaned["observed_support_count"] = support
        cleaned["observed_unstable_count"] = unstable
        cleaned["observed_split_support"] = splits
        cleaned["evidence_grade"] = grade
        output.append(cleaned)
    return output


def synthesize_dataset(
    env: str,
    summary: dict[str, Any],
    *,
    optimizer_model: str,
    max_few_shots: int,
    max_evidence_pairs: int,
    drafts_count: int,
) -> dict[str, Any]:
    evidence = compact_evidence(summary, max_evidence_pairs)
    current_prompt = load_prompt("type_guided_patch_record", env)
    base_user = (
        f"Dataset: {env}\n"
        f"Maximum final few-shots: {max_few_shots}\n\n"
        f"## Current dataset prompt\n{current_prompt}\n\n"
        f"## Measured evidence\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    drafts = [
        call_json(
            base_user + f"\n\nCreate independent adjudication draft {index + 1}.",
            f"few_shot_draft_{env}_{index + 1}",
        )
        for index in range(drafts_count)
    ]
    reconcile_user = (
        f"{base_user}\n\n"
        f"## Independent drafts\n{json.dumps(drafts, ensure_ascii=False, indent=2)}\n\n"
        "Reconcile the drafts. Re-check every source_pair and prefer measured "
        "evidence over either draft. Return the final JSON schema."
    )
    final = call_json(reconcile_user, f"few_shot_reconcile_{env}")
    pair_stats_by_key = {
        (str(pair["question_type"]), str(pair["revision_type"])): pair
        for pair in summary.get("pair_stats", []) or []
    }
    final["few_shots"] = validate_result(final, pair_stats_by_key, max_few_shots)
    final["env"] = env
    final["optimizer_model"] = optimizer_model
    final["adjudication_drafts"] = drafts_count
    return final


def main() -> None:
    args = parse_args()
    merged_dir = args.merged_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    requested = {
        token.strip()
        for token in args.datasets.replace(",", " ").split()
        if token.strip()
    }
    summaries = sorted(merged_dir.glob("*/summary.json"))
    if requested:
        summaries = [path for path in summaries if path.parent.name in requested]
    if not summaries:
        raise SystemExit("no dataset summary.json files selected")
    print(
        f"[plan] datasets={' '.join(path.parent.name for path in summaries)} "
        f"drafts={args.drafts} max_few_shots={args.max_few_shots}"
    )
    if args.dry_run:
        return
    configure_optimizer(args.optimizer_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in summaries:
        env = path.parent.name
        summary = json.loads(path.read_text(encoding="utf-8"))
        result = synthesize_dataset(
            env,
            summary,
            optimizer_model=args.optimizer_model,
            max_few_shots=args.max_few_shots,
            max_evidence_pairs=args.max_evidence_pairs,
            drafts_count=args.drafts,
        )
        (output_dir / f"{env}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            f"# Proposed observed few-shots: {env}",
            "",
            "Review before replacing the dataset prompt examples.",
            "",
        ]
        for item in result["few_shots"]:
            prompt_item = {
                key: item.get(key)
                for key in (
                    "question_type", "revision_type", "repair_signature",
                    "condition", "boundary", "patch",
                )
            }
            lines.append("```json")
            lines.append(json.dumps(prompt_item, ensure_ascii=False, separators=(",", ":")))
            lines.append("```")
            lines.append("")
        (output_dir / f"{env}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] {env}: few_shots={len(result['few_shots'])}")


if __name__ == "__main__":
    main()
