#!/usr/bin/env python3
"""Mine observed question/revision types from fixed-skill repeated rollouts.

The target model runs each item multiple times under the dataset's initial
Skill. Stable-success items receive only a question_type. Failed or unstable
items are passed to the normal dataset-specific PatchRecord analyst; only an
actual reusable repair produces revision_type.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli.train import get_adapter  # noqa: E402
from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.datasets.base import BatchSpec  # noqa: E402
from skillopt.engine.trainer import (  # noqa: E402
    _can_flatten_type_guided_repeats,
    _flatten_type_guided_repeat_env,
    _split_flattened_type_guided_results,
)
from skillopt.gradient.type_guided_merge_v2 import (  # noqa: E402
    _call_patch_record_analyst,
    _group_repeated_rollouts,
    _success_rate,
)
from skillopt.model import (  # noqa: E402
    chat_optimizer,
    configure_azure_openai,
    configure_qwen_chat,
    get_token_summary,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from skillopt.prompts import load_prompt  # noqa: E402
from skillopt.utils import compute_score, extract_json, skill_hash  # noqa: E402


SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
TAXONOMY_SCHEMA_VERSION = "observed_taxonomy_v2"


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:80] or "other"


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text[:120] or "sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train val test")
    parser.add_argument("--split-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--skill")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--target-workers", type=int, default=128)
    parser.add_argument("--analyst-workers", type=int, default=32)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-source", default=os.environ.get("OPTIMIZER_SOURCE", "unknown"))
    parser.add_argument("--optimizer-model", default=os.environ.get("OPTIMIZER_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--target-model", default=os.environ.get("TARGET_MODEL", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--target-base-url", default=os.environ.get("QWEN_CHAT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--target-api-key", default=os.environ.get("QWEN_CHAT_API_KEY", "dummy"))
    parser.add_argument("--target-temperature", type=float, default=0.2)
    parser.add_argument("--target-timeout-seconds", type=float, default=300)
    parser.add_argument("--target-max-tokens", type=int, default=16384)
    parser.add_argument("--target-enable-thinking", default="false")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be >= 2 for stability-aware taxonomy")
    if args.batch_size < 1 or args.target_workers < 1 or args.analyst_workers < 1:
        parser.error("batch/worker counts must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= shard-index < shard-count")
    splits: list[str] = []
    for raw in args.splits.replace(",", " ").split():
        key = raw.strip().lower()
        if key not in SPLIT_ALIASES:
            parser.error(f"unknown split: {raw}")
        split = SPLIT_ALIASES[key]
        if split not in splits:
            splits.append(split)
    args.splits = splits
    return args


def configure_models(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    configure_azure_openai(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or cfg.get("azure_openai_endpoint") or None,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION") or cfg.get("azure_openai_api_version") or None,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY") or cfg.get("azure_openai_api_key") or None,
        auth_mode=os.environ.get("AZURE_OPENAI_AUTH_MODE") or cfg.get("azure_openai_auth_mode") or None,
        optimizer_endpoint=os.environ.get("OPTIMIZER_AZURE_OPENAI_ENDPOINT") or None,
        optimizer_api_version=os.environ.get("OPTIMIZER_AZURE_OPENAI_API_VERSION") or None,
        optimizer_api_key=os.environ.get("OPTIMIZER_AZURE_OPENAI_API_KEY") or None,
        optimizer_auth_mode=os.environ.get("OPTIMIZER_AZURE_OPENAI_AUTH_MODE") or None,
    )
    set_optimizer_backend("openai_chat")
    set_target_backend("qwen_chat")
    set_optimizer_deployment(args.optimizer_model)
    set_target_deployment(args.target_model)
    configure_qwen_chat(
        target_base_url=args.target_base_url,
        target_api_key=args.target_api_key,
        target_temperature=args.target_temperature,
        target_timeout_seconds=args.target_timeout_seconds,
        target_max_tokens=args.target_max_tokens,
        target_enable_thinking=truthy(args.target_enable_thinking),
    )
    set_reasoning_effort(args.reasoning_effort or None)


def load_flat_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    flat = flatten_config(cfg) if is_structured(cfg) else dict(cfg)
    if args.split_dir:
        flat["split_dir"] = str(Path(args.split_dir).resolve())
    if args.data_root:
        flat["data_root"] = str(Path(args.data_root).resolve())
    flat["split_mode"] = "split_dir"
    flat["seed"] = args.seed
    flat["limit"] = 0
    flat["workers"] = args.target_workers
    flat["max_api_workers"] = args.target_workers
    flat["analyst_workers"] = args.analyst_workers
    flat["optimizer_model"] = args.optimizer_model
    flat["target_model"] = args.target_model
    flat["optimizer_backend"] = "openai_chat"
    flat["target_backend"] = "qwen_chat"
    flat["target_qwen_chat_base_url"] = args.target_base_url
    flat["target_qwen_chat_api_key"] = args.target_api_key
    flat["target_qwen_chat_temperature"] = args.target_temperature
    flat["target_qwen_chat_timeout_seconds"] = args.target_timeout_seconds
    flat["target_qwen_chat_max_tokens"] = args.target_max_tokens
    flat["target_qwen_chat_enable_thinking"] = truthy(args.target_enable_thinking)
    flat["out_root"] = str(args.output_dir.resolve())
    return flat


def item_id(item: dict[str, Any], fallback: int) -> str:
    for key in ("id", "uid", "questionId", "question_id"):
        if item.get(key) not in (None, ""):
            return str(item[key])
    return f"row_{fallback:06d}"


def batch_spec_for(
    dataloader: Any,
    items: list[dict[str, Any]],
    split: str,
    seed: int,
) -> BatchSpec:
    metadata: dict[str, Any] = {}
    metadata_builder = getattr(dataloader, "_metadata_for_items", None)
    if callable(metadata_builder):
        metadata = metadata_builder(items, split, "eval")
    return BatchSpec(
        phase="eval",
        split=split,
        seed=seed,
        batch_size=len(items),
        payload=items,
        metadata=metadata,
    )


def repeated_rollout(
    *,
    adapter: Any,
    dataloader: Any,
    items: list[dict[str, Any]],
    split: str,
    seed: int,
    repeats: int,
    skill_content: str,
    chunk_dir: Path,
) -> list[dict[str, Any]]:
    batch = batch_spec_for(dataloader, items, split, seed)
    env = adapter.build_env_from_batch(batch, out_root=str(chunk_dir))
    if _can_flatten_type_guided_repeats(env):
        flat_env, id_map = _flatten_type_guided_repeat_env(env, repeats=repeats)
        rollout_dir = chunk_dir / "rollout_flattened"
        results = adapter.rollout(
            flat_env,
            skill_content,
            str(rollout_dir),
            use_eval_feedback=True,
        )
        return _split_flattened_type_guided_results(
            results,
            repeats=repeats,
            id_map=id_map,
            prediction_dir=str(rollout_dir / "predictions"),
        )

    output: list[dict[str, Any]] = []
    for repeat_id in range(repeats):
        repeat_batch = batch_spec_for(dataloader, items, split, seed)
        repeat_env = adapter.build_env_from_batch(repeat_batch, out_root=str(chunk_dir))
        rollout_dir = chunk_dir / f"rollout_repeat_{repeat_id}"
        results = adapter.rollout(
            repeat_env,
            skill_content,
            str(rollout_dir),
            use_eval_feedback=True,
        )
        output.append({
            "repeat_id": repeat_id,
            "results": results,
            "prediction_dir": str(rollout_dir / "predictions"),
        })
    return output


QUESTION_TYPE_SYSTEM = """\
You classify the intrinsic question/task structure for a skill-repair taxonomy.
This request is only for question_type. Use the dataset catalog as a preferred
vocabulary, while allowing a materially better short snake_case label. Do not
diagnose a correction and do not copy sample-specific entities, values, paths,
coordinates, or answers.
Return only:
{"question_type":"short_snake_case","confidence":"high|medium|low","reasoning":"brief"}
"""


def question_type_only(
    *,
    env_name: str,
    sample_group: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("question_type"):
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    prompt = load_prompt("type_guided_patch_record", env_name)
    match = re.search(
        r"Useful question_type labels[^:]*:\s*(.*?)"
        r"(?=\nUseful revision_type labels|\nFew-shot examples:)",
        prompt,
        flags=re.DOTALL,
    )
    catalog = match.group(1).strip() if match else "(no preferred labels supplied)"
    compact_tasks = []
    seen_tasks: set[str] = set()
    intrinsic_keys = (
        "question",
        "query",
        "task_description",
        "instruction",
        "task_type",
        "instruction_type",
        "subtype",
        "reference_text",
    )
    for rollout in sample_group.get("rollouts", []) or []:
        result = dict(rollout.get("result") or {})
        task = {
            key: result.get(key)
            for key in intrinsic_keys
            if result.get(key) not in (None, "")
        }
        fingerprint = json.dumps(task, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint not in seen_tasks:
            seen_tasks.add(fingerprint)
            compact_tasks.append(task)
    user = (
        f"## Dataset catalog\n{catalog}\n\n"
        f"## Task structure\n{json.dumps(compact_tasks, ensure_ascii=False, indent=2)}"
    )
    response, _ = chat_optimizer(
        system=QUESTION_TYPE_SYSTEM,
        user=user,
        max_completion_tokens=800,
        retries=3,
        stage="taxonomy_question_type",
    )
    parsed = extract_json(response)
    output = {
        "question_type": slug(parsed.get("question_type") if isinstance(parsed, dict) else "other"),
        "confidence": str(parsed.get("confidence") if isinstance(parsed, dict) else "low"),
        "reasoning": str(parsed.get("reasoning") if isinstance(parsed, dict) else "")[:1000],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, cache_path)
    return output


def analyze_group(
    *,
    env_name: str,
    split: str,
    sample_id: str,
    sample_group: dict[str, Any],
    skill_content: str,
    optimizer_model: str,
    cache_dir: Path,
    step_cache_dir: Path,
) -> dict[str, Any]:
    q_i = _success_rate(sample_group)
    outcome = "stable_success" if q_i >= 1.0 else "failure" if q_i <= 0 else "unstable"
    record: dict[str, Any] | None = None
    report: dict[str, Any] = {}
    no_patch = False
    if q_i < 1.0:
        record, report = _call_patch_record_analyst(
            skill_content=skill_content,
            sample_group=sample_group,
            q_i=q_i,
            status=outcome,
            optimizer_model=optimizer_model,
            cache_dir=str(cache_dir / "patch_records"),
            step_cache_dir=str(step_cache_dir / "patch_records"),
            env_name=env_name,
        )
        no_patch = bool(record and record.get("no_patch"))

    question = question_type_only(
        env_name=env_name,
        sample_group=sample_group,
        cache_path=(
            cache_dir
            / f"question_types_{TAXONOMY_SCHEMA_VERSION}"
            / f"{safe_name(split)}_{safe_name(sample_id)}.json"
        ),
    )
    question_type = question["question_type"]
    question_confidence = question["confidence"]
    question_reasoning = question["reasoning"]
    analyst_question_type = (
        slug(record.get("question_type"))
        if record and not no_patch
        else None
    )

    hard_values = []
    for rollout in sample_group.get("rollouts", []) or []:
        try:
            hard_values.append(float((rollout.get("result") or {}).get("hard", 0) or 0))
        except (TypeError, ValueError):
            hard_values.append(0.0)
    row = {
        "env": env_name,
        "split": split,
        "id": sample_id,
        "n_rollouts": len(hard_values),
        "hard_values": hard_values,
        "q_i": q_i,
        "outcome_status": outcome,
        "question_type": question_type,
        "question_type_confidence": question_confidence,
        "question_type_reasoning": question_reasoning,
        "analyst_question_type": analyst_question_type,
        "question_type_agrees_with_analyst": (
            question_type == analyst_question_type
            if analyst_question_type is not None
            else None
        ),
        "has_reusable_revision": bool(record and not no_patch),
        "no_patch": no_patch,
        "analyst_status": report.get("status", "not_needed"),
        "analyst_error": report.get("error", ""),
    }
    if record and not no_patch:
        row.update({
            "revision_type": slug(record.get("revision_type")),
            "repair_signature": str(record.get("repair_signature") or ""),
            "condition": str(record.get("condition") or ""),
            "boundary": str(record.get("boundary") or ""),
            "patch": record.get("patch") or {},
            "revision_basis": "observed_repeated_target_trajectory",
        })
    else:
        row["revision_type"] = None
        row["revision_basis"] = "none"
    return row


def write_summary(output_dir: Path, rows: list[dict[str, Any]], cfg: dict[str, Any], args: argparse.Namespace) -> None:
    pair_counts = Counter(
        (str(row["question_type"]), str(row["revision_type"]))
        for row in rows if row.get("has_reusable_revision")
    )
    question_counts = Counter(str(row["question_type"]) for row in rows)
    outcome_counts = Counter(str(row["outcome_status"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    revision_counts = Counter(
        str(row["revision_type"]) for row in rows if row.get("has_reusable_revision")
    )

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("has_reusable_revision"):
            candidates[(str(row["question_type"]), str(row["revision_type"]))].append(row)
    few_shot_candidates = []
    for pair, pair_rows in sorted(candidates.items(), key=lambda item: (-len(item[1]), item[0])):
        pair_rows.sort(
            key=lambda row: (
                0 if row["outcome_status"] == "unstable" else 1,
                abs(float(row["q_i"]) - 0.5),
                -len(str(row.get("condition") or "")),
                str(row["split"]),
                str(row["id"]),
            )
        )
        few_shot_candidates.append({
            "question_type": pair[0],
            "revision_type": pair[1],
            "support_count": len(pair_rows),
            "unstable_count": sum(row["outcome_status"] == "unstable" for row in pair_rows),
            "failure_count": sum(row["outcome_status"] == "failure" for row in pair_rows),
            "candidates": [
                {
                    key: row.get(key)
                    for key in (
                        "split", "id", "q_i", "outcome_status", "repair_signature",
                        "condition", "boundary", "patch",
                    )
                }
                for row in pair_rows[:3]
            ],
        })

    summary = {
        "taxonomy_schema_version": TAXONOMY_SCHEMA_VERSION,
        "env": cfg.get("env"),
        "optimizer_source": args.optimizer_source,
        "skill": str(cfg.get("skill_init")),
        "skill_hash": skill_hash(
            Path(cfg["skill_init"]).read_text(encoding="utf-8")
        ),
        "repeats": args.repeats,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "n_samples": len(rows),
        "expected_samples": int(getattr(args, "expected_samples", len(rows))),
        "complete": len(rows) == int(getattr(args, "expected_samples", len(rows))),
        "split_counts": dict(split_counts),
        "outcome_counts": dict(outcome_counts),
        "n_reusable_revisions": sum(row.get("has_reusable_revision", False) for row in rows),
        "n_no_patch": sum(row.get("no_patch", False) for row in rows),
        "question_type_counts": dict(question_counts),
        "revision_type_counts": dict(revision_counts),
        "pair_counts": [
            {"question_type": key[0], "revision_type": key[1], "count": count}
            for key, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "token_usage": get_token_summary(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "few_shot_candidates.json").write_text(
        json.dumps(few_shot_candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# Observed taxonomy: {cfg.get('env')}",
        "",
        f"- Samples: `{len(rows)}`",
        f"- Stable success: `{outcome_counts['stable_success']}`",
        f"- Unstable: `{outcome_counts['unstable']}`",
        f"- Failure: `{outcome_counts['failure']}`",
        f"- Reusable revisions: `{summary['n_reusable_revisions']}`",
        f"- No-patch failures/unstable: `{summary['n_no_patch']}`",
        "",
        "| Question type | Revision type | Count |",
        "|---|---|---:|",
    ]
    for key, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key[0]}` | `{key[1]}` | {count} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_flat_config(args)
    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    dataloader = adapter.get_dataloader()
    if dataloader is None or not hasattr(dataloader, "get_split_items"):
        raise SystemExit(f"{cfg.get('env')} does not expose split items")

    skill_path = Path(args.skill or cfg["skill_init"])
    if not skill_path.is_absolute():
        skill_path = PROJECT_ROOT / skill_path
    skill_path = skill_path.resolve()
    skill_content = skill_path.read_text(encoding="utf-8")
    cfg["skill_init"] = str(skill_path)
    env_name = str(cfg.get("env") or "")

    split_items: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        items = list(dataloader.get_split_items(split))
        items = items[args.shard_index::args.shard_count]
        if args.limit_per_split > 0:
            items = items[:args.limit_per_split]
        split_items[split] = items
    args.expected_samples = sum(len(items) for items in split_items.values())

    print("============================================================")
    print("  Observed target-trajectory taxonomy audit")
    print("============================================================")
    print(f"  env:           {env_name}")
    print(f"  splits:        {' '.join(args.splits)}")
    print(f"  repeats:       {args.repeats}")
    print(f"  shard:         {args.shard_index}/{args.shard_count}")
    print(f"  target:        {args.target_model} @ {args.target_base_url}")
    print(f"  optimizer:     {args.optimizer_model}")
    print(f"  batch/workers: {args.batch_size}/{args.target_workers}")
    print(f"  analyst:       {args.analyst_workers}")
    print(f"  skill:         {skill_path}")
    print(f"  output:        {args.output_dir}")
    for split, items in split_items.items():
        print(f"    {split}: {len(items)}")
    print("============================================================")
    if args.dry_run:
        return
    configure_models(args, cfg)

    all_rows: list[dict[str, Any]] = []
    for split, items in split_items.items():
        for start in range(0, len(items), args.batch_size):
            chunk_items = items[start:start + args.batch_size]
            chunk_index = start // args.batch_size
            chunk_dir = args.output_dir / "chunks" / split / f"chunk_{chunk_index:05d}"
            taxonomy_path = chunk_dir / "sample_taxonomy.jsonl"
            if taxonomy_path.is_file():
                cached_rows = [
                    json.loads(line)
                    for line in taxonomy_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if (
                    len(cached_rows) == len(chunk_items)
                    and all(
                        row.get("taxonomy_schema_version") == TAXONOMY_SCHEMA_VERSION
                        for row in cached_rows
                    )
                ):
                    all_rows.extend(cached_rows)
                    print(f"[resume] {split} chunk={chunk_index} rows={len(cached_rows)}")
                    continue

            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_seed = args.seed + args.shard_index * 10_000_019 + chunk_index
            print(
                f"[rollout] split={split} chunk={chunk_index} "
                f"items={len(chunk_items)} repeats={args.repeats}"
            )
            repeated = repeated_rollout(
                adapter=adapter,
                dataloader=dataloader,
                items=chunk_items,
                split=split,
                seed=chunk_seed,
                repeats=args.repeats,
                skill_content=skill_content,
                chunk_dir=chunk_dir,
            )
            for repeat in repeated:
                hard, soft = compute_score(repeat.get("results", []) or [])
                print(
                    f"  repeat={int(repeat.get('repeat_id', 0)) + 1} "
                    f"hard={hard:.4f} soft={soft:.4f}"
                )
            groups = _group_repeated_rollouts(repeated, include_trajectories=True)
            if len(groups) != len(chunk_items):
                raise RuntimeError(
                    "Incomplete rollout coverage: "
                    f"split={split} chunk={chunk_index} "
                    f"expected={len(chunk_items)} observed={len(groups)}. "
                    "The audit stops instead of silently biasing the taxonomy."
                )
            cache_dir = args.output_dir / "taxonomy_cache"
            step_cache_dir = chunk_dir / "analyst_cache"
            rows: list[dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.analyst_workers
            ) as executor:
                futures = {
                    executor.submit(
                        analyze_group,
                        env_name=env_name,
                        split=split,
                        sample_id=sample_id,
                        sample_group=group,
                        skill_content=skill_content,
                        optimizer_model=args.optimizer_model,
                        cache_dir=cache_dir,
                        step_cache_dir=step_cache_dir,
                    ): sample_id
                    for sample_id, group in groups.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    row["optimizer_model"] = args.optimizer_model
                    row["optimizer_source"] = args.optimizer_source
                    row["target_model"] = args.target_model
                    row["skill_hash"] = skill_hash(skill_content)
                    row["repeats_requested"] = args.repeats
                    row["taxonomy_schema_version"] = TAXONOMY_SCHEMA_VERSION
                    rows.append(row)
            rows.sort(key=lambda row: str(row["id"]))
            taxonomy_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            all_rows.extend(rows)
            outcomes = Counter(row["outcome_status"] for row in rows)
            revisions = sum(row["has_reusable_revision"] for row in rows)
            print(
                f"[taxonomy] split={split} chunk={chunk_index} "
                f"stable={outcomes['stable_success']} unstable={outcomes['unstable']} "
                f"failure={outcomes['failure']} revisions={revisions}"
            )

    all_rows.sort(key=lambda row: (str(row["split"]), str(row["id"])))
    (args.output_dir / "sample_taxonomy.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    write_summary(args.output_dir, all_rows, cfg, args)
    print(f"[done] rows={len(all_rows)} summary={args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
