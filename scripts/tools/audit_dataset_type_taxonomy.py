#!/usr/bin/env python3
"""Audit question/revision type priors across every dataset split.

This is not a training entry point. It sends compact static task descriptions
to an OpenAI-compatible classifier, records one taxonomy prior per item, and
writes resumable JSONL plus aggregate CSV/JSON/Markdown reports.

Because no target-model trajectory is produced, ``revision_type`` means the
most plausible reusable correction category for this task structure. It is not
an observed diagnosis of an actual model failure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ALIASES = {
    "searchqa": "searchqa",
    "search": "searchqa",
    "spreadsheet": "spreadsheetbench",
    "spreadsheetbench": "spreadsheetbench",
    "officeqa": "officeqa",
    "office": "officeqa",
    "docvqa": "docvqa",
    "doc": "docvqa",
    "livemath": "livemath",
    "livemathematicianbench": "livemath",
    "alfworld": "alfworld",
}

DEFAULT_DATASET_DIRS = {
    "searchqa": "data/searchqa_split",
    "spreadsheetbench": "data/spreadsheetbench_split",
    "officeqa": "data/officeqa_split",
    "docvqa": "data/docvqa/splits",
    "livemath": "data/livemathematicianbench_split",
    "alfworld": "data/alfworld_path_split",
}

PROMPT_ENV_NAMES = {
    "searchqa": "searchqa",
    "spreadsheetbench": "spreadsheetbench",
    "officeqa": "officeqa",
    "docvqa": "docvqa",
    "livemath": "livemathematicianbench",
    "alfworld": "alfworld",
}

SYSTEM_PROMPT = """\
You audit task taxonomies for a reusable skill-repair system.

For one raw dataset item, infer:
1. question_type: the intrinsic task/reasoning structure.
2. revision_type: the single most plausible reusable skill-correction category
   that would be needed if a capable solver failed on this kind of item.
3. repair_signature: a concrete generic correction mechanism.

Important:
- There is no observed model trajectory. revision_type is therefore a task prior,
  not a claim about an actual failure.
- Use the dataset catalog as a consistency prior, not a closed vocabulary. Create
  a new short snake_case label when the catalog misses a materially different type.
- Do not copy sample-specific entities, answers, numbers, file names, paths, cell
  addresses, or object names into generic fields.
- generic_failure_scenario and generic_repair_rule must be suitable as sanitized
  few-shot material.
- Return only one JSON object, without Markdown.

Required schema:
{
  "question_type": "short_snake_case",
  "revision_type": "short_snake_case",
  "repair_signature": "3-8 generic words",
  "generic_failure_scenario": "generic failure pattern without sample facts",
  "generic_repair_rule": "one operational generic correction rule",
  "confidence": "high|medium|low",
  "reasoning": "brief classification rationale"
}
"""


def parse_names(raw: str, aliases: dict[str, str]) -> list[str]:
    names: list[str] = []
    for token in raw.replace(",", " ").split():
        key = token.strip().lower()
        if key not in aliases:
            raise ValueError(f"unknown dataset: {token}")
        name = aliases[key]
        if name not in names:
            names.append(name)
    return names


def parse_splits(raw: str) -> list[str]:
    splits: list[str] = []
    aliases = {"valid": "val", "validation": "val", "valid_seen": "val"}
    for token in raw.replace(",", " ").split():
        split = aliases.get(token.strip().lower(), token.strip().lower())
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {token}")
        if split not in splits:
            splits.append(split)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="searchqa spreadsheetbench officeqa docvqa livemath alfworld",
    )
    parser.add_argument("--splits", default="train val test")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a dataset split directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "TAXONOMY_BASE_URL",
            os.environ.get(
                "OPTIMIZER_AZURE_OPENAI_ENDPOINT",
                os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            ),
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TAXONOMY_MODEL",
            os.environ.get("OPTIMIZER_MODEL", "deepseek-v4-pro"),
        ),
    )
    parser.add_argument("--api-key-env", default="TAXONOMY_API_KEY")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-item-chars", type=int, default=7000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.datasets = parse_names(args.datasets, DATASET_ALIASES)
        args.splits = parse_splits(args.splits)
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.limit_per_split < 0:
        parser.error("--limit-per-split must be >= 0")
    if not args.dry_run and not args.base_url:
        parser.error("--base-url or TAXONOMY_BASE_URL is required")
    return args


def parse_dataset_dirs(values: list[str], project_root: Path) -> dict[str, Path]:
    resolved = {
        name: (project_root / relative).resolve()
        for name, relative in DEFAULT_DATASET_DIRS.items()
    }
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --dataset-dir {value!r}; expected NAME=PATH")
        raw_name, raw_path = value.split("=", 1)
        key = raw_name.strip().lower()
        if key not in DATASET_ALIASES:
            raise ValueError(f"unknown dataset in --dataset-dir: {raw_name}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved[DATASET_ALIASES[key]] = path.resolve()
    return resolved


def load_items(split_dir: Path, split: str) -> list[dict[str, Any]]:
    directory = split_dir / split
    if not directory.is_dir():
        raise FileNotFoundError(f"split directory not found: {directory}")
    candidates = [
        directory / "items.json",
        directory / "items.jsonl",
        directory / "items.csv",
        *sorted(directory.glob("*.json")),
        *sorted(directory.glob("*.jsonl")),
        *sorted(directory.glob("*.csv")),
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no JSON/JSONL/CSV split file found in {directory}")
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        rows = value.get("data") or value.get("items") or list(value.values())
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    raise ValueError(f"unsupported split structure in {path}")


def compact(value: Any, limit: int) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()[:limit]
    if isinstance(value, list):
        item_limit = max(80, limit // max(len(value), 1))
        return [compact(item, item_limit) for item in value[:8]]
    if isinstance(value, dict):
        item_limit = max(80, limit // max(len(value), 1))
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:12]:
            cleaned = compact(item, item_limit)
            if cleaned is not None:
                output[str(key)] = cleaned
        return output
    return value


def load_alfworld_task(item: dict[str, Any], project_root: Path) -> dict[str, Any]:
    gamefile = str(item.get("gamefile") or "").strip()
    if not gamefile:
        return {}
    path = Path(gamefile)
    if not path.is_absolute():
        candidates = [project_root / path, project_root / "data/alfworld" / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    try:
        with (path.parent / "traj_data.json").open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    annotations = ((data.get("turk_annotations") or {}).get("anns") or [])
    descriptions = [
        str(row.get("task_desc") or "").strip()
        for row in annotations[:3]
        if str(row.get("task_desc") or "").strip()
    ]
    return {
        "task_type": data.get("task_type"),
        "task_descriptions": descriptions,
        "pddl_params": data.get("pddl_params"),
    }


def item_card(
    dataset: str,
    item: dict[str, Any],
    project_root: Path,
    max_chars: int,
) -> dict[str, Any]:
    field_map = {
        "searchqa": ("question", "task_type", "answers", "context"),
        "officeqa": (
            "question", "category", "task_type", "answers", "ground_truth",
            "source_docs",
        ),
        "livemath": (
            "question", "theorem_type", "choices", "correct_choice", "theorem",
        ),
        "spreadsheetbench": (
            "instruction", "instruction_type", "answer_position",
        ),
        "docvqa": (
            "question", "topic", "category", "answers", "answer", "ground_truth",
        ),
        "alfworld": ("task_type",),
    }
    card: dict[str, Any] = {"dataset": dataset}
    per_field = max(300, max_chars // max(len(field_map[dataset]), 1))
    for field in field_map[dataset]:
        value = compact(item.get(field), per_field)
        if value is not None:
            card[field] = value
    if dataset == "alfworld":
        for key, value in load_alfworld_task(item, project_root).items():
            cleaned = compact(value, per_field)
            if cleaned is not None:
                card[key] = cleaned
    raw = json.dumps(card, ensure_ascii=False)
    return card if len(raw) <= max_chars else {"dataset": dataset, "task": raw[:max_chars]}


def item_id(item: dict[str, Any], index: int) -> str:
    for key in ("id", "uid", "questionId", "question_id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{index:06d}"


def load_catalog(project_root: Path, dataset: str) -> str:
    env_name = PROMPT_ENV_NAMES[dataset]
    path = project_root / "skillopt/envs" / env_name / "prompts/type_guided_patch_record.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text.split("Few-shot examples:", 1)[0][-5000:]


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:80] or "other"


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start:index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def validate_result(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    repair_signature = str(value.get("repair_signature") or "").strip()
    scenario = str(value.get("generic_failure_scenario") or "").strip()
    rule = str(value.get("generic_repair_rule") or "").strip()
    if not repair_signature or not scenario or not rule:
        return None
    confidence = str(value.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "question_type": slug(value.get("question_type")),
        "revision_type": slug(value.get("revision_type")),
        "repair_signature": repair_signature[:240],
        "generic_failure_scenario": scenario[:1200],
        "generic_repair_rule": rule[:1200],
        "confidence": confidence,
        "reasoning": str(value.get("reasoning") or "").strip()[:1200],
        "revision_basis": "task_structure_prior_without_observed_failure",
    }


class TaxonomyClient:
    def __init__(self, args: argparse.Namespace, api_key: str) -> None:
        self.args = args
        self.api_key = api_key
        self.endpoint = args.base_url.rstrip("/") + "/chat/completions"
        self._usage_lock = threading.Lock()
        self.usage = Counter()

    def classify(
        self,
        *,
        dataset: str,
        split: str,
        record_id: str,
        card: dict[str, Any],
        catalog: str,
    ) -> dict[str, Any]:
        user = (
            f"## Dataset\n{dataset}\n\n"
            f"## Dataset taxonomy catalog\n{catalog or '(no predefined catalog)'}\n\n"
            "## Raw task item\n"
            f"{json.dumps(card, ensure_ascii=False, indent=2)}"
        )
        payload = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.args.max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        last_error = ""
        for attempt in range(1, self.args.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "PatchTree-Taxonomy-Audit",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.args.timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"].get("content") or ""
                result = validate_result(extract_json_object(content))
                if result is None:
                    raise ValueError(f"invalid taxonomy JSON: {content[:500]}")
                usage = body.get("usage") or {}
                with self._usage_lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                    self.usage["completion_tokens"] += int(
                        usage.get("completion_tokens", 0) or 0
                    )
                    self.usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                return {"dataset": dataset, "split": split, "id": record_id, **result}
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")[:800]
                    except Exception:  # noqa: BLE001
                        detail = ""
                    last_error = f"HTTP {exc.code}: {detail}"
                else:
                    last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.args.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return {"dataset": dataset, "split": split, "id": record_id, "error": last_error}


def read_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if not path.is_file():
        return rows, completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("error"):
                continue
            key = (str(row.get("dataset")), str(row.get("split")), str(row.get("id")))
            rows.append(row)
            completed.add(key)
    return rows, completed


def write_reports(output_dir: Path, rows: list[dict[str, Any]], usage: Counter) -> None:
    valid = [row for row in rows if not row.get("error")]
    pair_counts = Counter(
        (
            str(row["dataset"]),
            str(row["split"]),
            str(row["question_type"]),
            str(row["revision_type"]),
        )
        for row in valid
    )
    question_counts = Counter(
        (str(row["dataset"]), str(row["split"]), str(row["question_type"]))
        for row in valid
    )
    revision_counts = Counter(
        (str(row["dataset"]), str(row["split"]), str(row["revision_type"]))
        for row in valid
    )

    def write_csv(path: Path, header: list[str], data: list[tuple[Any, ...]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(data)

    write_csv(
        output_dir / "pair_counts.csv",
        ["dataset", "split", "question_type", "revision_type", "count"],
        [(*key, count) for key, count in sorted(pair_counts.items())],
    )
    write_csv(
        output_dir / "question_type_counts.csv",
        ["dataset", "split", "question_type", "count"],
        [(*key, count) for key, count in sorted(question_counts.items())],
    )
    write_csv(
        output_dir / "revision_type_counts.csv",
        ["dataset", "split", "revision_type", "count"],
        [(*key, count) for key, count in sorted(revision_counts.items())],
    )

    representative_counts = Counter(
        (str(row["dataset"]), str(row["question_type"]), str(row["revision_type"]))
        for row in valid
    )
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in valid:
        key = (
            str(row["dataset"]),
            str(row["question_type"]),
            str(row["revision_type"]),
        )
        current = representatives.get(key)
        if current is None or confidence_rank.get(
            str(row.get("confidence")), 3
        ) < confidence_rank.get(str(current.get("confidence")), 3):
            representatives[key] = row
    few_shots = [
        {
            "dataset": key[0],
            "question_type": key[1],
            "revision_type": key[2],
            "count_all_splits": representative_counts[key],
            "repair_signature": row["repair_signature"],
            "generic_failure_scenario": row["generic_failure_scenario"],
            "generic_repair_rule": row["generic_repair_rule"],
            "confidence": row["confidence"],
            "source_split": row["split"],
            "source_id": row["id"],
            "revision_basis": row["revision_basis"],
        }
        for key, row in sorted(
            representatives.items(),
            key=lambda item: (
                item[0][0],
                -representative_counts[item[0]],
                item[0][1],
                item[0][2],
            ),
        )
    ]
    (output_dir / "few_shot_candidates.json").write_text(
        json.dumps(few_shots, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_split_counts = Counter(
        (str(row["dataset"]), str(row["split"])) for row in valid
    )
    report = {
        "n_classified": len(valid),
        "dataset_split_counts": {
            f"{dataset}/{split}": count
            for (dataset, split), count in sorted(dataset_split_counts.items())
        },
        "n_unique_pairs_by_dataset": {
            dataset: len({
                (question_type, revision_type)
                for row_dataset, _split, question_type, revision_type in pair_counts
                if row_dataset == dataset
            })
            for dataset in sorted({str(row["dataset"]) for row in valid})
        },
        "usage_this_process": dict(usage),
        "revision_basis": "task_structure_prior_without_observed_failure",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Dataset Type Taxonomy Audit",
        "",
        "> `revision_type` is a task-structure prior because this audit does not run "
        "or diagnose target-model trajectories.",
        "",
        f"- Classified items: `{len(valid)}`",
        f"- Few-shot pair candidates: `{len(few_shots)}`",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Train | Val | Test | Unique pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    datasets = sorted({str(row["dataset"]) for row in valid})
    for dataset in datasets:
        unique_pairs = len({
            (str(row["question_type"]), str(row["revision_type"]))
            for row in valid
            if row["dataset"] == dataset
        })
        lines.append(
            f"| {dataset} | {dataset_split_counts[(dataset, 'train')]} | "
            f"{dataset_split_counts[(dataset, 'val')]} | "
            f"{dataset_split_counts[(dataset, 'test')]} | {unique_pairs} |"
        )
    for dataset in datasets:
        lines.extend([
            "",
            f"## {dataset}",
            "",
            "| Question type | Revision type | Train | Val | Test | Total |",
            "|---|---|---:|---:|---:|---:|",
        ])
        keys = {
            (question_type, revision_type)
            for row_dataset, _split, question_type, revision_type in pair_counts
            if row_dataset == dataset
        }
        sorted_keys = sorted(
            keys,
            key=lambda key: (
                -sum(
                    pair_counts[(dataset, split, *key)]
                    for split in ("train", "val", "test")
                ),
                key,
            ),
        )
        for question_type, revision_type in sorted_keys:
            counts = [
                pair_counts[(dataset, split, question_type, revision_type)]
                for split in ("train", "val", "test")
            ]
            lines.append(
                f"| `{question_type}` | `{revision_type}` | {counts[0]} | "
                f"{counts[1]} | {counts[2]} | {sum(counts)} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    try:
        dataset_dirs = parse_dataset_dirs(args.dataset_dir, project_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[str, str, str, dict[str, Any], str]] = []
    split_sizes: dict[str, int] = {}
    for dataset in args.datasets:
        catalog = load_catalog(project_root, dataset)
        for split in args.splits:
            items = load_items(dataset_dirs[dataset], split)
            if args.limit_per_split > 0:
                items = items[:args.limit_per_split]
            split_sizes[f"{dataset}/{split}"] = len(items)
            for index, item in enumerate(items):
                work.append(
                    (
                        dataset,
                        split,
                        item_id(item, index),
                        item_card(dataset, item, project_root, args.max_item_chars),
                        catalog,
                    )
                )

    print("============================================================")
    print("  Dataset question/revision taxonomy audit")
    print("============================================================")
    print(f"  datasets:    {' '.join(args.datasets)}")
    print(f"  splits:      {' '.join(args.splits)}")
    print(f"  model:       {args.model}")
    print(f"  base_url:    {args.base_url or '(dry-run)'}")
    print(f"  workers:     {args.workers}")
    print(f"  output:      {output_dir}")
    print(f"  total items: {len(work)}")
    for name, size in split_sizes.items():
        print(f"    {name}: {size}")
    print("============================================================")
    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"API key is empty: export {args.api_key_env}")
    records_path = output_dir / "records.jsonl"
    existing_rows, completed = read_existing(records_path)
    pending = [
        item for item in work
        if (item[0], item[1], item[2]) not in completed
    ]
    print(f"[resume] completed={len(completed)} pending={len(pending)}")
    client = TaxonomyClient(args, api_key)
    new_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()
    with records_path.open("a", encoding="utf-8") as output_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    client.classify,
                    dataset=dataset,
                    split=split,
                    record_id=record_id,
                    card=card,
                    catalog=catalog,
                ): (dataset, split, record_id)
                for dataset, split, record_id, card, catalog in pending
            }
            for done, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                row = future.result()
                if row.get("error"):
                    errors.append(row)
                else:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    new_rows.append(row)
                if done == 1 or done % 50 == 0 or done == len(futures):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"[progress] {done}/{len(futures)} "
                        f"ok={len(new_rows)} errors={len(errors)} "
                        f"rate={done / elapsed:.2f} items/s"
                    )

    if errors:
        (output_dir / "errors.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
            encoding="utf-8",
        )
    all_rows = existing_rows + new_rows
    write_reports(output_dir, all_rows, client.usage)
    print(
        f"[done] classified={len(all_rows)} new={len(new_rows)} errors={len(errors)} "
        f"summary={output_dir / 'summary.md'}"
    )


if __name__ == "__main__":
    main()
