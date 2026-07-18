#!/usr/bin/env python3
"""Verify that a minimal one-epoch PatchTree smoke run really trained."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def nonempty_answer(row: dict) -> bool:
    for key in ("predicted_answer", "response", "predicted_text", "final_answer"):
        if str(row.get(key) or "").strip():
            return True
    return False


def verify_dataset(out_base: Path, dataset: str, expected_limit: int) -> list[str]:
    root = out_base / dataset
    errors: list[str] = []
    warnings: list[str] = []
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return [f"{dataset}: missing {summary_path}"]

    summary = load_json(summary_path)
    config = summary.get("config") or {}
    if int(config.get("num_epochs") or 0) != 1:
        errors.append(f"num_epochs={config.get('num_epochs')!r}, expected 1")
    if int(config.get("train_size") or 0) != expected_limit:
        errors.append(
            f"train_size={config.get('train_size')!r}, expected limited split size {expected_limit}"
        )
    if int(config.get("batch_size") or 0) != expected_limit:
        errors.append(f"batch_size={config.get('batch_size')!r}, expected {expected_limit}")
    if int(summary.get("total_steps") or 0) != 1:
        errors.append(f"total_steps={summary.get('total_steps')!r}, expected 1")
    epoch_stats = summary.get("epoch_stats") or []
    if len(epoch_stats) != 1 or int((epoch_stats[0] or {}).get("epoch") or 0) != 1:
        errors.append(f"epoch_stats does not contain exactly completed epoch 1: {epoch_stats!r}")
    if bool(config.get("eval_test")):
        errors.append("eval_test is true; smoke must not run the test split")
    if str(config.get("type_guided_version") or "") != "v2":
        errors.append(f"type_guided_version={config.get('type_guided_version')!r}, expected 'v2'")

    rollout_path = root / "steps" / "step_0001" / "rollout" / "results.jsonl"
    if not rollout_path.is_file():
        errors.append(f"missing primary training rollout: {rollout_path}")
        rows: list[dict] = []
    else:
        rows = load_jsonl(rollout_path)
        if len(rows) < expected_limit:
            errors.append(f"only {len(rows)} training rollout rows, expected at least {expected_limit}")
        if dataset == "spreadsheetbench":
            if rows and not any(bool(row.get("llm_ok")) for row in rows):
                errors.append("SpreadsheetBench has no completed LLM episode")
            infrastructure_failures = [
                row for row in rows
                if str(row.get("phase") or "").lower() in {"timeout", "error"}
                or str(row.get("fail_reason") or "").lower().startswith(
                    ("llm-call-failed", "unexpected")
                )
            ]
            if infrastructure_failures:
                errors.append(
                    "SpreadsheetBench has "
                    f"{len(infrastructure_failures)} timeout/infrastructure-failure rows"
                )
        else:
            if rows and not any(nonempty_answer(row) for row in rows):
                errors.append("all training rollout answers/responses are empty")
            if rows and not any(bool(row.get("agent_ok")) for row in rows):
                errors.append("all training rollout rows have agent_ok=false")

    if dataset == "officeqa" and rows:
        if not all(bool(row.get("use_local_tools")) for row in rows):
            errors.append("OfficeQA has rollout rows with use_local_tools=false")
        evidence_ok = any(
            bool(row.get("resolved_source_paths"))
            or bool(row.get("oracle_parsed_pages_included"))
            for row in rows
        )
        if not evidence_ok:
            errors.append("OfficeQA resolved no document path or oracle parsed-page evidence")
        conversations = list((root / "steps" / "step_0001" / "rollout").glob(
            "predictions/**/conversation.json"
        ))
        tool_calls = 0
        for path in conversations:
            try:
                content = json.dumps(load_json(path), ensure_ascii=False)
            except (ValueError, OSError):
                # conversation.json is normally a list, so load it directly.
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
            tool_calls += content.lower().count("tool_call")
        if tool_calls == 0:
            warnings.append(
                "OfficeQA sample produced no tool_call event; document evidence was present, "
                "but inspect conversation.json before a formal run"
            )
        if all(float(row.get("hard") or 0) == 0.0 for row in rows):
            warnings.append(
                "OfficeQA smoke accuracy is all zero; non-empty answers/evidence passed, "
                "so inspect predicted_answer versus ground_truth before formal training"
            )

    if dataset == "docvqa" and rows:
        if not any(row.get("image_paths") for row in rows):
            errors.append("DocVQA rollout contains no image_paths")

    if errors:
        return [f"{dataset}: {message}" for message in errors]

    print(
        f"[verify/pass] {dataset}: epoch=1 steps=1 train_size={expected_limit} "
        f"rollout_rows={len(rows)}"
    )
    for warning in warnings:
        print(f"[verify/warn] {warning}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-base", required=True, type=Path)
    parser.add_argument("--expected-limit", required=True, type=int)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    all_errors: list[str] = []
    for dataset in args.datasets:
        all_errors.extend(verify_dataset(args.out_base, dataset, args.expected_limit))
    if all_errors:
        for error in all_errors:
            print(f"[verify/fail] {error}")
        return 1
    print(f"[verify/pass] all datasets passed: {' '.join(args.datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
