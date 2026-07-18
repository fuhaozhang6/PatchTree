#!/usr/bin/env python3
"""Analyze a Type-Guided Merge Tree V2 training run.

This is an offline artifact analyzer. It does not call models. Given an output
directory produced by ``scripts/cli/train.py`` it summarizes:

- training/test improvement;
- accepted root vs accepted leaf-fallback updates;
- PatchRecord, leaf, and root edit distributions;
- fallback validation behavior;
- improved/regressed test examples;
- timing/token/skill-size signals.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common()}


def _pair_key(question_type: str | None, revision_type: str | None) -> str:
    return f"{question_type or 'unknown'} / {revision_type or 'unknown'}"


def _score(result: dict) -> float | None:
    value = result.get("candidate_gate_score")
    if value is None:
        value = result.get("selection_hard")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short(text: Any, n: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:n]


def _extract_test_comparison(run_root: Path, top_n: int) -> tuple[dict, list[dict], list[dict]]:
    baseline = _read_jsonl(run_root / "test_eval_baseline" / "results.jsonl")
    best = _read_jsonl(run_root / "test_eval" / "results.jsonl")
    if not baseline or not best:
        return {}, [], []

    by_base = {str(row.get("id")): row for row in baseline if row.get("id") is not None}
    by_best = {str(row.get("id")): row for row in best if row.get("id") is not None}
    ids = sorted(set(by_base) & set(by_best))

    improved: list[dict] = []
    regressed: list[dict] = []
    unchanged_good = 0
    unchanged_bad = 0
    for sample_id in ids:
        b = by_base[sample_id]
        c = by_best[sample_id]
        b_hard = bool(b.get("hard"))
        c_hard = bool(c.get("hard"))
        row = {
            "id": sample_id,
            "question": b.get("question") or c.get("question") or "",
            "baseline_answer": b.get("predicted_answer") or b.get("answer") or b.get("response") or "",
            "best_answer": c.get("predicted_answer") or c.get("answer") or c.get("response") or "",
            "gold_answers": b.get("gold_answers") or c.get("gold_answers") or b.get("answers") or c.get("answers") or [],
            "baseline_soft": b.get("soft"),
            "best_soft": c.get("soft"),
        }
        if not b_hard and c_hard:
            improved.append(row)
        elif b_hard and not c_hard:
            regressed.append(row)
        elif b_hard and c_hard:
            unchanged_good += 1
        else:
            unchanged_bad += 1

    def avg_soft(rows: list[dict]) -> float:
        return sum(float(row.get("soft", 0) or 0) for row in rows) / max(len(rows), 1)

    summary = {
        "n": len(ids),
        "baseline_hard": sum(1 for row in baseline if row.get("hard")) / max(len(baseline), 1),
        "best_hard": sum(1 for row in best if row.get("hard")) / max(len(best), 1),
        "baseline_soft": avg_soft(baseline),
        "best_soft": avg_soft(best),
        "improved": len(improved),
        "regressed": len(regressed),
        "unchanged_good": unchanged_good,
        "unchanged_bad": unchanged_bad,
        "net": len(improved) - len(regressed),
    }
    return summary, improved[:top_n], regressed[:top_n]


def analyze(run_root: Path, *, top_n: int = 20) -> dict:
    summary = _load_json(run_root / "summary.json", {}) or {}
    config = _load_json(run_root / "config.json", {}) or {}
    history = _load_json(run_root / "history.json", []) or []

    steps_dir = run_root / "steps"
    step_rows: list[dict] = []
    patch_status = Counter()
    question_types = Counter()
    revision_types = Counter()
    patch_pairs = Counter()
    leaf_pairs = Counter()
    q_values = Counter()
    root_edit_counts = Counter()
    fallback_kept_counts = Counter()
    fallback_attempts = 0
    fallback_accepts = 0
    fallback_leaf_results: list[dict] = []
    cache_status = Counter()
    cache_reports = 0

    accepted_root_steps: list[int] = []
    accepted_fallback_steps: list[int] = []

    for step_rec in history:
        step = int(step_rec.get("step") or 0)
        step_dir = steps_dir / f"step_{step:04d}"
        merge_artifact = _load_json(step_dir / "type_guided_v2_merge_artifact.json", {}) or {}
        fallback = _load_json(step_dir / "type_guided_v2_fallback.json", {}) or {}
        patch_records = _load_json(step_dir / "type_guided_v2_patch_records.json", []) or []
        cache_report = _load_json(step_dir / "type_guided_v2_cache_report.json", {}) or {}

        for record in patch_records:
            patch_status[str(record.get("status") or "unknown")] += 1
            question_types[str(record.get("question_type") or "unknown")] += 1
            revision_types[str(record.get("revision_type") or "unknown")] += 1
            patch_pairs[_pair_key(record.get("question_type"), record.get("revision_type"))] += 1
            q_values[str(record.get("q_i"))] += 1

        leaf_patches = merge_artifact.get("leaf_patches") or []
        for leaf in leaf_patches:
            if isinstance(leaf, dict):
                leaf_pairs[_pair_key(leaf.get("question_type"), leaf.get("revision_type"))] += 1
        root_patch = merge_artifact.get("root_patch") or {}
        root_edits = root_patch.get("edits") if isinstance(root_patch, dict) else []
        root_edit_counts[len(root_edits or [])] += 1

        if fallback:
            if fallback.get("attempted"):
                fallback_attempts += 1
            if fallback.get("accepted"):
                fallback_accepts += 1
            kept = fallback.get("kept_leaf_ids") or []
            fallback_kept_counts[len(kept)] += 1
            for leaf_result in fallback.get("leaf_results") or []:
                fallback_leaf_results.append({
                    "step": step,
                    "leaf_id": leaf_result.get("leaf_id"),
                    "question_type": leaf_result.get("question_type"),
                    "revision_type": leaf_result.get("revision_type"),
                    "support_count": leaf_result.get("support_count"),
                    "gate_score": leaf_result.get("gate_score"),
                    "hard": leaf_result.get("hard"),
                    "soft": leaf_result.get("soft"),
                    "kept": leaf_result.get("kept"),
                })

        for batch in cache_report.get("patch_record_batches") or []:
            for report in batch.get("analyst_reports") or []:
                cache_reports += 1
                cache_status[str(report.get("status") or "unknown")] += 1
                if report.get("cache_hit"):
                    cache_status["cache_hit_flag"] += 1

        action = str(step_rec.get("action") or "")
        fallback_selected = step_rec.get("type_guided_fallback_selected")
        if action in {"accept", "accept_new_best", "force_accept"}:
            if fallback_selected == "leaf_combination":
                accepted_fallback_steps.append(step)
            else:
                accepted_root_steps.append(step)

        tg = step_rec.get("type_guided_merge") or {}
        fb = step_rec.get("type_guided_leaf_fallback") or {}
        step_rows.append({
            "step": step,
            "epoch": step_rec.get("epoch"),
            "action": action,
            "accepted_by": "leaf_fallback" if fallback_selected == "leaf_combination" else ("root" if action in {"accept", "accept_new_best", "force_accept"} else ""),
            "rollout_hard": step_rec.get("rollout_hard"),
            "selection_hard": step_rec.get("selection_hard"),
            "selection_soft": step_rec.get("selection_soft"),
            "candidate_gate_score": step_rec.get("candidate_gate_score"),
            "current_score": step_rec.get("current_score"),
            "best_score": step_rec.get("best_score"),
            "patch_records": step_rec.get("n_type_guided_v2_records"),
            "leaf_groups": tg.get("n_leaf_groups"),
            "dropped_groups": tg.get("n_dropped_groups"),
            "root_edits": len(root_edits or []),
            "fallback_attempted": fb.get("attempted"),
            "fallback_accepted": fb.get("accepted"),
            "fallback_kept": fb.get("n_kept"),
            "candidate_skill_len": step_rec.get("candidate_skill_len"),
            "skill_len": step_rec.get("skill_len"),
        })

    timing_totals = Counter()
    apply_totals = Counter()
    actions = Counter(str(row.get("action") or "unknown") for row in history)
    epoch_actions: dict[str, Counter] = defaultdict(Counter)
    for row in history:
        epoch_actions[str(row.get("epoch"))][str(row.get("action") or "unknown")] += 1
        for key, value in (row.get("timing") or {}).items():
            try:
                timing_totals[key] += float(value)
            except (TypeError, ValueError):
                pass
        for key, value in (row.get("edit_apply_summary") or {}).items():
            try:
                apply_totals[key] += int(value)
            except (TypeError, ValueError):
                pass

    skill_stats = {}
    initial_skill = run_root / "skills" / "skill_v0000.md"
    best_skill = run_root / "best_skill.md"
    for label, path in [("initial", initial_skill), ("best", best_skill)]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            skill_stats[label] = {
                "chars": len(text),
                "lines": text.count("\n") + 1,
            }

    test_summary, improved, regressed = _extract_test_comparison(run_root, top_n)
    q_numeric = []
    for value, count in q_values.items():
        try:
            q_numeric.extend([float(value)] * count)
        except ValueError:
            pass

    return {
        "run_root": str(run_root),
        "config": {
            key: config.get(key)
            for key in [
                "type_guided_merge",
                "type_guided_version",
                "type_guided_rollout_repeats",
                "type_guided_min_support",
                "type_guided_max_leaf_groups",
                "type_guided_max_patch_records",
                "type_guided_leaf_fallback",
                "train_size",
                "batch_size",
                "num_epochs",
                "sel_env_num",
                "test_env_num",
                "workers",
                "analyst_workers",
            ]
        },
        "summary": {
            key: summary.get(key)
            for key in [
                "baseline_selection_hard",
                "best_selection_hard",
                "final_selection_hard",
                "baseline_test_hard",
                "baseline_test_soft",
                "test_hard",
                "test_soft",
                "final_test_hard",
                "final_test_soft",
                "test_delta_hard",
                "final_test_delta_hard",
                "best_step",
                "total_steps",
                "total_accepts",
                "total_rejects",
                "total_skips",
                "total_wall_time_s",
            ]
        },
        "actions": _counter_to_dict(actions),
        "epoch_actions": {
            epoch: _counter_to_dict(counter)
            for epoch, counter in sorted(epoch_actions.items())
        },
        "accepted_update_source": {
            "root_accept_steps": accepted_root_steps,
            "leaf_fallback_accept_steps": accepted_fallback_steps,
            "root_accept_count": len(accepted_root_steps),
            "leaf_fallback_accept_count": len(accepted_fallback_steps),
        },
        "patch_records": {
            "status": _counter_to_dict(patch_status),
            "question_types": _counter_to_dict(question_types),
            "revision_types": _counter_to_dict(revision_types),
            "top_pairs": _counter_to_dict(patch_pairs),
            "q_values": _counter_to_dict(q_values),
            "q_mean": mean(q_numeric) if q_numeric else None,
            "total": sum(patch_status.values()),
        },
        "leaf_and_root": {
            "leaf_pairs": _counter_to_dict(leaf_pairs),
            "root_edit_counts": _counter_to_dict(root_edit_counts),
        },
        "fallback": {
            "attempts": fallback_attempts,
            "accepts": fallback_accepts,
            "kept_counts": _counter_to_dict(fallback_kept_counts),
            "leaf_results": fallback_leaf_results,
        },
        "cache": {
            "reports": cache_reports,
            "status": _counter_to_dict(cache_status),
            "cache_files": len(list((run_root / "type_guided_cache").glob("patch_record_*.json"))),
        },
        "test_comparison": test_summary,
        "improved_examples": improved,
        "regressed_examples": regressed,
        "timing_totals_s": {k: round(v, 3) for k, v in timing_totals.items()},
        "token_summary": summary.get("token_summary") or {},
        "apply_totals": _counter_to_dict(apply_totals),
        "skill_stats": skill_stats,
        "steps": step_rows,
    }


def write_outputs(analysis: dict, out_dir: Path, *, top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    step_fields = [
        "step",
        "epoch",
        "action",
        "accepted_by",
        "rollout_hard",
        "selection_hard",
        "selection_soft",
        "candidate_gate_score",
        "current_score",
        "best_score",
        "patch_records",
        "leaf_groups",
        "dropped_groups",
        "root_edits",
        "fallback_attempted",
        "fallback_accepted",
        "fallback_kept",
        "candidate_skill_len",
        "skill_len",
    ]
    with (out_dir / "analysis_steps.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields)
        writer.writeheader()
        for row in analysis["steps"]:
            writer.writerow({field: row.get(field) for field in step_fields})

    for name in ["improved_examples", "regressed_examples"]:
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in analysis.get(name, []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = _render_markdown(analysis, top_n=top_n)
    (out_dir / "analysis_report.md").write_text(md, encoding="utf-8")


def _render_markdown(analysis: dict, *, top_n: int) -> str:
    s = analysis["summary"]
    accepted = analysis["accepted_update_source"]
    test = analysis.get("test_comparison") or {}
    patch = analysis["patch_records"]
    fallback = analysis["fallback"]
    skill = analysis.get("skill_stats") or {}
    tokens = analysis.get("token_summary") or {}
    total_tokens = (tokens.get("_total") or {}).get("total_tokens")

    lines = [
        "# Type-Guided V2 Run Analysis",
        "",
        f"Run root: `{analysis['run_root']}`",
        "",
        "## Headline",
        "",
        f"- Selection: `{s.get('baseline_selection_hard')}` -> `{s.get('best_selection_hard')}`",
        f"- Test hard: `{s.get('baseline_test_hard')}` -> `{s.get('test_hard')}`",
        f"- Test soft: `{s.get('baseline_test_soft')}` -> `{s.get('test_soft')}`",
        f"- Best step: `{s.get('best_step')}`",
        f"- Actions: `{analysis.get('actions')}`",
        f"- Accepted by root: `{accepted.get('root_accept_count')}` steps `{accepted.get('root_accept_steps')}`",
        f"- Accepted by leaf fallback: `{accepted.get('leaf_fallback_accept_count')}` steps `{accepted.get('leaf_fallback_accept_steps')}`",
        "",
        "## Test Movement",
        "",
        f"- Improved: `{test.get('improved')}`",
        f"- Regressed: `{test.get('regressed')}`",
        f"- Net: `{test.get('net')}`",
        f"- Unchanged good/bad: `{test.get('unchanged_good')}` / `{test.get('unchanged_bad')}`",
        "",
        "## PatchRecord Distribution",
        "",
        f"- Total PatchRecords: `{patch.get('total')}`",
        f"- Status: `{patch.get('status')}`",
        f"- q_i mean: `{patch.get('q_mean')}`",
        "",
        "Top PatchRecord type pairs:",
        "",
    ]
    for key, value in list((patch.get("top_pairs") or {}).items())[:10]:
        lines.append(f"- `{key}`: {value}")

    lines.extend([
        "",
        "## Leaf Fallback",
        "",
        f"- Attempts: `{fallback.get('attempts')}`",
        f"- Accepts: `{fallback.get('accepts')}`",
        f"- Kept leaf counts: `{fallback.get('kept_counts')}`",
        "",
        "## Cost And Size",
        "",
        f"- Total wall time seconds: `{s.get('total_wall_time_s')}`",
        f"- Total tokens: `{total_tokens}`",
        f"- Skill chars initial/best: `{(skill.get('initial') or {}).get('chars')}` -> `{(skill.get('best') or {}).get('chars')}`",
        "",
        f"## Improved Examples Top {top_n}",
        "",
    ])
    for row in analysis.get("improved_examples", []):
        lines.append(f"- `{row.get('id')}` Q: {_short(row.get('question'))}")
        lines.append(f"  - baseline: `{_short(row.get('baseline_answer'), 120)}`")
        lines.append(f"  - best: `{_short(row.get('best_answer'), 120)}`")
        lines.append(f"  - gold: `{row.get('gold_answers')}`")

    lines.extend(["", f"## Regressed Examples Top {top_n}", ""])
    for row in analysis.get("regressed_examples", []):
        lines.append(f"- `{row.get('id')}` Q: {_short(row.get('question'))}")
        lines.append(f"  - baseline: `{_short(row.get('baseline_answer'), 120)}`")
        lines.append(f"  - best: `{_short(row.get('best_answer'), 120)}`")
        lines.append(f"  - gold: `{row.get('gold_answers')}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Output directory of a completed training run.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for analysis files. Defaults to run_root/analysis.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of improved/regressed examples to export in report.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not (run_root / "summary.json").exists():
        raise SystemExit(f"Missing summary.json under {run_root}")
    out_dir = args.out_dir.resolve() if args.out_dir else run_root / "analysis"
    analysis = analyze(run_root, top_n=max(args.top_n, 0))
    write_outputs(analysis, out_dir, top_n=max(args.top_n, 0))

    accepted = analysis["accepted_update_source"]
    test = analysis.get("test_comparison") or {}
    print(f"[analysis] wrote {out_dir}")
    print(
        "[analysis] root_accepts="
        f"{accepted.get('root_accept_count')} leaf_fallback_accepts="
        f"{accepted.get('leaf_fallback_accept_count')} "
        f"test_net={test.get('net')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
