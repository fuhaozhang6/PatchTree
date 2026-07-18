#!/usr/bin/env python3
"""Evaluate the per-epoch final skill of a PatchTree run on the test split.

For every training epoch we take the skill that was in effect at the epoch's
last global step. Because the trainer unconditionally writes
``skills/skill_v{global_step:04d}.md`` at every step *and* re-saves that same
file after the end-of-epoch type-guided tail-bank stage
(trainer.py: ``_save_skill(out_root, global_step, current_skill)`` at both the
per-step and tail-bank sites), the file for the epoch's max global step already
reflects the epoch-level (tail-bank) computation. So:

    epoch_final_skill(e) = skills/skill_v{max_global_step_of_epoch_e:04d}.md

Identical skills (common once training plateaus and every later step is
reject/skip) are de-duplicated by content hash, so the expensive test rollout
runs only once per unique skill.

Outputs (under ``--result_dir``):
    epoch_eval_results.json   full structured results
    epoch_eval_table.md       markdown table: per-epoch val vs test
    epoch_eval_table.csv       same data as CSV

The "val" column is read from the training run itself (``history.json``): it is
the surviving skill's selection score at the epoch's last step, so no extra
val rollouts are spent. Only the test split is freshly evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── history.json parsing ─────────────────────────────────────────────────────

def load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"history.json not found under {out_root}")
    with open(path) as f:
        return json.load(f)


def epoch_final_steps(history: list[dict]) -> dict[int, int]:
    """Return {epoch: max_global_step_in_that_epoch}."""
    by_epoch: dict[int, int] = {}
    for rec in history:
        epoch = int(rec.get("epoch") or 0)
        step = int(rec.get("step") or 0)
        if epoch <= 0 or step <= 0:
            continue
        by_epoch[epoch] = max(by_epoch.get(epoch, 0), step)
    return dict(sorted(by_epoch.items()))


def running_val_scores(history: list[dict]) -> dict[int, dict]:
    """Map every global step -> the surviving skill's val hard/soft/gate.

    We walk history in order. At each accept/force-accept step the current
    skill becomes that step's candidate, so its val scores are that step's
    ``selection_hard``/``selection_soft``. On reject/skip the current skill is
    unchanged, so we carry the previous values forward. ``current_score`` is
    the gate score of the surviving skill and is always recorded.
    """
    out: dict[int, dict] = {}
    cur_hard: float | None = None
    cur_soft: float | None = None
    accept_actions = {"accept", "accept_new_best", "force_accept"}
    for rec in sorted(history, key=lambda r: int(r.get("step") or 0)):
        step = int(rec.get("step") or 0)
        if step <= 0:
            continue
        action = str(rec.get("action") or "")
        if action in accept_actions:
            if rec.get("selection_hard") is not None:
                cur_hard = float(rec["selection_hard"])
            if rec.get("selection_soft") is not None:
                cur_soft = float(rec["selection_soft"])
        out[step] = {
            "val_hard": cur_hard,
            "val_soft": cur_soft,
            "val_gate": rec.get("current_score"),
            "action": action,
            "epoch": int(rec.get("epoch") or 0),
        }
    return out


# ── model / adapter setup ────────────────────────────────────────────────────

def build_adapter(cfg: dict):
    from skillopt.envs.livemathematicianbench.adapter import (
        LiveMathematicianBenchAdapter,
    )
    import inspect

    sig = inspect.signature(LiveMathematicianBenchAdapter.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: cfg[k] for k in accepted if k in cfg}
    adapter = LiveMathematicianBenchAdapter(**kwargs)
    adapter.setup(cfg)
    return adapter


def configure_target(args) -> None:
    from skillopt.model import (
        configure_qwen_chat,
        set_target_backend,
        set_target_deployment,
        set_reasoning_effort,
    )

    set_target_backend("qwen_chat")
    set_target_deployment(args.target_model)
    configure_qwen_chat(
        target_base_url=args.base_url,
        target_api_key=args.api_key,
        target_temperature=args.temperature,
        target_timeout_seconds=args.timeout_seconds,
        target_max_tokens=args.max_completion_tokens,
        target_enable_thinking=args.enable_thinking,
    )
    set_reasoning_effort(None)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-epoch final skill test evaluator")
    p.add_argument("--out_root", required=True,
                   help="Training out_root, e.g. .../livemath_.../livemath")
    p.add_argument("--split_dir", required=True,
                   help="LiveMath split dir with train/val/test")
    p.add_argument("--result_dir", default="",
                   help="Where to write results (default: <out_root>/epoch_test_eval)")
    p.add_argument("--target_model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--base_url", default=os.environ.get(
        "TARGET_QWEN_CHAT_BASE_URL",
        os.environ.get("QWEN_CHAT_BASE_URL", "http://127.0.0.1:8000/v1")))
    p.add_argument("--api_key", default=os.environ.get(
        "TARGET_QWEN_CHAT_API_KEY", os.environ.get("QWEN_CHAT_API_KEY", "dummy")))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout_seconds", type=float, default=240.0)
    p.add_argument("--enable_thinking", default="false")
    p.add_argument("--max_completion_tokens", type=int, default=16384)
    p.add_argument("--workers", type=int, default=128,
                   help="rollout concurrency (ThreadPoolExecutor size)")
    p.add_argument("--test_env_num", type=int, default=0,
                   help="0 = full test split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_turns", type=int, default=1)
    p.add_argument("--eval_init", default="true",
                   help="also evaluate the initial skill as an epoch-0 baseline")
    p.add_argument("--init_skill", default="",
                   help="path to initial skill (default: config skill_init)")
    p.add_argument("--epochs", default="",
                   help="comma-separated epoch filter, e.g. 1,4,8,16 (default: all)")
    return p.parse_args()


def _truthy(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    args = parse_args()
    out_root = os.path.abspath(args.out_root)
    result_dir = os.path.abspath(
        args.result_dir or os.path.join(out_root, "epoch_test_eval"))
    os.makedirs(result_dir, exist_ok=True)

    from skillopt.utils import compute_score
    from skillopt.utils.scoring import skill_hash

    # ── discover epoch-final skills ────────────────────────────────────────
    history = load_history(out_root)
    finals = epoch_final_steps(history)
    val_by_step = running_val_scores(history)
    if not finals:
        raise RuntimeError("no epoch/step records found in history.json")

    epoch_filter: set[int] | None = None
    if args.epochs.strip():
        epoch_filter = {int(x) for x in args.epochs.split(",") if x.strip()}

    skills_dir = os.path.join(out_root, "skills")
    plan: list[dict] = []

    if _truthy(args.eval_init):
        init_path = args.init_skill or os.path.join(
            _PROJECT_ROOT,
            "skillopt/envs/livemathematicianbench/skills/initial.md")
        if os.path.exists(init_path):
            plan.append({
                "epoch": 0, "last_step": 0, "label": "init",
                "skill_path": os.path.abspath(init_path),
                "val_hard": None, "val_soft": None, "val_gate": None,
                "action": "baseline",
            })

    for epoch, last_step in finals.items():
        if epoch_filter is not None and epoch not in epoch_filter:
            continue
        skill_path = os.path.join(skills_dir, f"skill_v{last_step:04d}.md")
        vinfo = val_by_step.get(last_step, {})
        plan.append({
            "epoch": epoch,
            "last_step": last_step,
            "label": f"epoch_{epoch:02d}",
            "skill_path": skill_path,
            "val_hard": vinfo.get("val_hard"),
            "val_soft": vinfo.get("val_soft"),
            "val_gate": vinfo.get("val_gate"),
            "action": vinfo.get("action"),
        })

    # ── de-duplicate by skill content hash ─────────────────────────────────
    for row in plan:
        if not os.path.exists(row["skill_path"]):
            row["skill_hash"] = None
            row["skill_content"] = None
            continue
        with open(row["skill_path"]) as f:
            content = f.read()
        row["skill_content"] = content
        row["skill_hash"] = skill_hash(content)

    unique_hashes: dict[str, str] = {}
    for row in plan:
        h = row["skill_hash"]
        if h and h not in unique_hashes:
            unique_hashes[h] = row["skill_content"]

    print(f"  [plan] epochs to report : {len(plan)}")
    print(f"  [plan] unique skills    : {len(unique_hashes)} "
          f"(test rollout runs this many times)")
    print(f"  [plan] test split       : valid_unseen "
          f"(env_num={args.test_env_num or 'full'})")

    # ── configure target + adapter ─────────────────────────────────────────
    cfg = {
        "env": "livemathematicianbench",
        "split_mode": "split_dir",
        "split_dir": os.path.abspath(args.split_dir),
        "data_path": "",
        "split_output_dir": "",
        "max_turns": args.max_turns,
        "max_completion_tokens": args.max_completion_tokens,
        "workers": args.workers,
        "seed": args.seed,
        "limit": 0,
        "shuffle_choices": True,
        "use_theorem": False,
        "use_sketch": False,
    }
    configure_target(args)
    adapter = build_adapter(cfg)

    # Build the test env once (deterministic given seed) and reuse it.
    test_items = adapter.build_eval_env(args.test_env_num, "valid_unseen", args.seed)
    print(f"  [test] items = {len(test_items)}")

    # ── run one test rollout per unique skill ──────────────────────────────
    test_by_hash: dict[str, dict] = {}
    for i, (h, content) in enumerate(unique_hashes.items(), start=1):
        eval_dir = os.path.join(result_dir, "rollouts", f"skill_{h}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"\n  === test rollout {i}/{len(unique_hashes)}  hash={h} ===")
        results = adapter.rollout(test_items, content, eval_dir)
        hard, soft = compute_score(results)
        test_by_hash[h] = {"test_hard": hard, "test_soft": soft, "n": len(results)}
        print(f"      test_hard={hard:.4f}  test_soft={soft:.4f}  n={len(results)}")
        with open(os.path.join(eval_dir, "score.json"), "w") as f:
            json.dump(test_by_hash[h], f, indent=2)

    # ── assemble per-epoch rows ────────────────────────────────────────────
    for row in plan:
        h = row["skill_hash"]
        tinfo = test_by_hash.get(h, {}) if h else {}
        row["test_hard"] = tinfo.get("test_hard")
        row["test_soft"] = tinfo.get("test_soft")
        row["test_n"] = tinfo.get("n")
        row.pop("skill_content", None)

    # first epoch/label that introduced each hash (for dedup annotation)
    first_for_hash: dict[str, str] = {}
    for row in plan:
        h = row["skill_hash"]
        if h and h not in first_for_hash:
            first_for_hash[h] = row["label"]
    for row in plan:
        h = row["skill_hash"]
        row["same_skill_as"] = (
            first_for_hash.get(h) if h and first_for_hash.get(h) != row["label"]
            else None)

    # ── write outputs ──────────────────────────────────────────────────────
    results_path = os.path.join(result_dir, "epoch_eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "out_root": out_root,
            "split_dir": os.path.abspath(args.split_dir),
            "test_env_num": args.test_env_num,
            "test_items": len(test_items),
            "target_model": args.target_model,
            "n_unique_skills": len(unique_hashes),
            "rows": plan,
        }, f, indent=2, ensure_ascii=False)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "-"

    md_lines = [
        f"# Per-epoch skill: val vs test  ({os.path.basename(out_root)})",
        "",
        f"- test split: `valid_unseen`  (items={len(test_items)}, "
        f"env_num={args.test_env_num or 'full'})",
        f"- val: taken from training `history.json` (surviving skill's "
        f"selection score at the epoch's last step)",
        f"- unique skills actually evaluated on test: {len(unique_hashes)}",
        "",
        "| epoch | last_step | action | val_hard | val_gate | "
        "test_hard | test_soft | note |",
        "|------:|----------:|:-------|---------:|---------:|"
        "----------:|----------:|:-----|",
    ]
    csv_rows = []
    for row in plan:
        note = ""
        if row.get("skill_hash") is None:
            note = "MISSING skill file"
        elif row.get("same_skill_as"):
            note = f"same skill as {row['same_skill_as']}"
        md_lines.append(
            f"| {row['label']} | {row['last_step']} | {row.get('action') or '-'} "
            f"| {fmt(row.get('val_hard'))} | {fmt(row.get('val_gate'))} "
            f"| {fmt(row.get('test_hard'))} | {fmt(row.get('test_soft'))} "
            f"| {note} |")
        csv_rows.append({
            "epoch": row["epoch"],
            "label": row["label"],
            "last_step": row["last_step"],
            "action": row.get("action") or "",
            "val_hard": row.get("val_hard"),
            "val_gate": row.get("val_gate"),
            "test_hard": row.get("test_hard"),
            "test_soft": row.get("test_soft"),
            "test_n": row.get("test_n"),
            "skill_hash": row.get("skill_hash"),
            "note": note,
        })
    md_lines.append("")

    with open(os.path.join(result_dir, "epoch_eval_table.md"), "w") as f:
        f.write("\n".join(md_lines))
    with open(os.path.join(result_dir, "epoch_eval_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print("\n" + "\n".join(md_lines))
    print(f"\n  [saved] {results_path}")
    print(f"  [saved] {os.path.join(result_dir, 'epoch_eval_table.md')}")
    print(f"  [saved] {os.path.join(result_dir, 'epoch_eval_table.csv')}")


if __name__ == "__main__":
    main()
