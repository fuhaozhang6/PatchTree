#!/usr/bin/env python3
"""Finalize the conditional SearchQA top-down verdict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.tools.analyze_searchqa_tree_verdict import _read_results, compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown-root", required=True)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.topdown_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = root / "gate" / "topdown_gate_report.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    report: dict = {
        "gate": gate,
        "test_executed": False,
        "verdict": "Fail",
    }
    if gate.get("ready_for_test"):
        test_root = root / "test_eval"
        parent = _read_results(test_root / "td_parent" / "test" / "results.jsonl")
        rejected_root = _read_results(test_root / "td_root" / "test" / "results.jsonl")
        combo = _read_results(test_root / "td_combo" / "test" / "results.jsonl")
        combo_root = compare(
            combo,
            rejected_root,
            candidate_name="td_combo",
            reference_name="td_root",
        )
        combo_parent = compare(
            combo,
            parent,
            candidate_name="td_combo",
            reference_name="td_parent",
        )
        noninferior = (
            combo_parent["delta_correct"] >= -7
            and combo_parent["paired_bootstrap_95ci"][0] > -0.015
        )
        passed = (
            combo_root["delta_correct"] >= 14
            and combo_root["mcnemar_exact_p"] < 0.05
            and noninferior
        )
        report.update({
            "test_executed": True,
            "combo_vs_root": combo_root,
            "combo_vs_parent": combo_parent,
            "combo_vs_parent_noninferior": noninferior,
            "verdict": "Pass" if passed else "Fail",
        })
    json_path = out_dir / "topdown_verdict.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / "topdown_verdict.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# SearchQA top-down verdict\n\n")
        file.write(f"Verdict: **{report['verdict']}**\n\n")
        file.write(f"- Full-val ready: `{gate.get('ready_for_test')}`\n")
        file.write(f"- Test executed: `{report['test_executed']}`\n")
        if report["test_executed"]:
            row = report["combo_vs_root"]
            file.write(
                f"- Combo vs Root: delta={row['delta_correct']}, "
                f"p={row['mcnemar_exact_p']:.6f}\n"
            )
            file.write(
                f"- Combo vs Parent non-inferior: "
                f"`{report['combo_vs_parent_noninferior']}`\n"
            )
    print(md_path)


if __name__ == "__main__":
    main()
