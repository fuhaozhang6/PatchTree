#!/usr/bin/env python3
"""Check the phase-2 full-val gate and emit the conditional G4 TEST manifest."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from skillopt.evaluation.gate import select_gate_score
from skillopt.utils import compute_score


def _read_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        item_id = str(row.get("id") or "")
        if not item_id or item_id in rows:
            raise ValueError(f"missing/duplicate id at {path}:{line_no}")
        rows[item_id] = row
    return rows


def _score(rows: dict[str, dict], weight: float) -> dict[str, float | int]:
    hard, soft = compute_score(list(rows.values()))
    return {
        "n": len(rows),
        "hard": hard,
        "soft": soft,
        "mixed": select_gate_score(hard, soft, "mixed", weight),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-report", required=True)
    parser.add_argument("--combo-results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mixed-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.finalize_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parent_results = _read_results(
        Path(report["inputs"]["parent_results"]["path"])
    )
    root_results = _read_results(
        Path(report["inputs"]["root_results"]["path"])
    )
    combo_results = _read_results(Path(args.combo_results).resolve())
    if not (set(parent_results) == set(root_results) == set(combo_results)):
        raise ValueError("parent/root/combo full-val result ID sets differ")
    parent_score = _score(parent_results, args.mixed_weight)
    root_score = _score(root_results, args.mixed_weight)
    combo_score = _score(combo_results, args.mixed_weight)
    root_rejected = float(root_score["mixed"]) <= float(parent_score["mixed"])
    complementarity = bool(
        report.get("complementarity", {}).get("topdown_complementarity_pass")
    )
    combo_accepted = float(combo_score["mixed"]) > float(parent_score["mixed"])
    ready = root_rejected and complementarity and combo_accepted

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_report = {
        "root_rejected_under_fixed_protocol": root_rejected,
        "child_complementarity_pass": complementarity,
        "combo_full_val_accepted": combo_accepted,
        "ready_for_test": ready,
        "scores": {
            "parent": parent_score,
            "root": root_score,
            "combo": combo_score,
        },
        "threshold": "combo_mixed > parent_mixed; root_mixed <= parent_mixed",
    }
    (out_dir / "topdown_gate_report.json").write_text(
        json.dumps(gate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if ready:
        manifest_path = out_dir / "topdown_test_manifest.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                delimiter="\t",
                fieldnames=["run_name", "skill_path"],
            )
            writer.writeheader()
            writer.writerows([
                {
                    "run_name": "td_parent",
                    "skill_path": report["inputs"]["parent_skill"]["path"],
                },
                {
                    "run_name": "td_root",
                    "skill_path": report["inputs"]["root_skill"]["path"],
                },
                {
                    "run_name": "td_combo",
                    "skill_path": report["combination"]["candidate_skill_path"],
                },
            ])
        print(f"[ready] {manifest_path}")
        raise SystemExit(0)
    print(f"[skip-test] top-down gate failed: {gate_report}")
    raise SystemExit(3)


if __name__ == "__main__":
    main()
