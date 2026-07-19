#!/usr/bin/env python3
"""Paired statistical analysis for the SearchQA PatchTree verdict."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


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


def _mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = min(n01, n10)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def _paired_bootstrap(
    deltas: list[float],
    *,
    repeats: int = 20_000,
    seed: int = 20260719,
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(repeats)
    ]
    means.sort()
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def compare(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    candidate_name: str,
    reference_name: str,
) -> dict:
    if set(candidate) != set(reference):
        raise ValueError(
            f"ID mismatch: {candidate_name} vs {reference_name}"
        )
    ids = sorted(candidate)
    deltas: list[float] = []
    n01 = 0
    n10 = 0
    fixed: list[str] = []
    broken: list[str] = []
    for item_id in ids:
        cand = 1 if float(candidate[item_id].get("hard", 0) or 0) > 0 else 0
        ref = 1 if float(reference[item_id].get("hard", 0) or 0) > 0 else 0
        deltas.append(float(cand - ref))
        if cand and not ref:
            n01 += 1
            fixed.append(item_id)
        elif ref and not cand:
            n10 += 1
            broken.append(item_id)
    low, high = _paired_bootstrap(deltas)
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "n": len(ids),
        "candidate_correct": sum(
            1 for row in candidate.values() if float(row.get("hard", 0) or 0) > 0
        ),
        "reference_correct": sum(
            1 for row in reference.values() if float(row.get("hard", 0) or 0) > 0
        ),
        "n01_candidate_only": n01,
        "n10_reference_only": n10,
        "delta_correct": n01 - n10,
        "delta_rate": (n01 - n10) / max(len(ids), 1),
        "mcnemar_exact_p": _mcnemar_exact(n01, n10),
        "paired_bootstrap_95ci": [low, high],
        "fixed_ids": fixed,
        "broken_ids": broken,
    }


def _holm_two(p_values: list[float]) -> list[float]:
    if len(p_values) != 2:
        raise ValueError("this verdict uses exactly two layer comparisons")
    order = sorted(range(2), key=lambda index: p_values[index])
    adjusted = [1.0, 1.0]
    first, second = order
    adjusted[first] = min(1.0, p_values[first] * 2)
    adjusted[second] = max(adjusted[first], min(1.0, p_values[second]))
    return adjusted


def _increment(parent: str, candidate: str) -> dict:
    return {
        "character_delta": len(candidate) - len(parent),
        "positive_character_delta": max(0, len(candidate) - len(parent)),
        "whitespace_token_delta": len(candidate.split()) - len(parent.split()),
        "positive_whitespace_token_delta": max(
            0, len(candidate.split()) - len(parent.split())
        ),
    }


def _structural_row(replay_dir: Path, main_step: int) -> dict:
    rows = _read_json(replay_dir / "all_steps_structure_audit.json")
    row = next((item for item in rows if int(item["step"]) == main_step), None)
    if row is None:
        raise ValueError(f"main step {main_step} missing from structure audit")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir).resolve()
    eval_root = Path(args.eval_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else eval_root
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["g0_parent", "g1_flat", "g2_leaf", "g3_tree"]
    results = {
        name: _read_results(eval_root / name / "test" / "results.jsonl")
        for name in names
    }
    comparisons = {
        "leaf_vs_flat": compare(
            results["g2_leaf"],
            results["g1_flat"],
            candidate_name="g2_leaf",
            reference_name="g1_flat",
        ),
        "tree_vs_leaf": compare(
            results["g3_tree"],
            results["g2_leaf"],
            candidate_name="g3_tree",
            reference_name="g2_leaf",
        ),
        "tree_vs_flat": compare(
            results["g3_tree"],
            results["g1_flat"],
            candidate_name="g3_tree",
            reference_name="g1_flat",
        ),
        "tree_vs_parent": compare(
            results["g3_tree"],
            results["g0_parent"],
            candidate_name="g3_tree",
            reference_name="g0_parent",
        ),
        "leaf_vs_parent": compare(
            results["g2_leaf"],
            results["g0_parent"],
            candidate_name="g2_leaf",
            reference_name="g0_parent",
        ),
    }
    layer_keys = ["leaf_vs_flat", "tree_vs_leaf"]
    adjusted = _holm_two([
        comparisons[key]["mcnemar_exact_p"] for key in layer_keys
    ])
    for key, value in zip(layer_keys, adjusted):
        comparisons[key]["holm_adjusted_p"] = value

    manifest = _read_json(replay_dir / "replay_manifest.json")
    main_step = int(manifest["main_step"])
    structure = _structural_row(replay_dir, main_step)
    parent = (
        replay_dir / "main" / "g0_parent" / "candidate_skill.md"
    ).read_text(encoding="utf-8")
    candidates = {
        name: (
            replay_dir / "main" / name / "candidate_skill.md"
        ).read_text(encoding="utf-8")
        for name in ["g1_flat", "g2_leaf", "g3_tree"]
    }
    increments = {
        name: _increment(parent, text) for name, text in candidates.items()
    }
    tree_flat = comparisons["tree_vs_flat"]
    tree_parent = comparisons["tree_vs_parent"]
    structural_pass = (
        structure["n_multi_leaf_mids"] >= 2
        and structure["multi_leaf_coverage"] >= 0.5
        and not structure["unknown_mid_leaf_ids"]
    )
    parent_noninferior = (
        tree_parent["delta_correct"] >= -7
        and tree_parent["paired_bootstrap_95ci"][0] > -0.015
    )
    harmful = (
        tree_parent["delta_correct"] <= -14
        and tree_parent["mcnemar_exact_p"] < 0.05
    )
    performance_pass = (
        tree_flat["delta_correct"] >= 14
        and tree_flat["mcnemar_exact_p"] < 0.05
        and parent_noninferior
        and structural_pass
    )
    flat_added = increments["g1_flat"]["positive_whitespace_token_delta"]
    tree_added = increments["g3_tree"]["positive_whitespace_token_delta"]
    compressed = flat_added > 0 and tree_added <= 0.7 * flat_added
    compression_pass = (
        not performance_pass
        and not harmful
        and tree_flat["delta_correct"] >= -7
        and tree_flat["paired_bootstrap_95ci"][0] > -0.015
        and compressed
        and structural_pass
    )
    tree_flat_harm = (
        tree_flat["delta_correct"] <= -14
        and tree_flat["mcnemar_exact_p"] < 0.05
    )
    if harmful or tree_flat_harm:
        verdict = "Red"
    elif performance_pass:
        verdict = "Green"
    elif compression_pass:
        verdict = "Yellow"
    else:
        verdict = "Gray"

    report = {
        "version": "searchqa_tree_verdict_analysis_v1",
        "main_step": main_step,
        "structure": structure,
        "comparisons": comparisons,
        "candidate_increments": increments,
        "checks": {
            "structural_pass": structural_pass,
            "tree_vs_parent_noninferior": parent_noninferior,
            "tree_harmful_vs_parent": harmful,
            "tree_performance_pass": performance_pass,
            "tree_compression_proxy_pass": compressed,
            "compression_verdict_pass": compression_pass,
        },
        "main_verdict": verdict,
        "note": (
            "Whitespace-token candidate deltas are an operational compression "
            "proxy; artifact-level shared_core/residual audit remains required."
        ),
    }
    json_path = out_dir / "main_verdict.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / "main_verdict.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# SearchQA PatchTree main verdict\n\n")
        file.write(f"Verdict: **{verdict}**\n\n")
        file.write("| comparison | candidate | reference | delta | n01 | n10 | p | 95% CI |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for key, row in comparisons.items():
            low, high = row["paired_bootstrap_95ci"]
            file.write(
                f"| {key} | {row['candidate_correct']} | "
                f"{row['reference_correct']} | {row['delta_correct']} | "
                f"{row['n01_candidate_only']} | {row['n10_reference_only']} | "
                f"{row['mcnemar_exact_p']:.6f} | "
                f"[{low:.4f}, {high:.4f}] |\n"
            )
        file.write("\n")
        file.write(f"- Structural pass: `{structural_pass}`\n")
        file.write(f"- Parent non-inferiority: `{parent_noninferior}`\n")
        file.write(f"- Performance pass: `{performance_pass}`\n")
        file.write(f"- Compression proxy pass: `{compressed}`\n")
    print(md_path)


if __name__ == "__main__":
    main()
