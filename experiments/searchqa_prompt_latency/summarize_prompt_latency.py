#!/usr/bin/env python3
"""Print compact tables for SearchQA prompt latency/effect outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    summary_path = args.out_dir / "summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    summaries = sorted(summaries, key=lambda x: (str(x.get("prompt_variant")), int(x.get("workers") or 0)))

    headers = [
        "prompt",
        "workers",
        "n",
        "hard",
        "soft",
        "wall_s",
        "req/s",
        "avg_comp",
        "p50",
        "p95",
        "max",
        ">500",
        ">1000",
        ">4000",
        "length",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in summaries:
        rows = load_rows(Path(s.get("jsonl", "")))
        if not rows and s.get("jsonl"):
            rows = load_rows(args.out_dir / Path(str(s["jsonl"])).name)
        comps = [int(r.get("completion_tokens") or 0) for r in rows if r.get("ok")]
        gt500 = sum(1 for x in comps if x > 500)
        gt1000 = sum(1 for x in comps if x > 1000)
        gt4000 = sum(1 for x in comps if x > 4000)
        finish = s.get("finish_reasons") or {}
        line = [
            str(s.get("prompt_variant")),
            str(s.get("workers")),
            str(s.get("n")),
            fmt(s.get("hard")),
            fmt(s.get("soft")),
            fmt(s.get("wall_s"), 2),
            fmt(s.get("requests_per_s"), 2),
            fmt(s.get("avg_completion_tokens"), 1),
            fmt(s.get("p50_completion_tokens"), 0),
            fmt(s.get("p95_completion_tokens"), 0),
            fmt(s.get("max_completion_tokens"), 0),
            str(gt500),
            str(gt1000),
            str(gt4000),
            str(finish.get("length", 0)),
        ]
        print("| " + " | ".join(line) + " |")

    print("\nTop long-tail responses:")
    all_rows: list[dict[str, Any]] = []
    for s in summaries:
        rows = load_rows(Path(s.get("jsonl", "")))
        if not rows and s.get("jsonl"):
            rows = load_rows(args.out_dir / Path(str(s["jsonl"])).name)
        for row in rows:
            row.setdefault("prompt_variant", s.get("prompt_variant"))
            row.setdefault("workers", s.get("workers"))
        all_rows.extend(rows)
    for row in sorted(all_rows, key=lambda r: int(r.get("completion_tokens") or 0), reverse=True)[: args.top_k]:
        question = str(row.get("question") or "").replace("\n", " ")[:100]
        preview = str(row.get("response") or row.get("content_preview") or "").replace("\n", " ")[:160]
        print(
            f"- prompt={row.get('prompt_variant')} workers={row.get('workers')} "
            f"id={row.get('id')} comp={row.get('completion_tokens')} "
            f"dt={fmt(row.get('dt_s'), 2)} hard={row.get('hard')} "
            f"q={question!r} preview={preview!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
