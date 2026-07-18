#!/usr/bin/env python3
"""Concurrency sweep for an OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env",
        default="BENCH_API_KEY",
        help="Environment variable holding the API key (never pass keys on the CLI).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency-levels", default="1 2 4 8 16 32 64")
    parser.add_argument("--min-requests", type=int, default=16)
    parser.add_argument("--request-multiplier", type=int, default=2)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument(
        "--prompt-token-options",
        default="256 512 1024 2048",
        help="Approximate input token sizes, sampled evenly across requests.",
    )
    parser.add_argument(
        "--max-token-options",
        default="256 512 1024",
        help="Maximum output token sizes, sampled evenly across requests.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--round-pause-seconds", type=float, default=2)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--stop-on-unstable", action="store_true")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--provider",
        choices=("generic", "deepseek", "vllm"),
        default="generic",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        args.concurrency_levels = parse_int_options(args.concurrency_levels)
        args.prompt_token_options = parse_int_options(args.prompt_token_options)
        args.max_token_options = parse_int_options(args.max_token_options)
    except ValueError as exc:
        parser.error(f"invalid --concurrency-levels: {exc}")
    if not args.concurrency_levels or any(value < 1 for value in args.concurrency_levels):
        parser.error("--concurrency-levels must contain positive integers")
    for name in ("min_requests", "request_multiplier", "warmup_requests"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not args.prompt_token_options or any(
        value < 0 for value in args.prompt_token_options
    ):
        parser.error("--prompt-token-options must contain non-negative integers")
    if not args.max_token_options or any(value < 1 for value in args.max_token_options):
        parser.error("--max-token-options must contain positive integers")
    if not 0 < args.min_success_rate <= 1:
        parser.error("--min-success-rate must be in (0, 1]")
    return args


def parse_int_options(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def fmt_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def make_prompt(request_id: int, target_tokens: int, seed: int) -> str:
    """Build a mostly unique prompt; common-word count approximates token count."""
    rng = random.Random(seed + request_id * 104729)
    words = [
        "algebra", "matrix", "vector", "integer", "proof", "theorem",
        "function", "sequence", "geometry", "probability", "derivative",
        "integral", "graph", "logic", "equation", "fraction",
    ]
    filler = " ".join(rng.choice(words) for _ in range(target_tokens))
    return (
        f"Benchmark request {request_id}. Read the nonce text below, then produce "
        "a long numbered list of concise mathematical facts. Continue until the "
        "output limit; do not summarize and do not call tools.\n\n"
        f"Nonce text: {filler}"
    )


def error_text(exc: Exception) -> tuple[str, int | None]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            detail = ""
        return f"HTTP {exc.code}: {detail}", exc.code
    return f"{type(exc).__name__}: {exc}", None


class Benchmark:
    def __init__(self, args: argparse.Namespace, api_key: str):
        self.args = args
        self.api_key = api_key
        self.endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def request_shape(self, request_id: int) -> tuple[int, int]:
        """Cycle through the input/output cross-product with deterministic order."""
        args = self.args
        index = request_id + args.seed
        prompt_tokens = args.prompt_token_options[index % len(args.prompt_token_options)]
        output_group = index // len(args.prompt_token_options)
        max_tokens = args.max_token_options[output_group % len(args.max_token_options)]
        return prompt_tokens, max_tokens

    def payload(self, request_id: int) -> dict[str, Any]:
        args = self.args
        prompt_tokens, max_tokens = self.request_shape(request_id)
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a benchmark assistant. Follow the request directly.",
                },
                {
                    "role": "user",
                    "content": make_prompt(request_id, prompt_tokens, args.seed),
                },
            ],
            "temperature": args.temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if args.provider == "vllm":
            payload["ignore_eos"] = True
            if args.thinking != "auto":
                payload["chat_template_kwargs"] = {
                    "enable_thinking": args.thinking == "enabled"
                }
        elif args.thinking != "auto":
            payload["thinking"] = {"type": args.thinking}
        return payload

    def request_once(self, request_id: int) -> dict[str, Any]:
        requested_prompt_tokens, requested_max_tokens = self.request_shape(request_id)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self.payload(request_id)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        chunks_with_text = 0
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(
                        usage.get("completion_tokens") or completion_tokens
                    )
                    total_tokens = int(usage.get("total_tokens") or total_tokens)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or delta.get("reasoning_content")
                        if text:
                            chunks_with_text += 1
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
            finished = time.perf_counter()
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
            if chunks_with_text == 0:
                return {
                    "ok": False,
                    "status": None,
                    "latency_s": finished - started,
                    "ttft_s": None,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "requested_prompt_tokens": requested_prompt_tokens,
                    "requested_max_tokens": requested_max_tokens,
                    "error": "empty streamed response",
                }
            return {
                "ok": True,
                "status": 200,
                "latency_s": finished - started,
                "ttft_s": first_token_at - started if first_token_at else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "requested_prompt_tokens": requested_prompt_tokens,
                "requested_max_tokens": requested_max_tokens,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - errors are benchmark output
            finished = time.perf_counter()
            error, status = error_text(exc)
            return {
                "ok": False,
                "status": status,
                "latency_s": finished - started,
                "ttft_s": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "requested_prompt_tokens": requested_prompt_tokens,
                "requested_max_tokens": requested_max_tokens,
                "error": error,
            }

    def run_round(self, concurrency: int, total_requests: int, offset: int) -> dict[str, Any]:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(self.request_once, range(offset, offset + total_requests))
            )
        wall_s = time.perf_counter() - started
        ok_rows = [row for row in rows if row["ok"]]
        latencies = [float(row["latency_s"]) for row in ok_rows]
        ttfts = [float(row["ttft_s"]) for row in ok_rows if row["ttft_s"] is not None]
        prompt_tokens = sum(int(row["prompt_tokens"]) for row in ok_rows)
        completion_tokens = sum(int(row["completion_tokens"]) for row in ok_rows)
        total_tokens = sum(int(row["total_tokens"]) for row in ok_rows)
        requested_prompt_tokens = [
            int(row["requested_prompt_tokens"]) for row in rows
        ]
        requested_max_tokens = [int(row["requested_max_tokens"]) for row in rows]
        errors = Counter(str(row["error"])[:300] for row in rows if not row["ok"])
        statuses = Counter(str(row["status"] or "transport") for row in rows if not row["ok"])
        success_rate = len(ok_rows) / len(rows) if rows else 0.0
        usage_coverage = (
            sum(1 for row in ok_rows if int(row["total_tokens"]) > 0) / len(ok_rows)
            if ok_rows
            else 0.0
        )
        return {
            "name": self.args.name,
            "model": self.args.model,
            "concurrency": concurrency,
            "requests": len(rows),
            "ok": len(ok_rows),
            "failed": len(rows) - len(ok_rows),
            "success_rate": success_rate,
            "usage_coverage": usage_coverage,
            "wall_s": wall_s,
            "requests_per_s": len(ok_rows) / wall_s if wall_s else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "requested_prompt_tokens_mean": statistics.fmean(requested_prompt_tokens),
            "requested_max_tokens_mean": statistics.fmean(requested_max_tokens),
            "prompt_tokens_per_s": prompt_tokens / wall_s if wall_s else 0.0,
            "completion_tokens_per_s": completion_tokens / wall_s if wall_s else 0.0,
            "total_tokens_per_s": total_tokens / wall_s if wall_s else 0.0,
            "ttft_p50_s": percentile(ttfts, 0.50),
            "ttft_p95_s": percentile(ttfts, 0.95),
            "latency_mean_s": statistics.fmean(latencies) if latencies else None,
            "latency_p50_s": percentile(latencies, 0.50),
            "latency_p95_s": percentile(latencies, 0.95),
            "stable": success_rate >= self.args.min_success_rate,
            "status_counts": dict(statuses),
            "errors": dict(errors),
        }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "results.jsonl"
    csv_path = args.output_dir / "results.csv"
    report_path = args.output_dir / "report.md"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    fields = [
        "name", "model", "concurrency", "requests", "ok", "failed",
        "success_rate", "usage_coverage", "requested_prompt_tokens_mean",
        "requested_max_tokens_mean", "wall_s", "requests_per_s",
        "prompt_tokens_per_s", "completion_tokens_per_s", "total_tokens_per_s",
        "ttft_p50_s", "ttft_p95_s", "latency_mean_s", "latency_p50_s",
        "latency_p95_s", "stable",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    stable = [row for row in rows if row["stable"]]
    best = max(stable, key=lambda row: row["completion_tokens_per_s"], default=None)
    highest = max(stable, key=lambda row: row["concurrency"], default=None)
    lines = [
        f"# {args.name} throughput benchmark",
        "",
        f"- Endpoint: `{args.base_url}`",
        f"- Model: `{args.model}`",
        f"- Input size options: `{args.prompt_token_options}` approximate tokens",
        f"- Output limit options: `{args.max_token_options}` tokens",
        f"- Thinking: `{args.thinking}`",
        f"- Stable threshold: `{args.min_success_rate:.1%}` success",
        (
            f"- Best stable output throughput: `{best['completion_tokens_per_s']:.1f}` tok/s "
            f"at concurrency `{best['concurrency']}`"
            if best else "- Best stable output throughput: none"
        ),
        (
            f"- Highest stable tested concurrency: `{highest['concurrency']}`"
            if highest else "- Highest stable tested concurrency: none"
        ),
        "",
        "| concurrency | avg requested input | avg output limit | success | req/s | input tok/s | output tok/s | total tok/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 | stable |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | {row['requested_prompt_tokens_mean']:.0f} "
            f"| {row['requested_max_tokens_mean']:.0f} | {row['ok']}/{row['requests']} "
            f"| {row['requests_per_s']:.2f} | {row['prompt_tokens_per_s']:.1f} "
            f"| {row['completion_tokens_per_s']:.1f} | {row['total_tokens_per_s']:.1f} "
            f"| {fmt_seconds(row['ttft_p50_s'])} | {fmt_seconds(row['ttft_p95_s'])} "
            f"| {fmt_seconds(row['latency_p50_s'])} | {fmt_seconds(row['latency_p95_s'])} "
            f"| {'yes' if row['stable'] else 'no'} |"
        )
    error_rows = [row for row in rows if row["errors"]]
    if error_rows:
        lines.extend(["", "## Errors", ""])
        for row in error_rows:
            lines.append(
                f"- concurrency `{row['concurrency']}`: "
                f"`{json.dumps(row['status_counts'], ensure_ascii=False)}`; "
                f"{json.dumps(row['errors'], ensure_ascii=False)}"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    print(
        json.dumps(
            {
                "name": args.name,
                "base_url": args.base_url,
                "model": args.model,
                "provider": args.provider,
                "thinking": args.thinking,
                "concurrency_levels": args.concurrency_levels,
                "prompt_token_options": args.prompt_token_options,
                "max_token_options": args.max_token_options,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    benchmark = Benchmark(args, api_key)
    print(f"[warmup] requests={args.warmup_requests}", flush=True)
    warmup = benchmark.run_round(
        min(args.warmup_requests, max(args.concurrency_levels)),
        args.warmup_requests,
        -args.warmup_requests,
    )
    if warmup["ok"] != args.warmup_requests:
        raise SystemExit(f"warmup failed: {warmup['errors']}")
    if warmup["usage_coverage"] < 1:
        print(
            "[warn] endpoint did not return usage for every warmup request; token/s may be understated",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    offset = 0
    for concurrency in args.concurrency_levels:
        requests = max(args.min_requests, concurrency * args.request_multiplier)
        print(f"[round] concurrency={concurrency} requests={requests}", flush=True)
        row = benchmark.run_round(concurrency, requests, offset)
        offset += requests
        rows.append(row)
        write_outputs(args, rows)
        print(
            f"  ok={row['ok']}/{row['requests']} wall={row['wall_s']:.1f}s "
            f"req/s={row['requests_per_s']:.2f} out_tok/s={row['completion_tokens_per_s']:.1f} "
            f"ttft_p95={fmt_seconds(row['ttft_p95_s'])} "
            f"lat_p95={fmt_seconds(row['latency_p95_s'])}",
            flush=True,
        )
        if not row["stable"] and args.stop_on_unstable:
            print("[stop] success rate fell below the stability threshold", flush=True)
            break
        time.sleep(args.round_pause_seconds)

    stable = [row for row in rows if row["stable"]]
    best = max(stable, key=lambda row: row["completion_tokens_per_s"], default=None)
    if best:
        print(
            f"[result] best stable concurrency={best['concurrency']} "
            f"output_tokens/s={best['completion_tokens_per_s']:.1f}",
            flush=True,
        )
    print(f"[result] report={args.output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
