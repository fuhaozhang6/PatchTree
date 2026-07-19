#!/usr/bin/env python3
"""Isolated SearchQA prompt/concurrency latency probe for local Qwen endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.searchqa.evaluator import evaluate


MAX_CONTEXT_CHARS = 6000


def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(context) <= max_chars:
        return context
    docs = context.split("[DOC]")
    result = ""
    for doc in docs:
        candidate = result + "[DOC]" + doc if result else doc
        if len(candidate) > max_chars:
            break
        result = candidate
    return result or (context[:max_chars] + "\n...[truncated]")


def build_user(item: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"## Context\n{truncate_context(str(item.get('context') or ''))}",
            f"## Question\n{item.get('question') or ''}",
        ]
    )


def load_skill(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"## Skill\n{text}\n\n"


def load_prompt(path: Path, skill_section: str) -> str:
    text = path.read_text(encoding="utf-8")
    return text.format(skill_section=skill_section)


def chat_once(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    data = json.loads(raw)
    elapsed = time.time() - started
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "ok": True,
        "dt_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "content_chars": len(content),
        "has_answer_tag": "<answer>" in content.lower() and "</answer>" in content.lower(),
        "content_preview": content[:500],
        "response": content,
    }


def check_endpoint(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    models_url = f"{url}/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint check failed: GET {models_url} -> HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"endpoint check failed: cannot connect to {models_url}: {exc}. "
            "Start vLLM on this machine, run this script on the vLLM machine, "
            "or pass --base-url to a reachable OpenAI-compatible endpoint."
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:1000]}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    lat = [float(r["dt_s"]) for r in ok_rows]
    comp = [int(r.get("completion_tokens") or 0) for r in ok_rows]
    prompt = [int(r.get("prompt_tokens") or 0) for r in ok_rows]
    hard = [float(r.get("hard") or 0.0) for r in ok_rows if r.get("evaluated")]
    soft = [float(r.get("soft") or 0.0) for r in ok_rows if r.get("evaluated")]
    wall = max((float(r.get("finished_at", 0)) for r in rows), default=0) - min(
        (float(r.get("started_at", 0)) for r in rows),
        default=0,
    )
    total_completion = sum(comp)
    return {
        "n": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "wall_s": wall,
        "requests_per_s": len(ok_rows) / wall if wall > 0 else None,
        "completion_tokens_per_s": total_completion / wall if wall > 0 else None,
        "avg_latency_s": statistics.mean(lat) if lat else None,
        "p50_latency_s": percentile(lat, 50),
        "p90_latency_s": percentile(lat, 90),
        "p95_latency_s": percentile(lat, 95),
        "avg_prompt_tokens": statistics.mean(prompt) if prompt else None,
        "avg_completion_tokens": statistics.mean(comp) if comp else None,
        "p50_completion_tokens": percentile(comp, 50),
        "p90_completion_tokens": percentile(comp, 90),
        "p95_completion_tokens": percentile(comp, 95),
        "max_completion_tokens": max(comp) if comp else None,
        "answer_tag_rate": (
            sum(1 for r in ok_rows if r.get("has_answer_tag")) / len(ok_rows)
            if ok_rows
            else None
        ),
        "evaluated": len(hard),
        "hard": sum(hard) / len(hard) if hard else None,
        "soft": sum(soft) / len(soft) if soft else None,
        "finish_reasons": {
            str(reason): sum(1 for r in ok_rows if str(r.get("finish_reason")) == str(reason))
            for reason in sorted({r.get("finish_reason") for r in ok_rows}, key=str)
        },
    }


def run_setting(
    *,
    items: list[dict[str, Any]],
    prompt_name: str,
    system_prompt: str,
    workers: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    out_path = out_dir / f"{prompt_name}.workers{workers}.jsonl"
    rows: list[dict[str, Any]] = []

    def one(index_item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, item = index_item
        started_at = time.time()
        base = {
            "idx": idx,
            "id": str(item.get("id")),
            "question": item.get("question"),
            "prompt_variant": prompt_name,
            "workers": workers,
            "started_at": started_at,
        }
        try:
            result = chat_once(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                system=system_prompt,
                user=build_user(item),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                enable_thinking=args.enable_thinking,
            )
            if not args.no_eval:
                eval_result = evaluate(result.get("response") or "", item.get("answers", []))
                result.update(
                    {
                        "evaluated": True,
                        "em": eval_result["em"],
                        "f1": eval_result["f1"],
                        "sub_em": eval_result["sub_em"],
                        "hard": int(eval_result["em"]),
                        "soft": eval_result["f1"],
                        "predicted_answer": eval_result["predicted_answer"],
                        "gold_answers": item.get("answers", []),
                    }
                )
            if args.no_full_response:
                result.pop("response", None)
            base.update(result)
        except Exception as exc:  # noqa: BLE001
            base.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        base["finished_at"] = time.time()
        return base

    started = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, pair) for pair in enumerate(items)]
            for fut in futures.as_completed(futs):
                row = fut.result()
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    summary = summarize(rows)
    summary.update(
        {
            "prompt_variant": prompt_name,
            "workers": workers,
            "elapsed_s": time.time() - started,
            "jsonl": str(out_path),
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_CHAT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_CHAT_API_KEY", "dummy"))
    parser.add_argument("--model", default=os.environ.get("QWEN_CHAT_MODEL", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--split", type=Path, default=Path("data/searchqa_split/val/items.json"))
    parser.add_argument("--skill", type=Path, default=Path("skillopt/envs/searchqa/skills/initial.md"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("experiments/searchqa_prompt_latency/prompts"))
    parser.add_argument("--prompt", action="append", default=[], help="Prompt variant stem or path. Defaults to all *.md.")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, nargs="+", default=[48, 96, 128])
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="Skip SearchQA EM/F1 evaluation.")
    parser.add_argument("--no-full-response", action="store_true", help="Do not store full model responses in JSONL.")
    parser.add_argument("--dataset-label", default="searchqa", help="Label written into summary for load-test bookkeeping.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/searchqa_prompt_latency"))
    args = parser.parse_args()

    items = json.loads(args.split.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.sample_size > 0:
        items = items[: args.sample_size]

    if args.prompt:
        prompt_paths = []
        for raw in args.prompt:
            path = Path(raw)
            if not path.exists():
                path = args.prompt_dir / f"{raw}.md"
            prompt_paths.append(path)
    else:
        prompt_paths = sorted(args.prompt_dir.glob("*.md"))
    if not prompt_paths:
        raise SystemExit(f"No prompt variants found in {args.prompt_dir}")

    skill_section = load_skill(args.skill)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    if not args.skip_preflight:
        try:
            models = check_endpoint(args.base_url, args.api_key, args.timeout)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        model_names = [
            str(item.get("id"))
            for item in (models.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        if model_names and args.model not in model_names:
            print(
                f"[warn] requested model {args.model!r} not listed by endpoint. "
                f"available={model_names[:10]}",
                flush=True,
            )
    print(
        f"[config] samples={len(items)} prompts={len(prompt_paths)} "
        f"workers={args.workers} max_tokens={args.max_tokens} base_url={args.base_url}",
        flush=True,
    )

    for prompt_path in prompt_paths:
        system_prompt = load_prompt(prompt_path, skill_section)
        prompt_name = prompt_path.stem
        for workers in args.workers:
            print(f"[run] prompt={prompt_name} workers={workers}", flush=True)
            summary = run_setting(
                items=items,
                prompt_name=prompt_name,
                system_prompt=system_prompt,
                workers=workers,
                args=args,
                out_dir=args.out_dir,
            )
            summaries.append(summary)
            summary["dataset_label"] = args.dataset_label
            summary["split"] = str(args.split)
            summary["sample_size"] = len(items)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
