"""Shared helpers for blind SearchQA repair-taxonomy discovery."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "searchqa_blind_mechanism_v2"
INFRA_PATTERNS = (
    "timeout",
    "timed out",
    "request failed",
    "connection error",
    "context length",
    "maximum context",
    "cuda out of memory",
    "engine dead",
    "service unavailable",
    "rate limit",
    "http 429",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stable_hash(value: Any, length: int = 16) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def sample_key(split: str, sample_id: str) -> str:
    return f"{split}::{sample_id}"


def split_prediction_id(prediction_id: str) -> tuple[str, int]:
    match = re.match(r"^(.*)::tg_repeat(\d+)$", str(prediction_id))
    return (match.group(1), int(match.group(2))) if match else (str(prediction_id), 0)


def infrastructure_failure(result: dict[str, Any]) -> str:
    text = " ".join(
        str(result.get(key) or "")
        for key in ("fail_reason", "error", "response", "evaluator_feedback")
    ).lower()
    return next((pattern for pattern in INFRA_PATTERNS if pattern in text), "")


def outcome_from_rollouts(rollouts: list[dict[str, Any]]) -> tuple[str, float]:
    if not rollouts:
        return "invalid", 0.0
    hard = 0
    for rollout in rollouts:
        try:
            hard += int(float((rollout.get("result") or {}).get("hard", 0) or 0) > 0)
        except (TypeError, ValueError):
            hard += int(bool((rollout.get("result") or {}).get("hard")))
    q_i = hard / len(rollouts)
    return ("stable_success" if q_i >= 1 else "failure" if q_i <= 0 else "unstable"), q_i


def card_feature_text(card: dict[str, Any]) -> str:
    """Build clustering text without ever reading the old seeded labels."""
    primary = [
        str(card.get("missing_operation") or "").strip(),
        str(card.get("repair_signature") or "").strip(),
    ]
    # Weight the deliberately compact operational fields more heavily than
    # verbose observed-failure prose, which otherwise fragments lexical TF-IDF.
    parts = primary * 3
    keywords = card.get("mechanism_keywords") or []
    if isinstance(keywords, list):
        keyword_text = " ".join(str(value).strip() for value in keywords)
        parts.extend([keyword_text] * 3)
    parts.extend([
        str(card.get("condition") or "").strip(),
        str(card.get("boundary") or "").strip(),
        str(card.get("observed_failure") or "").strip(),
    ])
    patch = card.get("candidate_patch") or {}
    if isinstance(patch, dict):
        parts.append(str(patch.get("content") or "").strip())
    return "\n".join(part for part in parts if part)


def hashed_tfidf(documents: list[str], dimensions: int = 8192) -> np.ndarray:
    """Zero-extra-dependency hashed word 1/2-gram TF-IDF."""
    if dimensions < 256:
        raise ValueError("dimensions must be >= 256")
    sparse_rows: list[Counter[int]] = []
    document_frequency: Counter[int] = Counter()
    for document in documents:
        words = re.findall(r"[a-z0-9]+", document.lower())
        grams = words + [f"{left}_{right}" for left, right in zip(words, words[1:])]
        counts: Counter[int] = Counter()
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            counts[int.from_bytes(digest, "little") % dimensions] += 1
        sparse_rows.append(counts)
        document_frequency.update(counts.keys())
    matrix = np.zeros((len(documents), dimensions), dtype=np.float32)
    n_documents = max(len(documents), 1)
    for row_index, counts in enumerate(sparse_rows):
        for feature, count in counts.items():
            tf = 1.0 + math.log(float(count))
            idf = math.log((1.0 + n_documents) / (1.0 + document_frequency[feature])) + 1.0
            matrix[row_index, feature] = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_dbscan(
    matrix: np.ndarray,
    similarity_threshold: float,
    min_samples: int,
) -> list[int]:
    """Deterministic DBSCAN on normalized vectors; -1 denotes noise."""
    return cosine_dbscan_from_similarities(
        matrix @ matrix.T,
        similarity_threshold,
        min_samples,
    )


def cosine_dbscan_from_similarities(
    similarities: np.ndarray,
    similarity_threshold: float,
    min_samples: int,
) -> list[int]:
    """DBSCAN using a precomputed cosine-similarity matrix."""
    count = int(similarities.shape[0])
    if count == 0:
        return []
    neighbors = [
        np.flatnonzero(similarities[index] >= similarity_threshold).tolist()
        for index in range(count)
    ]
    core = [len(values) >= min_samples for values in neighbors]
    labels = [-99] * count
    cluster_id = 0
    for start in range(count):
        if labels[start] != -99:
            continue
        if not core[start]:
            labels[start] = -1
            continue
        labels[start] = cluster_id
        queue = list(neighbors[start])
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            point = queue[cursor]
            cursor += 1
            if labels[point] == -1:
                labels[point] = cluster_id
            if labels[point] != -99:
                continue
            labels[point] = cluster_id
            if core[point]:
                for neighbor in neighbors[point]:
                    if neighbor not in queued:
                        queued.add(neighbor)
                        queue.append(neighbor)
        cluster_id += 1
    return labels


def centroid(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    value = matrix[indices].mean(axis=0)
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def deterministic_partition(
    keys: list[str],
    fit_fraction: float,
    seed: int,
    min_holdout: int = 2,
) -> tuple[list[str], list[str]]:
    ranked = sorted(keys, key=lambda key: stable_hash({"seed": seed, "key": key}, 32))
    if len(ranked) <= min_holdout:
        return ranked, []
    fit_count = int(round(len(ranked) * fit_fraction))
    fit_count = max(1, min(fit_count, len(ranked) - min_holdout))
    return sorted(ranked[:fit_count]), sorted(ranked[fit_count:])


def discover_card_files(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            direct = path / "usable_mechanism_cards.jsonl"
            paths.add(direct) if direct.is_file() else paths.update(
                path.rglob("usable_mechanism_cards.jsonl")
            )
        else:
            raise FileNotFoundError(path)
    return sorted(paths)


def load_unique_cards(inputs: list[Path]) -> list[dict[str, Any]]:
    files = discover_card_files(inputs)
    if not files:
        raise FileNotFoundError("no usable_mechanism_cards.jsonl found")
    by_key: dict[str, dict[str, Any]] = {}
    for path in files:
        for card in read_jsonl(path):
            key = str(card.get("sample_key") or "")
            if not key:
                continue
            previous = by_key.get(key)
            if previous is not None and stable_hash(previous) != stable_hash(card):
                raise ValueError(f"conflicting duplicate card: {key}")
            by_key[key] = card
    return [by_key[key] for key in sorted(by_key)]


def member_counts(keys: list[str], cards_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    splits: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    for key in keys:
        card = cards_by_key[key]
        splits[str(card.get("split") or "unknown")] += 1
        outcomes[str(card.get("outcome_status") or "unknown")] += 1
    return {
        "support_count": len(keys),
        "split_counts": dict(sorted(splits.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
    }
