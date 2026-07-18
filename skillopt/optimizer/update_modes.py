"""Small helpers for the single PatchTree edit payload."""
from __future__ import annotations

from typing import Any

def payload_key(mode: str | None) -> str:
    return "edits"


def payload_label(mode: str | None, *, singular: bool = False, title: bool = False) -> str:
    word = "edit" if singular else "edits"
    return word.title() if title else word


def get_payload_items(container: dict | None, mode: str | None) -> list[dict]:
    if not isinstance(container, dict):
        return []
    items = container.get(payload_key(mode), [])
    return items if isinstance(items, list) else []


def set_payload_items(container: dict, items: list[dict], mode: str | None) -> dict:
    container[payload_key(mode)] = items
    return container


def truncate_payload(container: dict, max_items: int, mode: str | None) -> dict:
    if max_items < 0:
        return container
    items = get_payload_items(container, mode)
    if len(items) > max_items:
        set_payload_items(container, items[:max_items], mode)
    return container


def describe_item(item: dict, mode: str | None, *, max_chars: int | None = None) -> str:
    if not isinstance(item, dict):
        return ""
    op = item.get("op", "?")
    target = item.get("target", "")
    content = item.get("content", "")
    parts = [f"op={op}"]
    if target:
        parts.append(f"target={target!r}")
    if content:
        parts.append(f"content={content!r}")
    if item.get("support_count") is not None:
        parts.append(f"support={item.get('support_count')}")
    text = "  ".join(parts)
    # Truncation disabled: the optimizer is given the full item description.
    return text


def short_item_summary(item: dict, mode: str | None, *, max_chars: int | None = None) -> dict[str, Any]:
    return {
        "op": item.get("op", "?"),
        "content": str(item.get("content", "")),
        "target": item.get("target", ""),
    }
