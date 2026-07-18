"""LLM-driven ranking for executable PatchTree edits."""
from __future__ import annotations

from skillopt.model import chat_optimizer
from skillopt.optimizer.update_modes import (
    describe_item,
    get_payload_items,
    payload_key,
    payload_label,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


# ── Public API ────────────────────────────────────────────────────────────────

def rank_and_select(
    skill_content: str,
    patch: dict,
    max_edits: int,
    update_mode: str = "patch",
) -> dict:
    """Use a optimizer LLM to rank edits by importance, then keep top-L.

    If the edit pool is within budget, returns the patch unchanged.
    Otherwise, calls the optimizer to rank and select the most impactful edits.

    Parameters
    ----------
    skill_content : str
        Current skill document.
    patch : dict
        Merged :class:`~skillopt.types.Patch` dict with ``edits`` list.
    max_edits : int
        Maximum number of edits to keep (the "edit budget").

    Returns
    -------
    dict
        :class:`~skillopt.types.Patch` dict with selected edits and
        optional ``ranking_details``.
    """
    edits = get_payload_items(patch, update_mode)
    if len(edits) <= max_edits:
        return patch

    # Build the edit pool description for the optimizer
    edits_desc = []
    for i, edit in enumerate(edits):
        edits_desc.append(f"[{i}] {describe_item(edit, update_mode)}")

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## {payload_label(update_mode, title=True)} Pool ({len(edits)} {payload_label(update_mode)}, budget={max_edits})\n"
        + "\n".join(edits_desc)
        + f"\n\nSelect the {max_edits} most important {payload_label(update_mode)}. "
        f"Return their 0-based indices in priority order."
    )
    try:
        response, _ = chat_optimizer(
            system=load_prompt("ranking"), user=user,
            max_completion_tokens=16384, retries=3, stage="ranking",
        )
        result = extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected = []
            seen: set[int] = set()
            for idx in indices:
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(edits)
                    and idx not in seen
                ):
                    selected.append(edits[idx])
                    seen.add(idx)
                if len(selected) >= max_edits:
                    break
            if selected:
                return {
                    "reasoning": patch.get("reasoning", "")
                    + f" [optimizer-ranked: selected {len(selected)}/{len(edits)} {payload_label(update_mode)}]",
                    payload_key(update_mode): selected,
                    "ranking_details": result,
                }
    except Exception:  # noqa: BLE001
        pass

    # Fallback: simple truncation
    return {
        "reasoning": patch.get("reasoning", "")
        + f" [fallback truncated {len(edits)}->{max_edits} {payload_label(update_mode)}]",
        payload_key(update_mode): edits[:max_edits],
    }
