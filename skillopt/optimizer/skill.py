"""Apply executable PatchTree edits to a Skill document."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillopt.types import Edit as EditType, Patch as PatchType

def _edit_fields(edit: EditType | dict) -> tuple[str, str, str]:
    op = edit.op if hasattr(edit, "op") else edit.get("op", "")
    content = (edit.content if hasattr(edit, "content") else edit.get("content", "")).strip()
    target = edit.target if hasattr(edit, "target") else edit.get("target", "")
    return op, content, target


def _apply_edit_with_report(skill: str, edit: EditType | dict) -> tuple[str, dict]:
    op, content, target = _edit_fields(edit)
    report = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    if op == "append":
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", report

    if op == "insert_after":
        if not target or target not in skill:
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", report
        idx = skill.index(target) + len(target)
        newline = skill.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], report

    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), report

    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return skill, report


def apply_edit(skill: str, edit: EditType | dict) -> str:
    """Apply a single edit operation to the skill document.

    Parameters
    ----------
    skill : str
        Current skill document content.
    edit : Edit | dict
        An :class:`~skillopt.types.Edit` instance or a plain dict with
        keys ``op``, ``content``, ``target``.

    """
    updated_skill, _ = _apply_edit_with_report(skill, edit)
    return updated_skill


def apply_patch_with_report(
    skill: str,
    patch: PatchType | dict,
) -> tuple[str, list[dict]]:
    """Apply a patch and return a per-edit report for observability."""
    edits = patch.edits if hasattr(patch, "edits") else patch.get("edits", [])
    reports: list[dict] = []
    for idx, edit in enumerate(edits, 1):
        try:
            skill, report = _apply_edit_with_report(skill, edit)
            report["index"] = idx
        except Exception as exc:  # noqa: BLE001
            report = {
                "index": idx,
                "op": "",
                "target": "",
                "content_preview": "",
                "status": "error",
                "error": str(exc),
            }
        reports.append(report)
    return skill, reports


def apply_patch(skill: str, patch: PatchType | dict) -> str:
    """Apply a patch (list of edits) to the skill document sequentially.

    Parameters
    ----------
    skill : str
        Current skill document content.
    patch : Patch | dict
        A :class:`~skillopt.types.Patch` instance or a plain dict with
        key ``edits`` containing a list of edit operations.
    """
    updated_skill, _ = apply_patch_with_report(skill, patch)
    return updated_skill
