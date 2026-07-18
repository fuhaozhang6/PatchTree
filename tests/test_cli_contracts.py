"""Cheap regression checks for command-line and shell launcher contracts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.cli import eval_only


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eval_only_accepts_qwen_target_overrides() -> None:
    argv = [
        "eval_only.py",
        "--config", "configs/searchqa/default.yaml",
        "--skill", "skill.md",
        "--backend", "qwen_chat",
        "--target_qwen_chat_base_url", "http://127.0.0.1:8000/v1",
        "--target_qwen_chat_enable_thinking", "false",
    ]

    with patch.object(sys, "argv", argv):
        args = eval_only.parse_args()

    assert args.backend == "qwen_chat"
    assert args.target_qwen_chat_base_url == "http://127.0.0.1:8000/v1"
    assert args.target_qwen_chat_enable_thinking is False


def test_shell_launchers_have_valid_bash_syntax() -> None:
    launchers = sorted((PROJECT_ROOT / "scripts").rglob("*.sh"))
    for launcher in launchers:
        if launcher.name.startswith("._"):
            continue
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_shell_launchers_do_not_use_removed_vllm_flag() -> None:
    for launcher in (PROJECT_ROOT / "scripts").rglob("*.sh"):
        if launcher.name.startswith("._"):
            continue
        assert "--disable-log-requests" not in launcher.read_text(encoding="utf-8")
