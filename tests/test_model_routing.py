"""Regression tests for model backend routing contracts."""
from __future__ import annotations

from typing import Any

import pytest

import skillopt.model as model
from skillopt.model import backend_config, minimax_backend


def test_rejected_optimizer_backend_does_not_corrupt_global_state() -> None:
    previous = backend_config.get_optimizer_backend()

    with pytest.raises(ValueError, match="only as a target backend"):
        backend_config.set_optimizer_backend("minimax_chat")

    assert backend_config.get_optimizer_backend() == previous


def test_rejected_target_backend_does_not_corrupt_global_state() -> None:
    previous = backend_config.get_target_backend()

    with pytest.raises(ValueError, match="Unsupported target backend"):
        backend_config.set_target_backend("not-a-backend")

    assert backend_config.get_target_backend() == previous


@pytest.mark.parametrize("messages_api", [False, True])
def test_minimax_target_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
    messages_api: bool,
) -> None:
    previous = backend_config.get_target_backend()
    captured: dict[str, Any] = {}

    def fake_call(*args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
        captured.update(kwargs)
        return "ok", {"total_tokens": 0}

    try:
        backend_config.set_target_backend("minimax_chat")
        if messages_api:
            monkeypatch.setattr(minimax_backend, "chat_target_messages", fake_call)
            result, _ = model.chat_target_messages(
                [{"role": "user", "content": "hello"}],
                timeout=17,
            )
        else:
            monkeypatch.setattr(minimax_backend, "chat_target", fake_call)
            result, _ = model.chat_target("system", "hello", timeout=17)
    finally:
        backend_config.set_target_backend(previous)

    assert result == "ok"
    assert captured["timeout"] == 17
