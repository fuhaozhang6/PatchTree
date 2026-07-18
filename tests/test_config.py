from pathlib import Path

import pytest

from skillopt.config import apply_overrides, load_config


def test_empty_structured_section_preserves_base_mapping(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "gradient:\n"
        "  analyst_workers: 16\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: base.yaml\n"
        "gradient:\n",
        encoding="utf-8",
    )

    cfg = load_config(str(child))

    assert cfg["gradient"] == {"analyst_workers": 16}


def test_dotted_override_repairs_legacy_null_section() -> None:
    cfg = {"gradient": None}

    apply_overrides(cfg, ["gradient.analyst_workers=16"])

    assert cfg == {"gradient": {"analyst_workers": 16}}


def test_dotted_override_rejects_scalar_section() -> None:
    cfg = {"gradient": "invalid"}

    with pytest.raises(TypeError, match="must be a mapping"):
        apply_overrides(cfg, ["gradient.analyst_workers=16"])
