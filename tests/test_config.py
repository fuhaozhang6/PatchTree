from pathlib import Path

import pytest

from skillopt.config import apply_overrides, flatten_config, load_config


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


def test_default_patchtree_budget_is_non_binding_and_fuse_is_enabled() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = flatten_config(load_config(str(repo_root / "configs/searchqa/default.yaml")))

    assert cfg["edit_budget"] == 999
    assert cfg["min_edit_budget"] == 999
    assert cfg["lr_scheduler"] == "constant"
    assert cfg["type_guided_fallback_reconcile"] == "llm_fuse"
    assert cfg["type_guided_fallback_sel_env_num"] == 80
    assert cfg["type_guided_tail_bank"] is True
    assert cfg["type_guided_tree_builder"] == "recursive"
    assert cfg["type_guided_max_tree_depth"] == 4
    assert cfg["type_guided_merge_target_children"] == 3
    assert cfg["type_guided_merge_max_children"] == 4
    assert cfg["type_guided_top_mode"] == "auto"
    assert cfg["type_guided_fallback_enabled"] is True
    assert cfg["type_guided_fallback_max_hops"] == -1
    assert cfg["type_guided_fallback_allow_leaf"] is True
    assert cfg["type_guided_fallback_min_leaf_coverage"] == 1
    assert cfg["type_guided_validation_budget"] == 16
