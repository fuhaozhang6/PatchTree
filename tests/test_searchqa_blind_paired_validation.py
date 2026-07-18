from argparse import Namespace

import pytest

from scripts.tools.validate_searchqa_blind_transfer import (
    VALIDATION_SCHEMA_VERSION,
    paired_summary,
    reusable_cached_result,
    validation_fingerprint,
)


def _args(**overrides):
    values = {
        "target_model": "Qwen/Qwen3.5-4B",
        "target_base_url": "http://127.0.0.1:8000/v1",
        "target_temperature": 0.2,
        "target_timeout_seconds": 300,
        "target_max_tokens": 4096,
        "repeats": 3,
        "batch_size": 100,
        "seed": 4242,
        "min_holdout_samples": 10,
        "min_boundary_samples": 4,
        "min_delta_in": 0.05,
        "max_boundary_drop": 0.02,
    }
    values.update(overrides)
    return Namespace(**values)


def _fingerprint(**overrides):
    values = {
        "type_item": {
            "type_id": "T001",
            "revision_type": "evidence_alignment",
            "shared_patch": {"op": "append", "content": "Check the evidence."},
        },
        "initial_skill": "Initial skill",
        "selected_items": {
            "val::one": {"id": "one", "question": "Question one"},
            "test::two": {"id": "two", "question": "Question two"},
        },
        "holdout": ["val::one"],
        "boundary": ["test::two"],
        "type_index": 1,
        "runtime_config_sha256": "config-hash",
        "args": _args(),
    }
    values.update(overrides)
    return validation_fingerprint(**values)


def test_paired_summary_records_per_sample_changes_and_counts():
    keys = ["val::up", "val::same", "val::down"]
    summary = paired_summary(
        keys,
        baseline_new_q={"val::up": 0.0, "val::same": 2 / 3, "val::down": 1.0},
        patched_q={"val::up": 1 / 3, "val::same": 2 / 3, "val::down": 1 / 3},
    )

    assert summary["improved"] == 1
    assert summary["unchanged"] == 1
    assert summary["regressed"] == 1
    assert summary["paired_mean_delta"] == pytest.approx(-1 / 9)
    assert summary["accuracy_delta"] == pytest.approx(summary["paired_mean_delta"])
    assert [row["paired_outcome"] for row in summary["pairs"]] == [
        "improved",
        "unchanged",
        "regressed",
    ]


def test_validation_fingerprint_changes_for_skill_patch_and_sampling_inputs():
    original = _fingerprint()
    assert _fingerprint(initial_skill="Changed initial skill") != original
    assert _fingerprint(
        type_item={
            "type_id": "T001",
            "revision_type": "evidence_alignment",
            "shared_patch": {"op": "append", "content": "A different patch."},
        }
    ) != original
    assert _fingerprint(args=_args(repeats=5)) != original
    assert _fingerprint(type_index=2) != original


def test_only_current_paired_schema_and_fingerprint_are_resumable():
    fingerprint = _fingerprint()
    assert reusable_cached_result(
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_fingerprint": fingerprint,
        },
        fingerprint,
    )
    assert not reusable_cached_result(
        {"validation_fingerprint": fingerprint},
        fingerprint,
    )
    assert not reusable_cached_result(
        {
            "schema_version": "searchqa_blind_transfer_v1",
            "validation_fingerprint": fingerprint,
        },
        fingerprint,
    )
    assert not reusable_cached_result(
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validation_fingerprint": "stale",
        },
        fingerprint,
    )
