from argparse import Namespace

import pytest

from scripts.tools.validate_searchqa_blind_transfer import (
    VALIDATION_SCHEMA_VERSION,
    paired_summary,
    reusable_cached_result,
    stratified_sample_keys,
    validation_fingerprint,
    validation_protocol,
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


def test_holdout_cap_is_deterministically_stratified_by_split_and_outcome():
    cards = {}

    def add(split, outcome, suffix):
        key = f"{split}::{suffix}"
        cards[key] = {
            "sample_key": key,
            "split": split,
            "outcome_status": outcome,
        }
        return key

    keys = [
        *(add("train", "failure", f"f{i}") for i in range(4)),
        *(add("val", "failure", f"f{i}") for i in range(3)),
        *(add("train", "unstable", f"u{i}") for i in range(2)),
        add("test", "unstable", "u0"),
    ]

    selected = stratified_sample_keys(list(reversed(keys)), cards, maximum=6)
    selected_again = stratified_sample_keys(keys, cards, maximum=6)

    assert selected == selected_again
    assert len(selected) == 6
    assert {
        (cards[key]["split"], cards[key]["outcome_status"])
        for key in selected
    } == {
        ("train", "failure"),
        ("val", "failure"),
        ("train", "unstable"),
        ("test", "unstable"),
    }
    # Midpoint sampling must not reproduce the global lexicographic prefix.
    assert selected != sorted(keys)[:6]


def test_validation_protocol_records_sample_pairing_without_seed_claim():
    protocol = validation_protocol(repeats=5)

    assert protocol["paired_by_sample"] is True
    assert protocol["same_rollout_batch_seed_argument"] is True
    assert protocol["generation_randomness_seed_paired"] is False
    assert "pairing is by sample only" in protocol["seed_note"]
