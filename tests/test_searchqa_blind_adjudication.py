import json
from collections import Counter

from scripts.tools import adjudicate_searchqa_blind_clusters as adjudication


def _card(index: int, status: str, split: str) -> dict:
    return {
        "sample_key": f"{status}:{split}:{index:03d}",
        "outcome_status": status,
        "split": split,
        "missing_operation": f"operation {index}",
    }


def test_fit_sampling_is_stratified_deterministic_and_not_unstable_only():
    cards = {}
    fit_keys = []
    for status, split, count in (
        ("unstable", "test", 50),
        ("unstable", "train", 10),
        ("unstable", "val", 8),
        ("failure", "test", 80),
        ("failure", "train", 25),
        ("failure", "val", 14),
    ):
        for index in range(count):
            card = _card(index, status, split)
            cards[card["sample_key"]] = card
            fit_keys.append(card["sample_key"])
    cluster = {"fit_member_keys": list(reversed(fit_keys))}

    first = adjudication.select_fit_cards(cluster, cards, 36)
    second = adjudication.select_fit_cards(cluster, cards, 36)

    assert first == second
    assert len(first) == 36
    strata = Counter((card["outcome_status"], card["split"]) for card in first)
    assert set(strata) == {
        ("unstable", "test"),
        ("unstable", "train"),
        ("unstable", "val"),
        ("failure", "test"),
        ("failure", "train"),
        ("failure", "val"),
    }
    assert sum(
        count for (status, _), count in strata.items() if status == "failure"
    ) > 0
    assert sum(
        count for (status, _), count in strata.items() if status == "unstable"
    ) <= 24


def test_fit_sampling_fills_budget_when_other_evidence_is_scarce():
    all_cards = [
        *[_card(index, "unstable", "test") for index in range(40)],
        _card(0, "failure", "val"),
    ]
    cards = {card["sample_key"]: card for card in all_cards}
    cluster = {"fit_member_keys": list(cards)}

    selected = adjudication.select_fit_cards(cluster, cards, 36)

    assert len(selected) == 36
    assert any(card["outcome_status"] == "failure" for card in selected)


def test_adjudication_cache_changes_with_model_drafts_prompt_and_evidence(
    tmp_path, monkeypatch
):
    calls = []

    def fake_call(system, user, stage):
        calls.append((system, user, stage))
        if stage.endswith("reconcile"):
            return {
                "accepted": False,
                "reasoning": "test",
                "suspected_submechanisms": [],
                "confidence": "low",
            }
        return {"accepted": False}

    monkeypatch.setattr(adjudication, "call_json", fake_call)
    cluster = {
        "cluster_id": "C001",
        "support_count": 3,
        "origin": "contrast_core",
        "split_counts": {"train": 3},
        "outcome_counts": {"unstable": 3},
    }
    evidence = [_card(index, "unstable", "train") for index in range(3)]

    adjudication.adjudicate_cluster(
        cluster, evidence, 1, tmp_path, "model-a"
    )
    assert len(calls) == 2
    calls.clear()
    adjudication.adjudicate_cluster(
        cluster, evidence, 1, tmp_path, "model-a"
    )
    assert calls == []

    adjudication.adjudicate_cluster(
        cluster, evidence, 1, tmp_path, "model-b"
    )
    assert len(calls) == 2
    calls.clear()
    adjudication.adjudicate_cluster(
        cluster, evidence, 2, tmp_path, "model-a"
    )
    assert len(calls) == 3

    cache_files_before = set(tmp_path.glob("*.json"))
    monkeypatch.setattr(
        adjudication, "CLUSTER_SYSTEM", adjudication.CLUSTER_SYSTEM + "\nchanged"
    )
    calls.clear()
    adjudication.adjudicate_cluster(
        cluster, evidence, 1, tmp_path, "model-a"
    )
    assert len(calls) == 2
    assert set(tmp_path.glob("*.json")) != cache_files_before

    calls.clear()
    changed_evidence = json.loads(json.dumps(evidence))
    changed_evidence[0]["missing_operation"] = "changed evidence"
    adjudication.adjudicate_cluster(
        cluster, changed_evidence, 1, tmp_path, "model-a"
    )
    assert len(calls) == 2
