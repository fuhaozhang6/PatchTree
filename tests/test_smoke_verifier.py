import json

from scripts.runs.smoke.resource_pool_epoch1.verify_smoke import verify_dataset


def _write_spreadsheet_smoke(tmp_path, row: dict) -> None:
    root = tmp_path / "spreadsheetbench"
    rollout = root / "steps" / "step_0001" / "rollout"
    rollout.mkdir(parents=True)
    summary = {
        "config": {
            "num_epochs": 1,
            "train_size": 2,
            "batch_size": 2,
            "eval_test": False,
            "type_guided_version": "v2",
        },
        "total_steps": 1,
        "epoch_stats": [{"epoch": 1}],
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (rollout / "results.jsonl").write_text(
        "\n".join(json.dumps({"id": str(index), **row}) for index in range(2)) + "\n",
        encoding="utf-8",
    )


def test_spreadsheet_smoke_accepts_completed_wrong_answers(tmp_path) -> None:
    _write_spreadsheet_smoke(
        tmp_path,
        {"llm_ok": True, "phase": "eval", "hard": 0, "fail_reason": "eval-mismatch"},
    )

    assert verify_dataset(tmp_path, "spreadsheetbench", 2) == []


def test_spreadsheet_smoke_rejects_task_timeouts(tmp_path) -> None:
    _write_spreadsheet_smoke(
        tmp_path,
        {"llm_ok": False, "phase": "timeout", "hard": 0, "fail_reason": "task-timeout-1200s"},
    )

    errors = verify_dataset(tmp_path, "spreadsheetbench", 2)

    assert any("timeout/infrastructure-failure" in error for error in errors)
