import json

from skillopt.envs.spreadsheetbench.adapter import SpreadsheetBenchAdapter


def test_spreadsheetbench_adapter_forwards_split_limit(tmp_path) -> None:
    split_dir = tmp_path / "splits"
    items = [{"id": str(index)} for index in range(5)]
    for split in ("train", "val", "test"):
        target = split_dir / split
        target.mkdir(parents=True)
        (target / "items.json").write_text(json.dumps(items), encoding="utf-8")

    adapter = SpreadsheetBenchAdapter(
        split_dir=str(split_dir),
        split_mode="split_dir",
        limit=2,
    )
    adapter.setup({})

    dataloader = adapter.get_dataloader()
    assert len(dataloader.train_items) == 2
    assert len(dataloader.val_items) == 2
    assert len(dataloader.test_items) == 2


def test_spreadsheetbench_adapter_uses_layered_timeout_defaults() -> None:
    adapter = SpreadsheetBenchAdapter()
    assert adapter.exec_timeout == 1200
    assert adapter.llm_timeout == 300
