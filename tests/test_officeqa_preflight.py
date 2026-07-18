import json

import pytest

from skillopt.envs.officeqa.adapter import OfficeQAAdapter
from skillopt.envs.officeqa.rollout import _decode_tool_arguments


def _write_officeqa_splits(root, *, source_file: str) -> None:
    item = {
        "id": "q1",
        "question": "What is the reported value?",
        "answer": "42",
        "source_files": [source_file],
        "source_docs": [],
    }
    for split in ("train", "val", "test"):
        target = root / split
        target.mkdir(parents=True)
        (target / "items.json").write_text(json.dumps([item]), encoding="utf-8")


def test_officeqa_preflight_accepts_matching_local_evidence(tmp_path) -> None:
    split_dir = tmp_path / "splits"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "source.txt").write_text("reported value: 42", encoding="utf-8")
    _write_officeqa_splits(split_dir, source_file="source.txt")

    adapter = OfficeQAAdapter(
        split_dir=str(split_dir),
        data_dirs=str(docs_dir),
        limit=1,
    )
    adapter.setup({})


def test_officeqa_preflight_rejects_wrong_local_corpus(tmp_path) -> None:
    split_dir = tmp_path / "splits"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "unrelated.txt").write_text("unrelated", encoding="utf-8")
    _write_officeqa_splits(split_dir, source_file="missing.txt")

    adapter = OfficeQAAdapter(
        split_dir=str(split_dir),
        data_dirs=str(docs_dir),
        limit=1,
    )
    with pytest.raises(RuntimeError, match="none of the sampled split references"):
        adapter.setup({})


def test_decode_tool_arguments_accepts_valid_object() -> None:
    assert _decode_tool_arguments('{"path": "source.txt", "start": 1}') == {
        "path": "source.txt",
        "start": 1,
    }


def test_decode_tool_arguments_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        _decode_tool_arguments('["source.txt"]')
