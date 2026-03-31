import json
from pathlib import Path

from ragas_qa_dataset.exporters import export_jsonl
from ragas_qa_dataset.generator import QAItem


def test_export_jsonl_writes_expected_rows(tmp_path: Path) -> None:
    out = tmp_path / "dataset.jsonl"
    items = [
        QAItem(
            question="Q?",
            answer="A",
            context="C",
            source="file.txt",
        )
    ]

    export_jsonl(items, out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["question"] == "Q?"
    assert payload["contexts"] == ["C"]
