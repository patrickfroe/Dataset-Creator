import csv
import json
from pathlib import Path

from ragas_qa_dataset.exporters import OUTPUT_SCHEMA, export_csv, export_jsonl


def test_export_jsonl_writes_stable_schema_and_timestamped_filename(tmp_path: Path) -> None:
    records = [
        {
            "question": "Q?",
            "answer": "Antwort",
            "source_doc": "file.txt",
            "chunk_id": "file.txt:0",
            "question_type": "factual",
            "difficulty": "easy",
            "language": "de",
        }
    ]

    out_file = export_jsonl(records, tmp_path / "dataset.jsonl", timestamp="20260331_120000")

    assert out_file.name == "dataset_20260331_120000.jsonl"
    payload = json.loads(out_file.read_text(encoding="utf-8").strip())
    assert tuple(payload.keys()) == OUTPUT_SCHEMA
    assert payload["source_excerpt"] == ""


def test_export_csv_writes_header_and_rows_with_optional_fields(tmp_path: Path) -> None:
    records = [
        {
            "question": "Q1",
            "answer": "A1",
            "source_doc": "doc.md",
            "chunk_id": "doc.md:0",
            "question_type": "reasoning",
            "difficulty": "medium",
            "language": "de",
            "source_excerpt": "kurzer Auszug",
        },
        {
            "question": "Q2",
            "answer": "A2",
            "source": "fallback-source.txt",
        },
    ]

    out_file = export_csv(records, tmp_path, timestamp="20260331_120000")

    assert out_file.name == "qa_dataset_20260331_120000.csv"

    with out_file.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == OUTPUT_SCHEMA
    assert rows[0]["source_excerpt"] == "kurzer Auszug"
    assert rows[1]["source_doc"] == "fallback-source.txt"
    assert rows[1]["chunk_id"] == ""
