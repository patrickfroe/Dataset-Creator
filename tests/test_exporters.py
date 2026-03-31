import sys
import types
from pathlib import Path

from ragas_qa_dataset.exporters import OUTPUT_SCHEMA, export_xlsx


class _WorksheetStub:
    def __init__(self) -> None:
        self.title = ""
        self.rows: list[tuple[str, str, str] | tuple[str, ...]] = []

    def append(self, row: list[str]) -> None:
        self.rows.append(tuple(row))


class _WorkbookStub:
    saved_path: Path | None = None
    last_instance: "_WorkbookStub | None" = None

    def __init__(self) -> None:
        self.active = _WorksheetStub()
        _WorkbookStub.last_instance = self

    def save(self, path: Path) -> None:
        _WorkbookStub.saved_path = path
        path.write_text("stub-xlsx", encoding="utf-8")


def _install_openpyxl_stub() -> None:
    module = types.SimpleNamespace(Workbook=_WorkbookStub)
    sys.modules["openpyxl"] = module


def test_export_xlsx_writes_exact_three_columns_and_order(tmp_path: Path) -> None:
    _install_openpyxl_stub()
    records = [
        {"question": "Q1", "answer": "A1", "source_doc": "doc.md", "chunk_id": "doc.md:0"},
        {"question": "Q2", "answer": "A2", "source_doc": "doc2.md", "chunk_id": "doc2.md:1"},
    ]

    out_file = export_xlsx(records, tmp_path / "dataset.xlsx", timestamp="20260331_120000")

    assert out_file.name == "dataset_20260331_120000.xlsx"
    assert _WorkbookStub.saved_path == out_file
    worksheet = _WorkbookStub.last_instance.active
    assert worksheet.rows[0] == OUTPUT_SCHEMA
    assert worksheet.rows[1] == ("001", "Q1", "A1")
    assert worksheet.rows[2] == ("002", "Q2", "A2")


def test_export_xlsx_generates_zero_padded_id_without_prefix(tmp_path: Path) -> None:
    _install_openpyxl_stub()
    export_xlsx([{"question": "Q", "answer": "A"}], tmp_path, timestamp="20260331_120000")

    worksheet = _WorkbookStub.last_instance.active
    assert worksheet.rows[1][0] == "001"
