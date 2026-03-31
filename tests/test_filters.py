import pytest

from ragas_qa_dataset.filters import clean_dataset_records


def test_clean_dataset_records_removes_duplicates_by_question() -> None:
    records = [
        {"question": " Was ist X? ", "answer": "Eine ausreichend lange Antwort.", "source_doc": "a.md", "language": "de"},
        {"question": "was ist x?", "answer": "Noch eine Antwort, die ignoriert werden soll.", "source_doc": "b.md", "language": "de"},
    ]

    result = clean_dataset_records(records, min_answer_chars=10)

    assert len(result) == 1
    assert result[0]["question"] == "Was ist X?"


def test_clean_dataset_records_discards_short_or_empty_answers() -> None:
    records = [
        {"question": "Q1", "answer": "", "source_doc": "a.md", "language": "de"},
        {"question": "Q2", "answer": "kurz", "source_doc": "a.md", "language": "de"},
        {"question": "Q3", "answer": "Das ist lang genug.", "source_doc": "a.md", "language": "de"},
    ]

    result = clean_dataset_records(records, min_answer_chars=8)

    assert [row["question"] for row in result] == ["Q3"]


def test_clean_dataset_records_supports_optional_language_filter() -> None:
    records = [
        {"question": "Q1", "answer": "Lange Antwort für Deutsch.", "source_doc": "a.md", "language": "de"},
        {"question": "Q2", "answer": "Long enough English answer.", "source_doc": "a.md", "language": "en"},
    ]

    result = clean_dataset_records(records, min_answer_chars=10, language="de")

    assert len(result) == 1
    assert result[0]["language"] == "de"


def test_clean_dataset_records_adds_quality_flags_when_enabled() -> None:
    records = [
        {
            "question": "Q1",
            "answer": "Dies ist eine gute Antwort.",
            "source_doc": "file.md",
            "language": "de",
        }
    ]

    result = clean_dataset_records(records, min_answer_chars=10, with_quality_flags=True)

    flags = result[0]["quality_flags"]
    assert flags == {
        "has_question": True,
        "has_answer": True,
        "has_source": True,
        "answer_length_ok": True,
    }


def test_clean_dataset_records_rejects_negative_min_answer_chars() -> None:
    with pytest.raises(ValueError, match="min_answer_chars"):
        clean_dataset_records([], min_answer_chars=-1)
