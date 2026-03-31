from __future__ import annotations

from typing import Any, TypedDict

from ragas_qa_dataset.generator import QAItem


class QualityFlags(TypedDict):
    has_question: bool
    has_answer: bool
    has_source: bool
    answer_length_ok: bool


def filter_by_context_length(items: list[QAItem], min_chars: int) -> list[QAItem]:
    return [item for item in items if len(item.context) >= min_chars]


def deduplicate_questions(items: list[QAItem]) -> list[QAItem]:
    seen: set[str] = set()
    unique: list[QAItem] = []
    for item in items:
        key = item.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _build_quality_flags(record: dict[str, Any], min_answer_chars: int) -> QualityFlags:
    question = _normalize_text(record.get("question"))
    answer = _normalize_text(record.get("answer"))
    source = _normalize_text(record.get("source_doc") or record.get("source"))
    return QualityFlags(
        has_question=bool(question),
        has_answer=bool(answer),
        has_source=bool(source),
        answer_length_ok=len(answer) >= min_answer_chars,
    )


def clean_dataset_records(
    records: list[dict[str, Any]],
    min_answer_chars: int = 10,
    language: str | None = None,
    with_quality_flags: bool = False,
) -> list[dict[str, Any]]:
    """Return cleaned dataset records.

    Rules:
    - remove duplicates by normalized question text
    - discard rows with empty/very short answers
    - optional language filter
    - optional quality flag enrichment
    """
    if min_answer_chars < 0:
        raise ValueError("min_answer_chars must be >= 0")

    language_filter = language.strip().lower() if language else None
    seen_questions: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    for raw_record in records:
        question = _normalize_text(raw_record.get("question"))
        answer = _normalize_text(raw_record.get("answer"))
        record_language = _normalize_text(raw_record.get("language"))

        if len(answer) < min_answer_chars:
            continue

        if language_filter and record_language.lower() != language_filter:
            continue

        dedup_key = question.lower()
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        cleaned_record = dict(raw_record)
        cleaned_record["question"] = question
        cleaned_record["answer"] = answer
        cleaned_record["language"] = record_language

        if with_quality_flags:
            cleaned_record["quality_flags"] = _build_quality_flags(
                cleaned_record,
                min_answer_chars=min_answer_chars,
            )

        cleaned.append(cleaned_record)

    return cleaned
