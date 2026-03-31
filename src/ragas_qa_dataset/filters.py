from __future__ import annotations

from ragas_qa_dataset.generator import QAItem


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
