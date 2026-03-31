from __future__ import annotations

import re
from dataclasses import dataclass

from ragas_qa_dataset.loaders import Document


@dataclass(slots=True)
class ProcessedChunk:
    chunk_id: str
    text: str
    source_doc: str
    file_path: str
    file_type: str
    source_excerpt: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    index = 0
    start = 0
    while start < len(text):
        end = _find_chunk_end(text=text, start=start, chunk_size=chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((index, chunk))
            index += 1
        if end >= len(text):
            break
        next_start = max(0, end - chunk_overlap)
        next_start = _align_start_to_word(text=text, position=next_start)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    ideal_end = min(start + chunk_size, len(text))
    if ideal_end == len(text):
        return ideal_end

    boundary = text.rfind(" ", start + 1, ideal_end + 1)
    if boundary == -1 or boundary <= start:
        return ideal_end
    return boundary


def _align_start_to_word(text: str, position: int) -> int:
    if position <= 0 or position >= len(text):
        return max(0, min(position, len(text)))
    while position > 0 and text[position - 1] != " " and text[position] != " ":
        position -= 1
    return position


def preprocess_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    include_source_excerpt: bool = False,
    excerpt_chars: int = 200,
) -> list[ProcessedChunk]:
    processed_chunks: list[ProcessedChunk] = []

    for doc in documents:
        normalized = normalize_text(doc.text)
        if not normalized:
            continue

        excerpt = normalized[:excerpt_chars] if include_source_excerpt else None

        for chunk_idx, chunk in chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            processed_chunks.append(
                ProcessedChunk(
                    chunk_id=f"{doc.source_doc}:{chunk_idx}",
                    text=chunk,
                    source_doc=doc.source_doc,
                    file_path=doc.file_path,
                    file_type=doc.file_type,
                    source_excerpt=excerpt,
                )
            )

    return processed_chunks
