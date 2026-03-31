from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA: tuple[str, ...] = (
    "question",
    "answer",
    "source_doc",
    "chunk_id",
    "question_type",
    "difficulty",
    "language",
    "source_excerpt",
)


def _timestamp(now: datetime | None = None) -> str:
    current = now or datetime.utcnow()
    return current.strftime("%Y%m%d_%H%M%S")


def _resolve_output_path(
    output_path: Path,
    suffix: str,
    timestamp: str | None = None,
) -> Path:
    ts = timestamp or _timestamp()

    if output_path.suffix:
        output_dir = output_path.parent
        stem = output_path.stem
    else:
        output_dir = output_path
        stem = "qa_dataset"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}_{ts}.{suffix}"


def _item_to_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        raw = dict(item)
    elif is_dataclass(item):
        raw = asdict(item)
    else:
        raw = {
            key: getattr(item, key)
            for key in ("question", "answer", "source_doc", "chunk_id", "question_type", "difficulty", "language", "source_excerpt", "source")
            if hasattr(item, key)
        }

    source_doc = str(raw.get("source_doc") or raw.get("source") or "").strip()
    payload: dict[str, str] = {
        "question": str(raw.get("question", "")).strip(),
        "answer": str(raw.get("answer", "")).strip(),
        "source_doc": source_doc,
        "chunk_id": str(raw.get("chunk_id", "")).strip(),
        "question_type": str(raw.get("question_type", "")).strip(),
        "difficulty": str(raw.get("difficulty", "")).strip(),
        "language": str(raw.get("language", "")).strip(),
        "source_excerpt": str(raw.get("source_excerpt", "")).strip(),
    }
    return payload


def export_jsonl(items: list[Any], output_path: Path, timestamp: str | None = None) -> Path:
    output_file = _resolve_output_path(output_path=output_path, suffix="jsonl", timestamp=timestamp)
    with output_file.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(_item_to_payload(item), ensure_ascii=False) + "\n")
    return output_file


def export_csv(items: list[Any], output_path: Path, timestamp: str | None = None) -> Path:
    output_file = _resolve_output_path(output_path=output_path, suffix="csv", timestamp=timestamp)
    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(OUTPUT_SCHEMA))
        writer.writeheader()
        for item in items:
            writer.writerow(_item_to_payload(item))
    return output_file
