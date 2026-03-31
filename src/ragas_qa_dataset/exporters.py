from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA: tuple[str, ...] = ("id", "question", "answer")


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


def _row_id(index: int) -> str:
    return f"{index:03d}"


def export_xlsx(items: list[Any], output_path: Path, timestamp: str | None = None) -> Path:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Missing dependency 'openpyxl'. Install it to enable Excel export."
        ) from exc

    output_file = _resolve_output_path(output_path=output_path, suffix="xlsx", timestamp=timestamp)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "qa_dataset"
    worksheet.append(list(OUTPUT_SCHEMA))

    for index, item in enumerate(items, start=1):
        payload = _item_to_payload(item)
        worksheet.append([_row_id(index), payload["question"], payload["answer"]])

    workbook.save(output_file)
    return output_file
