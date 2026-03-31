from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(slots=True)
class Settings:
    input_dir: Path = Path("data/raw")
    file_types: tuple[str, ...] = ("pdf", "docx", "md", "txt")
    language: str = "de"
    max_docs: int | None = None
    chunk_size: int = 800
    chunk_overlap: int = 120
    testset_size: int = 50
    distribution_preset: str = "balanced"
    include_metadata: bool = True
    include_source_excerpt: bool = True
    output_formats: tuple[str, ...] = ("jsonl",)
    random_seed: int = 42
    openai_api_key: str = ""


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return value


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse a minimal YAML subset (key/value + list values)."""
    data: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("-") and current_list_key is not None:
            current = data.setdefault(current_list_key, [])
            if isinstance(current, list):
                current.append(_parse_scalar(line.removeprefix("-").strip()))
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()

        if value == "":
            data[key] = []
            current_list_key = key
            continue

        current_list_key = None
        data[key] = _parse_scalar(value)

    return data


def _coerce_tuple_str(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise SettingsError(f"Field '{field_name}' must be a list or comma-separated string.")


def _coerce_optional_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and (value.isdigit() or (value.startswith("-") and value[1:].isdigit())):
        return int(value)
    raise SettingsError(f"Field '{field_name}' must be an integer or null.")


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise SettingsError(f"Field '{field_name}' must be a boolean.")


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and (value.isdigit() or (value.startswith("-") and value[1:].isdigit())):
        return int(value)
    raise SettingsError(f"Field '{field_name}' must be an integer.")


def _to_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _parse_simple_yaml(Path(path))


def load_settings(path: str | Path | None = None) -> Settings:
    payload = _to_payload(path)

    raw_openai_api_key = os.getenv("OPENAI_API_KEY", str(payload.get("openai_api_key", ""))).strip()
    if not raw_openai_api_key:
        raise SettingsError(
            "OPENAI_API_KEY fehlt. Bitte setze die Umgebungsvariable OPENAI_API_KEY "
            "oder den Wert 'openai_api_key' in der YAML-Konfiguration."
        )

    input_dir = Path(str(os.getenv("RAGAS_INPUT_DIR", payload.get("input_dir", "data/raw"))))
    file_types = _coerce_tuple_str(os.getenv("RAGAS_FILE_TYPES", payload.get("file_types", ["pdf", "docx", "md", "txt"])), "file_types")
    output_formats = _coerce_tuple_str(os.getenv("RAGAS_OUTPUT_FORMATS", payload.get("output_formats", ["jsonl"])), "output_formats")

    language = str(os.getenv("RAGAS_LANGUAGE", payload.get("language", "de"))).strip()
    max_docs = _coerce_optional_int(os.getenv("RAGAS_MAX_DOCS", payload.get("max_docs")), "max_docs")
    chunk_size = _coerce_int(os.getenv("RAGAS_CHUNK_SIZE", payload.get("chunk_size", 800)), "chunk_size")
    chunk_overlap = _coerce_int(os.getenv("RAGAS_CHUNK_OVERLAP", payload.get("chunk_overlap", 120)), "chunk_overlap")
    testset_size = _coerce_int(os.getenv("RAGAS_TESTSET_SIZE", payload.get("testset_size", 50)), "testset_size")
    distribution_preset = str(os.getenv("RAGAS_DISTRIBUTION_PRESET", payload.get("distribution_preset", "balanced"))).strip()
    include_metadata = _coerce_bool(os.getenv("RAGAS_INCLUDE_METADATA", payload.get("include_metadata", True)), "include_metadata")
    include_source_excerpt = _coerce_bool(os.getenv("RAGAS_INCLUDE_SOURCE_EXCERPT", payload.get("include_source_excerpt", True)), "include_source_excerpt")
    random_seed = _coerce_int(os.getenv("RAGAS_RANDOM_SEED", payload.get("random_seed", 42)), "random_seed")

    if chunk_size <= 0:
        raise SettingsError("Field 'chunk_size' must be > 0.")
    if chunk_overlap < 0:
        raise SettingsError("Field 'chunk_overlap' must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise SettingsError("Field 'chunk_overlap' must be smaller than 'chunk_size'.")
    if testset_size <= 0:
        raise SettingsError("Field 'testset_size' must be > 0.")

    return Settings(
        input_dir=input_dir,
        file_types=file_types,
        language=language,
        max_docs=max_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        testset_size=testset_size,
        distribution_preset=distribution_preset,
        include_metadata=include_metadata,
        include_source_excerpt=include_source_excerpt,
        output_formats=output_formats,
        random_seed=random_seed,
        openai_api_key=raw_openai_api_key,
    )
