from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ragas_qa_dataset.config import Settings, SettingsError, load_settings
from ragas_qa_dataset.exporters import export_csv, export_jsonl
from ragas_qa_dataset.filters import clean_dataset_records
from ragas_qa_dataset.generator import GeneratedSample, generate_testset_from_prepared_documents
from ragas_qa_dataset.loaders import load_local_documents
from ragas_qa_dataset.preprocess import preprocess_documents

SUPPORTED_FILE_TYPES = {"pdf", "docx", "md", "txt"}
SUPPORTED_OUTPUT_FORMATS = {"jsonl", "csv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ragas QA Dataset CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate dataset from local documents")
    generate_parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    generate_parser.add_argument("--input-dir", type=Path, default=None, help="Override input directory")
    generate_parser.add_argument("--output-dir", type=Path, default=Path("data/output"), help="Output directory")
    generate_parser.add_argument("--mode", choices=["fast", "controlled"], default="fast", help="Generation mode")
    generate_parser.add_argument("--graph-path", type=Path, default=None, help="Path for controlled graph JSON")
    generate_parser.add_argument("--load-graph", action="store_true", help="Load graph instead of building one")
    generate_parser.add_argument("--save-graph", action="store_true", help="Save graph after generation")

    validate_parser = subparsers.add_parser("validate", help="Validate configuration and input setup")
    validate_parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")

    show_parser = subparsers.add_parser("show-config", help="Show effective configuration")
    show_parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")

    return parser


def _samples_to_records(samples: list[GeneratedSample], include_source_excerpt: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        payload = asdict(sample)
        source_excerpt = ""
        if include_source_excerpt:
            source_excerpt = payload.get("context", "")[:200]

        records.append(
            {
                "question": payload["question"],
                "answer": payload["answer"],
                "source_doc": payload["source_doc"],
                "chunk_id": payload["chunk_id"],
                "question_type": payload["question_type"],
                "difficulty": payload["difficulty"],
                "language": payload["language"],
                "source_excerpt": source_excerpt,
            }
        )
    return records


def _validate_settings(settings: Settings) -> list[str]:
    errors: list[str] = []

    if not settings.input_dir.exists() or not settings.input_dir.is_dir():
        errors.append(f"Input path is invalid: {settings.input_dir}")

    invalid_types = [ft for ft in settings.file_types if ft.lower().lstrip(".") not in SUPPORTED_FILE_TYPES]
    if invalid_types:
        errors.append(f"Unsupported file types configured: {', '.join(invalid_types)}")

    invalid_formats = [fmt for fmt in settings.output_formats if fmt.lower() not in SUPPORTED_OUTPUT_FORMATS]
    if invalid_formats:
        errors.append(f"Unsupported output formats configured: {', '.join(invalid_formats)}")

    return errors


def run_generate(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"❌ Konfigurationsfehler: {exc}")
        return 1

    input_dir = args.input_dir or settings.input_dir
    print(f"[generate] Lade Dokumente aus: {input_dir}")
    docs = load_local_documents(
        input_dir=input_dir,
        file_types=settings.file_types,
        max_docs=settings.max_docs,
    )
    print(f"[generate] Dokumente geladen: {len(docs)}")

    chunks = preprocess_documents(
        documents=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        include_source_excerpt=settings.include_source_excerpt,
    )
    print(f"[generate] Chunks erzeugt: {len(chunks)}")

    testset = generate_testset_from_prepared_documents(
        chunks=chunks,
        testset_size=settings.testset_size,
        distribution_preset=settings.distribution_preset,
        language=settings.language,
        openai_api_key=settings.openai_api_key,
        mode=args.mode,
        graph_path=args.graph_path,
        load_graph=args.load_graph,
        save_graph=args.save_graph,
    )
    print(f"[generate] Testset generiert: {len(testset.samples)} Samples (mode={testset.mode})")

    records = _samples_to_records(testset.samples, include_source_excerpt=settings.include_source_excerpt)
    cleaned = clean_dataset_records(
        records,
        min_answer_chars=10,
        language=settings.language,
        with_quality_flags=False,
    )
    print(f"[generate] Nach Filterung: {len(cleaned)} Datensätze")

    output_dir: Path = args.output_dir
    exports: list[Path] = []
    formats = {fmt.lower() for fmt in settings.output_formats if fmt.lower() in SUPPORTED_OUTPUT_FORMATS}
    if "jsonl" in formats:
        exports.append(export_jsonl(cleaned, output_dir / "qa_dataset.jsonl"))
    if "csv" in formats:
        exports.append(export_csv(cleaned, output_dir / "qa_dataset.csv"))

    if not exports:
        print("⚠️ Keine unterstützten Ausgabeformate konfiguriert (jsonl/csv).")
        return 1

    print("[generate] Export abgeschlossen:")
    for path in exports:
        print(f"  - {path}")

    return 0


def run_validate(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"❌ Konfigurationsfehler: {exc}")
        return 1

    errors = _validate_settings(settings)
    if errors:
        print("❌ Validierung fehlgeschlagen:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("✅ Validierung erfolgreich.")
    print(f"  input_dir: {settings.input_dir}")
    print(f"  file_types: {', '.join(settings.file_types)}")
    print("  OPENAI_API_KEY: gesetzt")
    return 0


def run_show_config(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"❌ Konfigurationsfehler: {exc}")
        return 1

    print(json.dumps(asdict(settings), default=str, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)
    if args.command == "validate":
        return run_validate(args)
    if args.command == "show-config":
        return run_show_config(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
