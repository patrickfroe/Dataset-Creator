from __future__ import annotations

import argparse
from pathlib import Path

from ragas_qa_dataset.config import load_settings
from ragas_qa_dataset.exporters import export_jsonl
from ragas_qa_dataset.filters import deduplicate_questions, filter_by_context_length
from ragas_qa_dataset.generator import QAItem, generate_qa_items
from ragas_qa_dataset.loaders import load_local_documents
from ragas_qa_dataset.preprocess import preprocess_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a QA dataset from local documents.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/output/qa_dataset.jsonl"),
        help="Output JSONL file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings(args.config)
    input_dir = args.input_dir or settings.input_dir
    output_file: Path = args.output_file

    docs = load_local_documents(
        input_dir=input_dir,
        file_types=settings.file_types,
        max_docs=settings.max_docs,
    )
    chunks = preprocess_documents(
        documents=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        include_source_excerpt=settings.include_source_excerpt,
    )

    qa_items: list[QAItem] = []
    for chunk in chunks:
        qa_items.extend(
            generate_qa_items(
                source=chunk.source_doc,
                chunk=chunk.text,
                max_questions_per_chunk=1,
            )
        )

    filtered = filter_by_context_length(qa_items, min_chars=80)
    final_items = deduplicate_questions(filtered)
    export_jsonl(final_items, output_file)

    print(f"Generated {len(final_items)} QA items -> {output_file}")


if __name__ == "__main__":
    main()
