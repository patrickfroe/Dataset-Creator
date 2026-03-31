from __future__ import annotations

import json
from pathlib import Path

from ragas_qa_dataset.generator import QAItem


def export_jsonl(items: list[QAItem], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for item in items:
            row = {
                "question": item.question,
                "answer": item.answer,
                "contexts": [item.context],
                "ground_truth": item.answer,
                "metadata": {"source": item.source},
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
