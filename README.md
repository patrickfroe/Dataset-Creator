# ragas_qa_dataset

Ein kleines Python-Tool, das aus lokalen Dokumenten ein QA-Dataset vorbereitet und mit Ragas-kompatiblen Strukturen exportiert.

## Features
- Laden lokaler Dateien
- YAML + ENV Konfiguration
- Einfaches Preprocessing (Whitespace-Normalisierung)
- Generierung von QA-Beispielen aus Textabschnitten
- Qualitätsfilter (Mindestlänge, Duplikatentfernung)
- Export als JSONL
- CLI-Einstiegspunkt

## Voraussetzungen
- Python **3.11+**
- [uv](https://docs.astral.sh/uv/)
- `OPENAI_API_KEY` gesetzt (ENV oder YAML)

## Installation
```bash
uv sync
```

## Konfiguration
1. Kopiere `.env.example` nach `.env` und setze `OPENAI_API_KEY`.
2. Kopiere `config/settings.example.yaml` zu `config/settings.yaml` und passe Werte an.

## CLI
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset --config config/settings.yaml --output-file data/output/qa_dataset.jsonl
```

## Tests
```bash
PYTHONPATH=src python -m pytest -q
```
