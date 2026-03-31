# ragas_qa_dataset

Ein kleines Python-Tool, das aus lokalen Dokumenten ein QA-Dataset vorbereitet und mit Ragas-kompatiblen Strukturen exportiert.

## Features
- Laden lokaler Dateien (`pdf`, `docx`, `md`, `txt`)
- YAML + ENV Konfiguration
- Einfaches Preprocessing (Whitespace-Normalisierung + Chunking)
- Zwei Generierungsmodi mit gemeinsamer Export-Schnittstelle:
  - **fast (MVP, Standard):** direkte Chunk-basierte Generierung
  - **controlled (optional):** Aufbau/Laden eines KnowledgeGraph + `default_transforms`
- Qualitätsfilter (Mindestlänge, Duplikatentfernung)
- Export als JSONL (MVP)
- CLI-Einstiegspunkt

## Voraussetzungen
- Python **3.11+**
- [uv](https://docs.astral.sh/uv/)
- `OPENAI_API_KEY` gesetzt (ENV oder YAML)
- Für `controlled` zusätzlich eine Ragas-Version mit `ragas.testset.graph` und `ragas.testset.transforms`

## Installation
```bash
uv sync
```

## Konfiguration
1. Kopiere `.env.example` nach `.env` und setze `OPENAI_API_KEY`.
2. Kopiere `config/settings.example.yaml` zu `config/settings.yaml` und passe Werte an.

## CLI (MVP fast)
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset --config config/settings.yaml --output-file data/output/qa_dataset.jsonl
```

## Controlled-Modus (API)
`generate_testset_from_prepared_documents(...)` unterstützt jetzt optionale Graph-Steuerung:

- `mode="controlled"`
- `graph_path=".../kg.json"`
- `save_graph=True` (Graph persistieren)
- `load_graph=True` (Graph statt Chunk-Liste laden)

Beispiel:
```python
from ragas_qa_dataset.generator import generate_testset_from_prepared_documents

result = generate_testset_from_prepared_documents(
    chunks=prepared_chunks,
    testset_size=20,
    distribution_preset="balanced",
    language="de",
    openai_api_key="sk-...",
    mode="controlled",
    graph_path="data/output/kg.json",
    save_graph=True,
)
```

## Stabilitätshinweis zu `controlled`
`controlled` ist bewusst klar vom MVP getrennt und aktuell als **optional/experimentell** zu betrachten.
Die produktiv stabile Standardroute bleibt `mode="fast"`.
Wenn die lokale Ragas-Version keine `KnowledgeGraph`/`default_transforms`-APIs bereitstellt, bricht `controlled` mit einer klaren Fehlermeldung ab.

## Tests
```bash
PYTHONPATH=src python -m pytest -q
```
