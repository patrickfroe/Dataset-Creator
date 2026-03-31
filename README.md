# ragas_qa_dataset

CLI-Tool, das aus lokalen Dokumenten ein synthetisches Frage/Antwort-Dataset mit einem LLM erzeugt und als Excel-Datei exportiert.

## Features

- Eingabeformate: `pdf`, `docx`, `md`, `txt`
- Chunking ohne Worttrennung in der Mitte
- QA-Generierung explizit per Azure OpenAI LLM
- Optionaler `controlled`-Modus mit Ragas Knowledge-Graph-Transforms
- Primärer Export: `.xlsx` mit **genau** den Spalten `id`, `question`, `answer`

## Installation

### Mit `uv` (empfohlen)

```bash
uv sync
```

### Mit `pip`

```bash
pip install -r requirements.txt
```

## Azure OpenAI Konfiguration

Pflicht-ENV-Variablen:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

Optional, aber empfohlen:

- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

Beispiel `.env`:

```dotenv
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

RAGAS_INPUT_DIR=data/raw
RAGAS_OUTPUT_FORMATS=xlsx
RAGAS_CHUNK_SIZE=800
RAGAS_CHUNK_OVERLAP=120
RAGAS_TESTSET_SIZE=50
```

Wenn API-Key oder Endpoint fehlen, bricht das Tool mit klarer Fehlermeldung ab.

## Beispiel `config/settings.yaml`

```yaml
input_dir: data/raw
file_types:
  - pdf
  - docx
  - md
  - txt
language: de
chunk_size: 800
chunk_overlap: 120
testset_size: 50
distribution_preset: balanced
output_formats:
  - xlsx
azure_openai_api_key: ""
azure_openai_endpoint: ""
azure_openai_api_version: "2024-10-21"
azure_openai_chat_deployment: "gpt-4o-mini"
azure_openai_embedding_deployment: "text-embedding-3-small"
```

## Ausführen

```bash
AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/ \
uv run ragas-qa-dataset generate \
  --config config/settings.yaml \
  --output-dir data/output \
  --mode fast
```

Beispiel-Ausgabe:

- `data/output/qa_dataset_20260331_120000.xlsx`

## Excel-Ausgabe

Die Excel-Datei enthält exakt diese 3 Spalten in dieser Reihenfolge:

1. `id`
2. `question`
3. `answer`

`id` wird als zero-padded String ohne Präfix erzeugt (`001`, `002`, ...).

## Pipeline-Überblick

1. Dokumente laden
2. Text normalisieren und in Chunks aufteilen (ohne Worttrennung)
3. Pro Chunk QA via Azure OpenAI LLM generieren
4. Datensätze filtern
5. Export nach `.xlsx`

Hinweis: Embeddings werden im `controlled`-Modus über Azure OpenAI initialisiert und bei Ragas-Transforms genutzt.

## Tests

```bash
uv run pytest -q
```
