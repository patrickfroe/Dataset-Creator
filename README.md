# ragas_qa_dataset

CLI-Tool, das aus lokalen Dokumenten ein synthetisches Frage/Antwort-Dataset erzeugt und als `jsonl`/`csv` exportiert.

## Ziel des Projekts

`ragas_qa_dataset` hilft dir dabei, aus bestehenden Dokumenten schnell ein auswertbares QA-Dataset aufzubauen – mit reproduzierbarer Konfiguration (YAML + ENV), klaren CLI-Kommandos und einem festen Ausgabe-Schema.

Typischer Ablauf:
1. Dokumente in `data/raw` legen.
2. Konfiguration laden (`settings.yaml` + `OPENAI_API_KEY`).
3. Chunks erzeugen und Fragen/Antworten generieren.
4. Datensätze filtern und exportieren.

## Unterstützte Formate

### Eingabe
- `pdf`
- `docx`
- `md`
- `txt`

### Ausgabe
- `jsonl`
- `csv`

## Setup mit `uv`

### 1) Voraussetzungen
- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/)

### 2) Abhängigkeiten installieren
```bash
uv sync
```

### 3) Optional: Dev-Tools installieren (Tests/Lint)
```bash
uv sync --group dev
```

## `OPENAI_API_KEY` konfigurieren

Das Projekt bricht bewusst früh ab, wenn kein API-Key gesetzt ist.

### Variante A (empfohlen): über `.env`
```bash
cp .env.example .env
```
Danach in `.env` setzen:
```dotenv
OPENAI_API_KEY=sk-...
```

### Variante B: direkt pro Aufruf
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset show-config --config config/settings.yaml
```

## Beispiel für `settings.yaml`

Datei z. B. unter `config/settings.yaml` anlegen:

```yaml
input_dir: data/raw
file_types:
  - pdf
  - docx
  - md
  - txt
language: de
max_docs: null
chunk_size: 800
chunk_overlap: 120
testset_size: 50
distribution_preset: balanced
include_metadata: true
include_source_excerpt: true
output_formats:
  - jsonl
  - csv
random_seed: 42
openai_api_key: ""
```

Hinweise:
- `openai_api_key` kann leer bleiben, wenn `OPENAI_API_KEY` als ENV gesetzt ist.
- ENV-Variablen (`RAGAS_*`) überschreiben YAML-Werte.

## CLI-Beispiele

> Alle Befehle nutzen den Script-Entry-Point `ragas-qa-dataset`.

### 1) Konfiguration anzeigen
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset show-config --config config/settings.yaml
```

### 2) Setup validieren
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset validate --config config/settings.yaml
```

### 3) Dataset generieren (Standard: `fast`)
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset generate \
  --config config/settings.yaml \
  --output-dir data/output \
  --mode fast
```

### 4) Controlled-Modus mit Graph speichern
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset generate \
  --config config/settings.yaml \
  --output-dir data/output \
  --mode controlled \
  --graph-path data/output/kg.json \
  --save-graph
```

### 5) Controlled-Modus mit vorhandenem Graph laden
```bash
OPENAI_API_KEY=sk-... uv run ragas-qa-dataset generate \
  --config config/settings.yaml \
  --output-dir data/output \
  --mode controlled \
  --graph-path data/output/kg.json \
  --load-graph
```

## Unterschied zwischen `fast` und `controlled`

### `fast` (Default, MVP, stabil)
- Direkte Generierung aus vorbereiteten Chunks.
- Minimaler Overhead, schnellere Laufzeit.
- Gute Wahl für erste Datenaufbereitung und Iteration.

### `controlled` (optional/experimentell)
- Baut/lädt einen Knowledge-Graph und nutzt Ragas-Transforms.
- Bietet mehr Steuerungsmöglichkeiten über den Graph-Flow.
- Benötigt eine Ragas-Version mit `ragas.testset.graph` und `ragas.testset.transforms`.
- Wenn diese APIs fehlen, endet der Lauf mit klarer Fehlermeldung.

**Praxisregel:** Starte mit `fast`. Nutze `controlled` nur, wenn du gezielt den Graph-basierten Pfad brauchst.

## Output-Schema

Jeder Datensatz enthält mindestens folgende Pflichtfelder:

- `question`
- `answer`
- `source_doc`
- `chunk_id`
- `question_type`
- `difficulty`
- `language`

Optional kann zusätzlich `source_excerpt` enthalten sein (abhängig von der Konfiguration).

Beispiel (`jsonl`):
```json
{
  "question": "Was steht im Dokument über: '...'? (#1)",
  "answer": "...",
  "source_doc": "leitfaden.md",
  "chunk_id": "leitfaden.md::chunk-0001",
  "question_type": "factual",
  "difficulty": "easy",
  "language": "de",
  "source_excerpt": "..."
}
```

## Typische Fehlerquellen

- **`OPENAI_API_KEY fehlt`**  
  Key als ENV setzen oder in `settings.yaml` hinterlegen.

- **Ungültiger `input_dir`**  
  Prüfen, ob Ordner existiert und Dateien mit erlaubten Endungen enthält.

- **Nicht unterstützte `file_types`**  
  Nur `pdf`, `docx`, `md`, `txt` verwenden.

- **`chunk_overlap` ist größer/gleich `chunk_size`**  
  `chunk_overlap` muss kleiner als `chunk_size` sein.

- **Kein Export erzeugt**  
  In `output_formats` mindestens `jsonl` oder `csv` konfigurieren.

- **`controlled` schlägt fehl**  
  Ragas-Version prüfen (Graph-/Transform-Support erforderlich) oder auf `fast` wechseln.

## Weiterentwicklung (kurz)

Sinnvolle nächste Schritte:
- Zusätzliche Qualitätsmetriken/Filter in der Pipeline ergänzen.
- Verteilung (`distribution_preset`) um domänenspezifische Fragearten erweitern.
- Loader/Preprocessing für weitere Dokumentquellen ausbauen.
- Integrationstests mit kleinen Beispielkorpora ergänzen.

## Tests

```bash
uv run pytest -q
```
