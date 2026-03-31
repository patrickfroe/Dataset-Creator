# AGENTS.md

## Ziel
- Erzeuge aus lokalen Dokumenten ein synthetisches Frage/Antwort-Dataset mit Ragas.

## Arbeitsprinzipien
- Liefere zuerst ein MVP; vermeide Overengineering.
- Nutze Python 3.11, `uv`, `pytest` und konsequente Type Hints.
- Bevorzuge kleine Funktionen und klar getrennte Module.
- Baue keine UI; dieses Projekt ist ausschließlich CLI-basiert.

## Datenformate
- Unterstützte Eingaben: `pdf`, `docx`, `md`, `txt`.
- Ausgabe-Schema (Pflichtfelder):
  - `question`
  - `answer`
  - `source_doc`
  - `chunk_id`
  - `question_type`
  - `difficulty`
  - `language`
- Unterstützte Exporte: `jsonl` und `csv`.

## Konfiguration & Robustheit
- Konfiguriere über YAML + ENV.
- Behandle Fehler explizit und mit klaren Fehlermeldungen.

## Qualität
- Ergänze Tests bei jeder Verhaltensänderung.
- Halte die README mit einer echten, lauffähigen Run-Anleitung aktuell.
