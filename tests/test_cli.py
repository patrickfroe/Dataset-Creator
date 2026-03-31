from pathlib import Path

from ragas_qa_dataset.cli import main
from ragas_qa_dataset.config import Settings
from ragas_qa_dataset.generator import GeneratedSample, GeneratedTestset


def test_show_config_command_prints_effective_config(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "ragas_qa_dataset.cli.load_settings",
        lambda _: Settings(
            azure_openai_api_key="azure-key",
            azure_openai_endpoint="https://example.openai.azure.com/",
        ),
    )

    exit_code = main(["show-config"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"azure_openai_api_key": "azure-key"' in out


def test_validate_command_detects_invalid_file_types(monkeypatch, capsys, tmp_path: Path) -> None:
    invalid_settings = Settings(
        input_dir=tmp_path,
        file_types=("txt", "xlsx"),
        azure_openai_api_key="azure-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
    )
    monkeypatch.setattr("ragas_qa_dataset.cli.load_settings", lambda _: invalid_settings)

    exit_code = main(["validate"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Unsupported file types configured" in out


def test_validate_command_detects_invalid_output_formats(monkeypatch, capsys, tmp_path: Path) -> None:
    invalid_settings = Settings(
        input_dir=tmp_path,
        file_types=("txt",),
        output_formats=("jsonl",),
        azure_openai_api_key="azure-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
    )
    monkeypatch.setattr("ragas_qa_dataset.cli.load_settings", lambda _: invalid_settings)

    exit_code = main(["validate"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Unsupported output formats configured" in out


def test_generate_command_runs_pipeline_and_exports(monkeypatch, tmp_path: Path, capsys) -> None:
    settings = Settings(
        input_dir=tmp_path,
        file_types=("txt",),
        testset_size=1,
        output_formats=("xlsx",),
        azure_openai_api_key="azure-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
    )

    monkeypatch.setattr("ragas_qa_dataset.cli.load_settings", lambda _: settings)
    monkeypatch.setattr("ragas_qa_dataset.cli.load_local_documents", lambda **_: [object()])
    monkeypatch.setattr("ragas_qa_dataset.cli.preprocess_documents", lambda **_: [object()])
    monkeypatch.setattr(
        "ragas_qa_dataset.cli.generate_testset_from_prepared_documents",
        lambda **_: GeneratedTestset(
            samples=[
                GeneratedSample(
                    question="Q",
                    answer="Eine lange Antwort",
                    source_doc="doc.txt",
                    chunk_id="doc.txt:0",
                    question_type="factual",
                    difficulty="easy",
                    language="de",
                    context="Kontext",
                )
            ],
            distribution_preset="balanced",
            provider="azure_openai",
            mode="fast",
        ),
    )
    monkeypatch.setattr("ragas_qa_dataset.cli.clean_dataset_records", lambda records, **_: records)
    monkeypatch.setattr("ragas_qa_dataset.cli.export_xlsx", lambda records, path: path)

    exit_code = main(["generate", "--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Export abgeschlossen" in out
