from pathlib import Path

import pytest

from ragas_qa_dataset.config import SettingsError, load_settings


def test_load_settings_uses_yaml_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                "input_dir: custom/input",
                "file_types:",
                "  - txt",
                "language: en",
                "max_docs: 5",
                "chunk_size: 300",
                "chunk_overlap: 30",
                "testset_size: 12",
                "distribution_preset: reasoning",
                "include_metadata: false",
                "include_source_excerpt: true",
                "output_formats:",
                "  - jsonl",
                "random_seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.input_dir == Path("custom/input")
    assert settings.file_types == ("txt",)
    assert settings.language == "en"
    assert settings.max_docs == 5
    assert settings.chunk_size == 300
    assert settings.chunk_overlap == 30
    assert settings.testset_size == 12
    assert settings.distribution_preset == "reasoning"
    assert settings.include_metadata is False
    assert settings.include_source_excerpt is True
    assert settings.output_formats == ("jsonl",)
    assert settings.random_seed == 7
    assert settings.openai_api_key == "sk-test"


def test_load_settings_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("chunk_size: 200\nopenai_api_key: yaml-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("RAGAS_CHUNK_SIZE", "512")
    monkeypatch.setenv("RAGAS_FILE_TYPES", "pdf,md")

    settings = load_settings(config_file)

    assert settings.openai_api_key == "env-key"
    assert settings.chunk_size == 512
    assert settings.file_types == ("pdf", "md")


def test_load_settings_raises_when_openai_key_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("language: de\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SettingsError, match="OPENAI_API_KEY fehlt"):
        load_settings(config_file)
