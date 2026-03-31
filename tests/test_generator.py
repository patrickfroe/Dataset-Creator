from pathlib import Path

import pytest

from ragas_qa_dataset.generator import (
    GenerationError,
    ProviderBundle,
    generate_testset_from_prepared_documents,
)
from ragas_qa_dataset.preprocess import ProcessedChunk


def _provider_stub(api_key: str) -> ProviderBundle:
    return ProviderBundle(provider="openai", llm=object(), embeddings=object())


def test_generate_testset_balanced_preset_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_openai_provider", _provider_stub)

    chunks = [
        ProcessedChunk(
            chunk_id="doc1:0",
            text="Ein kurzer Abschnitt über ein Thema.",
            source_doc="doc1.md",
            file_path="/tmp/doc1.md",
            file_type="md",
        )
    ]

    result = generate_testset_from_prepared_documents(
        chunks=chunks,
        testset_size=3,
        distribution_preset="balanced",
        language="de",
        openai_api_key="sk-test",
    )

    assert result.provider == "openai"
    assert result.mode == "fast"
    assert len(result.samples) == 3
    assert {sample.question_type for sample in result.samples} == {"factual", "reasoning"}


def test_generate_testset_controlled_mode_with_graph_save_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_openai_provider", _provider_stub)
    monkeypatch.setattr(
        "ragas_qa_dataset.generator._apply_ragas_default_transforms",
        lambda graph_data, provider: None,
    )

    chunks = [
        ProcessedChunk(
            chunk_id="doc1:0",
            text="Abschnitt A über Richtlinien.",
            source_doc="doc1.md",
            file_path="/tmp/doc1.md",
            file_type="md",
        ),
        ProcessedChunk(
            chunk_id="doc1:1",
            text="Abschnitt B über Ausnahmen.",
            source_doc="doc1.md",
            file_path="/tmp/doc1.md",
            file_type="md",
        ),
    ]
    graph_file = tmp_path / "kg.json"

    generated = generate_testset_from_prepared_documents(
        chunks=chunks,
        testset_size=2,
        distribution_preset="simple",
        language="de",
        openai_api_key="sk-test",
        mode="controlled",
        graph_path=graph_file,
        save_graph=True,
    )

    assert generated.mode == "controlled"
    assert graph_file.exists()

    loaded = generate_testset_from_prepared_documents(
        chunks=[],
        testset_size=2,
        distribution_preset="simple",
        language="de",
        openai_api_key="sk-test",
        mode="controlled",
        graph_path=graph_file,
        load_graph=True,
    )

    assert len(loaded.samples) == 2
    assert loaded.samples[0].source_doc == "doc1.md"


def test_generate_testset_rejects_unknown_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_openai_provider", _provider_stub)

    with pytest.raises(GenerationError, match="Unknown distribution_preset"):
        generate_testset_from_prepared_documents(
            chunks=[],
            testset_size=2,
            distribution_preset="unknown",
            language="de",
            openai_api_key="sk-test",
        )


def test_generate_testset_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_openai_provider", _provider_stub)

    with pytest.raises(GenerationError, match="Unknown mode"):
        generate_testset_from_prepared_documents(
            chunks=[],
            testset_size=2,
            distribution_preset="simple",
            language="de",
            openai_api_key="sk-test",
            mode="invalid",
        )


def test_generate_testset_fast_mode_does_not_initialize_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(api_key: str) -> ProviderBundle:
        raise AssertionError("Provider should not be initialized in fast mode")

    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_openai_provider", _boom)

    chunks = [
        ProcessedChunk(
            chunk_id="doc1:0",
            text="Fast mode should run without provider initialization.",
            source_doc="doc1.md",
            file_path="/tmp/doc1.md",
            file_type="md",
        )
    ]

    generated = generate_testset_from_prepared_documents(
        chunks=chunks,
        testset_size=1,
        distribution_preset="simple",
        language="de",
        openai_api_key="sk-test",
        mode="fast",
    )

    assert generated.provider == "openai"
    assert len(generated.samples) == 1
