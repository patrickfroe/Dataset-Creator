from pathlib import Path

import pytest

from ragas_qa_dataset.generator import (
    GenerationError,
    ProviderBundle,
    generate_testset_from_prepared_documents,
)
from ragas_qa_dataset.preprocess import ProcessedChunk


class _LLMStub:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"question": "Welche Aussage ist korrekt?", "answer": "Die Aussage stammt aus dem Chunk."}'


def _provider_stub(*_args: object, **_kwargs: object) -> ProviderBundle:
    return ProviderBundle(provider="azure_openai", llm=_LLMStub(), embeddings=object())


def test_generate_testset_balanced_preset_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_azure_openai_provider", _provider_stub)

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
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_version="2024-10-21",
        azure_openai_chat_deployment="gpt-4o-mini",
        azure_openai_embedding_deployment="text-embedding-3-small",
    )

    assert result.provider == "azure_openai"
    assert result.mode == "fast"
    assert len(result.samples) == 3
    assert result.samples[0].question == "Welche Aussage ist korrekt?"


def test_generate_testset_controlled_mode_with_graph_save_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_azure_openai_provider", _provider_stub)
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
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_version="2024-10-21",
        azure_openai_chat_deployment="gpt-4o-mini",
        azure_openai_embedding_deployment="text-embedding-3-small",
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
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_version="2024-10-21",
        azure_openai_chat_deployment="gpt-4o-mini",
        azure_openai_embedding_deployment="text-embedding-3-small",
        mode="controlled",
        graph_path=graph_file,
        load_graph=True,
    )

    assert len(loaded.samples) == 2
    assert loaded.samples[0].source_doc == "doc1.md"


def test_generate_testset_rejects_unknown_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_azure_openai_provider", _provider_stub)

    with pytest.raises(GenerationError, match="Unknown distribution_preset"):
        generate_testset_from_prepared_documents(
            chunks=[],
            testset_size=2,
            distribution_preset="unknown",
            language="de",
            azure_openai_api_key="test-key",
            azure_openai_endpoint="https://example.openai.azure.com/",
            azure_openai_api_version="2024-10-21",
            azure_openai_chat_deployment="gpt-4o-mini",
            azure_openai_embedding_deployment="text-embedding-3-small",
        )


def test_generate_testset_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragas_qa_dataset.generator.initialize_azure_openai_provider", _provider_stub)

    with pytest.raises(GenerationError, match="Unknown mode"):
        generate_testset_from_prepared_documents(
            chunks=[],
            testset_size=2,
            distribution_preset="simple",
            language="de",
            azure_openai_api_key="test-key",
            azure_openai_endpoint="https://example.openai.azure.com/",
            azure_openai_api_version="2024-10-21",
            azure_openai_chat_deployment="gpt-4o-mini",
            azure_openai_embedding_deployment="text-embedding-3-small",
            mode="invalid",
        )


def test_generate_testset_raises_on_invalid_llm_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadLLM:
        def invoke(self, prompt: str) -> str:
            return "not-json"

    monkeypatch.setattr(
        "ragas_qa_dataset.generator.initialize_azure_openai_provider",
        lambda *_args, **_kwargs: ProviderBundle(provider="azure_openai", llm=_BadLLM(), embeddings=object()),
    )

    chunks = [
        ProcessedChunk(
            chunk_id="doc1:0",
            text="Fast mode should run with LLM output.",
            source_doc="doc1.md",
            file_path="/tmp/doc1.md",
            file_type="md",
        )
    ]

    with pytest.raises(GenerationError, match="not valid JSON"):
        generate_testset_from_prepared_documents(
            chunks=chunks,
            testset_size=1,
            distribution_preset="simple",
            language="de",
            azure_openai_api_key="test-key",
            azure_openai_endpoint="https://example.openai.azure.com/",
            azure_openai_api_version="2024-10-21",
            azure_openai_chat_deployment="gpt-4o-mini",
            azure_openai_embedding_deployment="text-embedding-3-small",
            mode="fast",
        )
