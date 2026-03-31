import pytest

from ragas_qa_dataset.generator import (
    GenerationError,
    ProviderBundle,
    generate_testset_from_prepared_documents,
)
from ragas_qa_dataset.preprocess import ProcessedChunk


def test_generate_testset_balanced_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragas_qa_dataset.generator.initialize_openai_provider",
        lambda api_key: ProviderBundle(provider="openai", llm=object(), embeddings=object()),
    )

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
    assert len(result.samples) == 3
    assert {sample.question_type for sample in result.samples} == {"factual", "reasoning"}


def test_generate_testset_rejects_unknown_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragas_qa_dataset.generator.initialize_openai_provider",
        lambda api_key: ProviderBundle(provider="openai", llm=object(), embeddings=object()),
    )

    with pytest.raises(GenerationError, match="Unknown distribution_preset"):
        generate_testset_from_prepared_documents(
            chunks=[],
            testset_size=2,
            distribution_preset="unknown",
            language="de",
            openai_api_key="sk-test",
        )
