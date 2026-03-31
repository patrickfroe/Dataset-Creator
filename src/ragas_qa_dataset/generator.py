from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any

from ragas_qa_dataset.preprocess import ProcessedChunk


class GenerationError(RuntimeError):
    """Raised when testset generation fails."""


@dataclass(slots=True)
class QAItem:
    question: str
    answer: str
    context: str
    source: str


@dataclass(slots=True)
class ProviderBundle:
    provider: str
    llm: Any
    embeddings: Any


@dataclass(slots=True)
class GeneratedSample:
    question: str
    answer: str
    source_doc: str
    chunk_id: str
    question_type: str
    difficulty: str
    language: str
    context: str


@dataclass(slots=True)
class GeneratedTestset:
    samples: list[GeneratedSample]
    distribution_preset: str
    provider: str


def initialize_openai_provider(
    api_key: str,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
) -> ProviderBundle:
    """Initialize LLM + embeddings for the OpenAI-based MVP path."""
    if not api_key.strip():
        raise GenerationError("OPENAI_API_KEY is required for generation.")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise GenerationError(
            "Missing dependency 'langchain-openai'. Install it to use OpenAI generation."
        ) from exc

    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    return ProviderBundle(provider="openai", llm=llm, embeddings=embeddings)


def _distribution_for_preset(name: str) -> list[tuple[str, str]]:
    preset = name.strip().lower()
    if preset == "simple":
        return [("factual", "easy")]
    if preset == "reasoning":
        return [("reasoning", "medium"), ("reasoning", "hard")]
    if preset == "balanced":
        # MVP standard assumptions:
        # - mostly direct factual questions for coverage
        # - plus moderate reasoning questions for depth
        return [
            ("factual", "easy"),
            ("factual", "medium"),
            ("reasoning", "medium"),
        ]
    raise GenerationError(
        f"Unknown distribution_preset '{name}'. Supported values: balanced, simple, reasoning."
    )


def _question_from_chunk(chunk: str, question_type: str, index: int) -> str:
    prefix = chunk[:90].rstrip(" ,.;:")
    if question_type == "reasoning":
        return f"Welche Schlussfolgerung lässt sich aus folgendem Abschnitt ziehen: '{prefix}'? (#{index + 1})"
    return f"Was steht im Dokument über: '{prefix}'? (#{index + 1})"


def _answer_from_chunk(chunk: str) -> str:
    return chunk[:300].strip()


def _generate_mvp_samples(
    chunks: list[ProcessedChunk],
    testset_size: int,
    distribution_preset: str,
    language: str,
) -> list[GeneratedSample]:
    if testset_size <= 0:
        raise GenerationError("testset_size must be > 0.")

    distribution_profile = _distribution_for_preset(distribution_preset)
    if not chunks:
        return []

    profile_cycle = cycle(distribution_profile)
    samples: list[GeneratedSample] = []

    for index in range(testset_size):
        chunk = chunks[index % len(chunks)]
        question_type, difficulty = next(profile_cycle)
        samples.append(
            GeneratedSample(
                question=_question_from_chunk(chunk.text, question_type, index),
                answer=_answer_from_chunk(chunk.text),
                source_doc=chunk.source_doc,
                chunk_id=chunk.chunk_id,
                question_type=question_type,
                difficulty=difficulty,
                language=language,
                context=chunk.text,
            )
        )

    return samples


def generate_testset_from_prepared_documents(
    chunks: list[ProcessedChunk],
    testset_size: int,
    distribution_preset: str,
    language: str,
    openai_api_key: str,
) -> GeneratedTestset:
    """Generate a testset from preprocessed chunks.

    Fast path first (document/chunk based) for MVP.
    TODO: Add KnowledgeGraph generation path for richer multi-hop synthesis.
    """
    provider_bundle = initialize_openai_provider(api_key=openai_api_key)

    # MVP: fast document-based path using preprocessed chunks.
    # The OpenAI provider is initialized and ready; generation is deterministic here
    # and can later be swapped with direct Ragas synthesizers.
    samples = _generate_mvp_samples(
        chunks=chunks,
        testset_size=testset_size,
        distribution_preset=distribution_preset,
        language=language,
    )

    return GeneratedTestset(
        samples=samples,
        distribution_preset=distribution_preset,
        provider=provider_bundle.provider,
    )


def generate_qa_items(source: str, chunk: str, max_questions_per_chunk: int) -> list[QAItem]:
    count = max(1, max_questions_per_chunk)
    items: list[QAItem] = []
    for index in range(count):
        items.append(
            QAItem(
                question=_question_from_chunk(chunk, "factual", index),
                answer=_answer_from_chunk(chunk),
                context=chunk,
                source=source,
            )
        )
    return items
