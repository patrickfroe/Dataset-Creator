from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Literal

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
    mode: str


@dataclass(slots=True)
class ControlledGraphData:
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


GenerationMode = Literal["fast", "controlled"]


def initialize_azure_openai_provider(
    api_key: str,
    endpoint: str,
    api_version: str,
    chat_deployment: str,
    embedding_deployment: str,
) -> ProviderBundle:
    """Initialize LLM + embeddings for the Azure OpenAI provider path."""
    if not api_key.strip():
        raise GenerationError("AZURE_OPENAI_API_KEY is required for generation.")
    if not endpoint.strip():
        raise GenerationError("AZURE_OPENAI_ENDPOINT is required for generation.")

    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise GenerationError(
            "Missing dependency 'langchain-openai'. Install it to use Azure OpenAI generation."
        ) from exc

    llm = AzureChatOpenAI(
        azure_deployment=chat_deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        temperature=0,
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return ProviderBundle(provider="azure_openai", llm=llm, embeddings=embeddings)


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


def _generate_samples_from_texts(
    texts: list[tuple[str, str, str]],
    testset_size: int,
    distribution_preset: str,
    language: str,
    llm: Any,
) -> list[GeneratedSample]:
    if testset_size <= 0:
        raise GenerationError("testset_size must be > 0.")

    distribution_profile = _distribution_for_preset(distribution_preset)
    if not texts:
        return []

    profile_cycle = cycle(distribution_profile)
    samples: list[GeneratedSample] = []

    for index in range(testset_size):
        chunk_id, source_doc, text = texts[index % len(texts)]
        question_type, difficulty = next(profile_cycle)
        qa = _generate_qa_with_llm(
            llm=llm,
            chunk_text=text,
            language=language,
            question_type=question_type,
            difficulty=difficulty,
            sample_number=index + 1,
        )
        samples.append(
            GeneratedSample(
                question=qa["question"],
                answer=qa["answer"],
                source_doc=source_doc,
                chunk_id=chunk_id,
                question_type=question_type,
                difficulty=difficulty,
                language=language,
                context=text,
            )
        )

    return samples


def _generate_mvp_samples(
    chunks: list[ProcessedChunk],
    testset_size: int,
    distribution_preset: str,
    language: str,
    llm: Any,
) -> list[GeneratedSample]:
    texts = [(chunk.chunk_id, chunk.source_doc, chunk.text) for chunk in chunks]
    return _generate_samples_from_texts(
        texts=texts,
        testset_size=testset_size,
        distribution_preset=distribution_preset,
        language=language,
        llm=llm,
    )


def _chunks_to_graph_data(chunks: list[ProcessedChunk]) -> ControlledGraphData:
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        nodes.append(
            {
                "id": chunk.chunk_id,
                "type": "DOCUMENT",
                "properties": {
                    "page_content": chunk.text,
                    "source_doc": chunk.source_doc,
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "file_type": chunk.file_type,
                },
            }
        )
        if index > 0:
            relationships.append(
                {
                    "source": chunks[index - 1].chunk_id,
                    "target": chunk.chunk_id,
                    "type": "NEXT_CHUNK",
                }
            )

    return ControlledGraphData(nodes=nodes, relationships=relationships)


def _apply_ragas_default_transforms(graph_data: ControlledGraphData, provider: ProviderBundle) -> None:
    """Try applying ragas default_transforms on a KnowledgeGraph; fail with clear message if unavailable."""
    try:
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
        from ragas.testset.transforms import apply_transforms, default_transforms
    except ImportError as exc:
        raise GenerationError(
            "Controlled mode requires 'ragas' with testset graph support. "
            "Install/upgrade ragas to use KnowledgeGraph transforms."
        ) from exc

    ragas_nodes = []
    for node in graph_data.nodes:
        ragas_nodes.append(
            Node(
                type=NodeType[node["type"]],
                properties=node["properties"],
            )
        )

    ragas_relationships = []
    for rel in graph_data.relationships:
        source_idx = _node_index_for_id(graph_data.nodes, rel["source"])
        target_idx = _node_index_for_id(graph_data.nodes, rel["target"])
        ragas_relationships.append(
            Relationship(
                source=ragas_nodes[source_idx],
                target=ragas_nodes[target_idx],
                type=rel["type"],
                properties={},
            )
        )

    kg = KnowledgeGraph(nodes=ragas_nodes, relationships=ragas_relationships)
    transforms = default_transforms(documents=[], llm=provider.llm, embedding_model=provider.embeddings)
    apply_transforms(kg, transforms)


def _node_index_for_id(nodes: list[dict[str, Any]], node_id: str) -> int:
    for index, node in enumerate(nodes):
        if node.get("id") == node_id:
            return index
    raise GenerationError(f"Unknown graph node id '{node_id}' in relationship.")


def _save_graph(graph_data: ControlledGraphData, graph_path: Path) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"nodes": graph_data.nodes, "relationships": graph_data.relationships}
    graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_graph(graph_path: Path) -> ControlledGraphData:
    if not graph_path.exists():
        raise GenerationError(f"Graph file not found: {graph_path}")

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenerationError("Invalid graph file: top-level JSON object expected.")

    nodes = payload.get("nodes")
    relationships = payload.get("relationships", [])
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        raise GenerationError("Invalid graph file: 'nodes' and 'relationships' must be lists.")

    return ControlledGraphData(nodes=nodes, relationships=relationships)


def _extract_texts_from_graph(graph_data: ControlledGraphData) -> list[tuple[str, str, str]]:
    texts: list[tuple[str, str, str]] = []
    for node in graph_data.nodes:
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            continue
        chunk_text = str(properties.get("page_content", "")).strip()
        if not chunk_text:
            continue
        chunk_id = str(properties.get("chunk_id", node.get("id", "unknown")))
        source_doc = str(properties.get("source_doc", "unknown"))
        texts.append((chunk_id, source_doc, chunk_text))
    return texts


def _generate_controlled_samples(
    chunks: list[ProcessedChunk],
    testset_size: int,
    distribution_preset: str,
    language: str,
    provider_bundle: ProviderBundle,
    graph_path: Path | None,
    load_graph: bool,
    save_graph: bool,
) -> list[GeneratedSample]:
    if load_graph:
        if graph_path is None:
            raise GenerationError("controlled mode: graph_path is required when load_graph=True.")
        graph_data = _load_graph(graph_path)
    else:
        graph_data = _chunks_to_graph_data(chunks)
        _apply_ragas_default_transforms(graph_data, provider_bundle)

    if save_graph and graph_path is not None:
        _save_graph(graph_data, graph_path)

    texts = _extract_texts_from_graph(graph_data)
    return _generate_samples_from_texts(
        texts=texts,
        testset_size=testset_size,
        distribution_preset=distribution_preset,
        language=language,
        llm=provider_bundle.llm,
    )


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return " ".join(part.strip() for part in parts if part.strip())
    return str(response)


def _generate_qa_with_llm(
    llm: Any,
    chunk_text: str,
    language: str,
    question_type: str,
    difficulty: str,
    sample_number: int,
) -> dict[str, str]:
    prompt = (
        "Erzeuge genau ein Frage/Antwort-Paar aus dem folgenden Dokumentauszug.\n"
        f"Sprache: {language}\n"
        f"Fragetyp: {question_type}\n"
        f"Schwierigkeit: {difficulty}\n"
        "Antwort nur im JSON-Format mit den Schlüsseln 'question' und 'answer'.\n"
        "Die Antwort muss im Auszug begründet sein und darf nichts erfinden.\n"
        f"Auszug:\n{chunk_text}\n"
        f"Sample-Nummer: {sample_number}"
    )
    response_text = _extract_response_text(llm.invoke(prompt)).strip()
    if response_text.startswith("```"):
        response_text = response_text.strip("`")
        response_text = response_text.replace("json\n", "", 1).strip()
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise GenerationError("LLM response is not valid JSON for QA generation.") from exc

    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    if not question or not answer:
        raise GenerationError("LLM response must contain non-empty 'question' and 'answer'.")
    return {"question": question, "answer": answer}


def generate_testset_from_prepared_documents(
    chunks: list[ProcessedChunk],
    testset_size: int,
    distribution_preset: str,
    language: str,
    azure_openai_api_key: str,
    azure_openai_endpoint: str,
    azure_openai_api_version: str,
    azure_openai_chat_deployment: str,
    azure_openai_embedding_deployment: str,
    mode: GenerationMode = "fast",
    graph_path: str | Path | None = None,
    load_graph: bool = False,
    save_graph: bool = False,
) -> GeneratedTestset:
    """Generate a testset from preprocessed chunks.

    - fast: MVP chunk-based generation (stable default)
    - controlled: builds/loads a Ragas-oriented KnowledgeGraph, applies default_transforms,
      then generates from graph nodes.
    """
    mode_normalized = mode.strip().lower()
    graph_file = Path(graph_path) if graph_path is not None else None
    provider_bundle = initialize_azure_openai_provider(
        api_key=azure_openai_api_key,
        endpoint=azure_openai_endpoint,
        api_version=azure_openai_api_version,
        chat_deployment=azure_openai_chat_deployment,
        embedding_deployment=azure_openai_embedding_deployment,
    )

    if mode_normalized == "fast":
        provider_name = provider_bundle.provider
        samples = _generate_mvp_samples(
            chunks=chunks,
            testset_size=testset_size,
            distribution_preset=distribution_preset,
            language=language,
            llm=provider_bundle.llm,
        )
    elif mode_normalized == "controlled":
        provider_name = provider_bundle.provider
        samples = _generate_controlled_samples(
            chunks=chunks,
            testset_size=testset_size,
            distribution_preset=distribution_preset,
            language=language,
            provider_bundle=provider_bundle,
            graph_path=graph_file,
            load_graph=load_graph,
            save_graph=save_graph,
        )
    else:
        raise GenerationError("Unknown mode. Supported values: fast, controlled.")

    return GeneratedTestset(
        samples=samples,
        distribution_preset=distribution_preset,
        provider=provider_name,
        mode=mode_normalized,
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
