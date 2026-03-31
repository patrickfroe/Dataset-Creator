from ragas_qa_dataset.loaders import Document
from ragas_qa_dataset.preprocess import chunk_text, normalize_text, preprocess_documents


def test_normalize_text_compacts_whitespace() -> None:
    assert normalize_text("  a\n\n b\t c  ") == "a b c"


def test_chunk_text_creates_multiple_chunks() -> None:
    chunks = chunk_text("eins zwei drei vier", chunk_size=10, chunk_overlap=0)
    assert chunks == [(0, "eins zwei"), (1, "drei vier")]


def test_chunk_text_applies_overlap() -> None:
    chunks = chunk_text("eins zwei drei vier", chunk_size=9, chunk_overlap=3)
    assert chunks[0] == (0, "eins zwei")
    assert chunks[1][1].startswith("zwei")


def test_chunk_text_does_not_split_words_at_boundaries() -> None:
    text = "alpha beta gamma delta epsilon"
    chunks = chunk_text(text, chunk_size=12, chunk_overlap=4)
    assert all("-" not in chunk for _, chunk in chunks)
    assert chunks[0][1].endswith("beta")
    assert chunks[1][1].startswith("beta")


def test_preprocess_documents_skips_empty_contents() -> None:
    docs = [
        Document(text="   \n\t  ", source_doc="empty.txt", file_path="/tmp/empty.txt", file_type="txt"),
        Document(text="content", source_doc="ok.txt", file_path="/tmp/ok.txt", file_type="txt"),
    ]

    chunks = preprocess_documents(docs, chunk_size=50, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].source_doc == "ok.txt"


def test_preprocess_documents_preserves_metadata_and_sets_chunk_id() -> None:
    docs = [
        Document(
            text="abcdef",
            source_doc="doc.md",
            file_path="/data/doc.md",
            file_type="md",
        )
    ]

    chunks = preprocess_documents(
        docs,
        chunk_size=3,
        chunk_overlap=0,
        include_source_excerpt=True,
        excerpt_chars=4,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["doc.md:0", "doc.md:1"]
    assert all(chunk.source_doc == "doc.md" for chunk in chunks)
    assert all(chunk.file_path == "/data/doc.md" for chunk in chunks)
    assert all(chunk.file_type == "md" for chunk in chunks)
    assert all(chunk.source_excerpt == "abcd" for chunk in chunks)
