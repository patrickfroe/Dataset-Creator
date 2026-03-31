from pathlib import Path

from ragas_qa_dataset.loaders import load_local_documents


def test_load_local_documents_reads_txt_and_md(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello txt", encoding="utf-8")
    (tmp_path / "b.md").write_text("# hello md", encoding="utf-8")

    docs = load_local_documents(tmp_path, ("txt", "md"))

    assert len(docs) == 2
    assert docs[0].file_type == "txt"
    assert docs[0].source_doc == "a.txt"
    assert docs[1].file_type == "md"
    assert docs[1].source_doc == "b.md"


def test_load_local_documents_filters_file_types(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "b.md").write_text("skip", encoding="utf-8")

    docs = load_local_documents(tmp_path, ("txt",))

    assert len(docs) == 1
    assert docs[0].source_doc == "a.txt"
    assert docs[0].file_type == "txt"


def test_load_local_documents_respects_max_docs(tmp_path: Path) -> None:
    (tmp_path / "1.txt").write_text("one", encoding="utf-8")
    (tmp_path / "2.txt").write_text("two", encoding="utf-8")

    docs = load_local_documents(tmp_path, ("txt",), max_docs=1)

    assert len(docs) == 1
