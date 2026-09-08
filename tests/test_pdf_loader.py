"""Tests for PDF extraction.

The corpus ships as PDF, so extraction correctness is upstream of everything:
a clause lost or mangled here becomes a citation that cannot be verified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import chunk_document, chunk_documents
from app.rag.pdf_loader import load_pdf, load_pdf_corpus

PDF_DIR = Path(__file__).resolve().parents[1] / "corpus" / "demo"
SOURCE_DIR = Path(__file__).resolve().parents[1] / "corpus" / "source"


@pytest.fixture(scope="module")
def docs():
    return load_pdf_corpus(PDF_DIR)


def test_every_source_has_a_built_pdf() -> None:
    sources = {p.stem for p in SOURCE_DIR.glob("*.md")}
    pdfs = {p.stem for p in PDF_DIR.glob("*.pdf")}
    assert sources == pdfs, "run scripts/build_corpus_pdfs.py after editing sources"


def test_document_ids_come_from_the_control_table(docs) -> None:
    """Guards a real bug: the running header is the bare document reference,
    so stripping it by value also removed the identical cell inside the control
    table, shifting every key onto the next row's value."""
    assert [d.doc_id for d in docs] == [f"SFS-{n:03d}" for n in range(1, 7)]


def test_titles_are_extracted_without_the_id_prefix(docs) -> None:
    for doc in docs:
        assert doc.title
        assert not doc.title.startswith("SFS-")


def test_running_headers_and_footers_are_stripped(docs) -> None:
    for doc in docs:
        assert "Internal Sharia Compliance Manual" not in doc.body
        assert "Internal use" not in doc.body
        assert "Page 1" not in doc.body


def test_attribution_is_captured_as_provenance_not_content(docs) -> None:
    for doc in docs:
        assert "no authority" in doc.provenance.lower()
        # The attribution block must not be chunked as if it were a clause.
        assert "Sources and attribution" not in doc.body


def test_clauses_wrapped_across_lines_are_rejoined(docs) -> None:
    """A clause split across a page break must return as one span, or a quoted
    citation cannot be matched back to it."""
    riba = next(d for d in docs if d.doc_id == "SFS-001")
    for chunk in chunk_document(riba):
        for line in chunk.text.splitlines():
            assert line.strip(), "no blank fragments from line wrapping"
        # A rejoined clause reads as prose, not as short broken lines.
        assert "  " not in chunk.text


def test_section_headings_survive_extraction(docs) -> None:
    riba = next(d for d in docs if d.doc_id == "SFS-001")
    headings = {c.heading for c in chunk_document(riba)}
    assert any("Scope" in h for h in headings)
    assert any("Assessment criteria" in h for h in headings)


def test_clause_numbering_is_preserved(docs) -> None:
    riba = next(d for d in docs if d.doc_id == "SFS-001")
    sections = {s for c in chunk_document(riba) for s in c.sections}
    for expected in ("1.1", "3.2", "4.1", "7.1"):
        assert expected in sections, f"clause {expected} lost in extraction"


def test_every_document_yields_chunks(docs) -> None:
    for doc in docs:
        assert chunk_document(doc), f"{doc.doc_id} produced no chunks"


def test_corpus_size_is_reasonable(docs) -> None:
    chunks = chunk_documents(docs)
    assert len(chunks) >= 30
    words = [len(c.text.split()) for c in chunks]
    assert max(words) <= 400, "a chunk this large will crowd the context window"
    assert min(words) >= 10, "a chunk this small is unlikely to retrieve usefully"


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError, match="no PDF documents"):
        load_pdf_corpus(Path(__file__).parent / "nope")


def test_non_pdf_input_is_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "not.pdf"
    fake.write_bytes(b"this is not a pdf")
    with pytest.raises(Exception):
        load_pdf(fake)
