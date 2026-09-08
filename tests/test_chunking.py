"""Tests for section-aware chunking.

These guard the property the citation pipeline depends on: every chunk must be
attributable to a contiguous clause range within a single document section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import (
    MAX_WORDS,
    Chunk,
    chunk_corpus,
    chunk_document,
    load_corpus,
    parse_document,
)

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_corpus(CORPUS_DIR)


def test_corpus_loads_and_every_doc_declares_provenance() -> None:
    docs = load_corpus(CORPUS_DIR)
    assert len(docs) >= 5, "task requires 3-5+ documents"
    for doc in docs:
        assert doc.doc_id.startswith("SFS-")
        assert "SYNTHESIZED FOR DEMONSTRATION" in doc.provenance, (
            f"{doc.doc_id} must declare that it is not authentic AAOIFI text"
        )


def test_frontmatter_is_required() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        parse_document(Path(__file__))  # a .py file has no frontmatter


def test_chunks_are_produced_for_every_document(chunks: list[Chunk]) -> None:
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {f"SFS-{n:03d}" for n in range(1, 7)}


def test_chunk_ids_are_unique(chunks: list[Chunk]) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_point_ids_are_unique_and_deterministic(chunks: list[Chunk]) -> None:
    point_ids = [c.point_id() for c in chunks]
    assert len(point_ids) == len(set(point_ids))
    # Re-chunking must yield identical IDs, or ingestion stops being idempotent.
    again = {c.chunk_id: c.point_id() for c in chunk_corpus(CORPUS_DIR)}
    assert again == {c.chunk_id: c.point_id() for c in chunks}


def test_no_chunk_exceeds_the_size_ceiling(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        # A single oversized clause is allowed through whole rather than split
        # mid-sentence, so the ceiling is checked against multi-clause chunks.
        if len(chunk.sections) > 1:
            assert len(chunk.text.split()) <= MAX_WORDS + 60, chunk.chunk_id


def test_chunks_never_span_two_top_level_sections(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        tops = {s.split(".")[0] for s in chunk.sections}
        assert len(tops) == 1, f"{chunk.chunk_id} spans sections {tops}"


def test_clause_numbers_within_a_chunk_are_contiguous(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        minors = [int(s.split(".")[1]) for s in chunk.sections]
        assert minors == sorted(minors)
        assert minors == list(range(minors[0], minors[0] + len(minors))), chunk.chunk_id


def test_citation_format_is_lookup_ready(chunks: list[Chunk]) -> None:
    single = next(c for c in chunks if len(c.sections) == 1)
    assert single.citation == f"{single.doc_id} §{single.sections[0]}"

    multi = next(c for c in chunks if len(c.sections) > 1)
    assert multi.citation == f"{multi.doc_id} §{multi.sections[0]}-{multi.sections[-1]}"


def test_embedding_text_carries_document_context(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        text = chunk.embedding_text()
        assert chunk.title in text
        if chunk.heading:
            assert chunk.heading in text
        assert chunk.text in text


def test_clause_markers_are_preserved_in_text(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        for section in chunk.sections:
            assert f"§{section}" in chunk.text


def test_known_clause_survives_chunking_verbatim(chunks: list[Chunk]) -> None:
    """The riba clause the agent leans on most must be retrievable intact."""
    hits = [c for c in chunks if c.doc_id == "SFS-001" and "3.2" in c.sections]
    assert len(hits) == 1
    assert "guarantee of the principal" in hits[0].text


def test_document_with_no_clauses_yields_no_chunks(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text(
        "---\ndoc_id: X-1\ntitle: T\ntopic: t\nprovenance: p\n---\n\n"
        "## §1 Heading\n\nProse with no numbered clauses.\n",
        encoding="utf-8",
    )
    assert chunk_document(parse_document(path)) == []
