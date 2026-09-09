"""Section-aware chunking for the Sharia standards corpus.

The corpus is written as numbered clauses (``§3.2``) grouped under headed
sections (``## §3 Deposit products``). That structure is the whole point: a
citation must resolve to a real clause, not to "somewhere in SFS-001". So the
chunker splits on clause boundaries and never merges across a top-level section,
which keeps every chunk attributable to a contiguous clause range.

Clauses are small (40-80 words), which retrieves poorly in isolation, so
adjacent clauses within one section are packed together up to a target size.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Matches a clause opener such as "§3.2 " at the start of a line.
CLAUSE_RE = re.compile(r"^§(\d+)\.(\d+)\s+", re.MULTILINE)

# Matches a section heading such as "## §3 Deposit products".
HEADING_RE = re.compile(r"^##\s+§(\d+)\s+(.+?)\s*$", re.MULTILINE)

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# A stable namespace so chunk UUIDs are reproducible across ingests.
CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

TARGET_WORDS = 180
MAX_WORDS = 320


@dataclass(frozen=True)
class Document:
    """A parsed corpus document."""

    doc_id: str
    title: str
    topic: str
    provenance: str
    body: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable, citable span of one document."""

    chunk_id: str
    doc_id: str
    title: str
    topic: str
    heading: str
    sections: list[str] = field(default_factory=list)
    text: str = ""
    breadcrumb: str = ""
    lang: str = "en"
    citation_override: str = ""

    @property
    def section_label(self) -> str:
        """Human-readable clause range, e.g. ``3.1-3.3`` or ``3.2``."""
        if not self.sections:
            return ""
        if len(self.sections) == 1:
            return self.sections[0]
        return f"{self.sections[0]}-{self.sections[-1]}"

    @property
    def citation(self) -> str:
        """Citation string a reviewer can look up, e.g. ``SFS-001 §3.1-3.3``."""
        if self.citation_override:
            return self.citation_override
        label = self.section_label
        return f"{self.doc_id} §{label}" if label else self.doc_id

    def point_id(self) -> str:
        """Deterministic UUID for this chunk.

        Qdrant point IDs must be an unsigned integer or a UUID, and deriving one
        from ``chunk_id`` makes ingestion idempotent: re-running upserts in place
        rather than accumulating duplicates.
        """
        return str(uuid.uuid5(CHUNK_NAMESPACE, self.chunk_id))

    def embedding_text(self) -> str:
        """Text sent to the embedding model.

        A chunk must carry its own context. Standards clauses are elliptical --
        "It is permissible for the Institution to decline" is near-identical to
        dozens of others until its lineage is attached -- so a breadcrumb is
        prepended when one is available, falling back to title and heading.
        """
        if self.breadcrumb:
            return f"{self.breadcrumb}\n\n{self.text}"
        prefix = f"{self.title} — {self.heading}" if self.heading else self.title
        return f"{prefix}\n\n{self.text}"


def parse_document(path: Path) -> Document:
    """Parse a corpus markdown file with YAML frontmatter."""
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{path.name}: missing YAML frontmatter delimited by ---")

    meta = yaml.safe_load(match.group(1)) or {}
    for key in ("doc_id", "title", "topic", "provenance"):
        if not meta.get(key):
            raise ValueError(f"{path.name}: frontmatter missing required key {key!r}")

    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        topic=str(meta["topic"]),
        provenance=str(meta["provenance"]).strip(),
        body=raw[match.end() :],
        source_path=path.name,
    )


def _sections(body: str) -> list[tuple[str, str]]:
    """Split a document body into ``(heading, section_text)`` pairs."""
    headings = list(HEADING_RE.finditer(body))
    if not headings:
        return [("", body)]

    out: list[tuple[str, str]] = []
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        heading = f"§{match.group(1)} {match.group(2)}"
        out.append((heading, body[match.end() : end]))
    return out


def _clauses(section_text: str) -> list[tuple[str, str]]:
    """Split section text into ``(clause_number, clause_text)`` pairs."""
    matches = list(CLAUSE_RE.finditer(section_text))
    if not matches:
        return []

    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        number = f"{match.group(1)}.{match.group(2)}"
        text = section_text[match.end() : end].strip()
        text = re.sub(r"\s*\n\s*", " ", text)
        if text:
            out.append((number, text))
    return out


def chunk_document(doc: Document) -> list[Chunk]:
    """Chunk one document into citable spans.

    Clauses are packed greedily up to ``TARGET_WORDS``, never crossing a
    top-level section boundary. An oversized single clause becomes its own chunk
    rather than being split mid-sentence, so a citation always covers whole
    clauses.
    """
    chunks: list[Chunk] = []

    for heading, section_text in _sections(doc.body):
        buffer: list[tuple[str, str]] = []
        words = 0

        def flush() -> None:
            nonlocal buffer, words
            if not buffer:
                return
            numbers = [n for n, _ in buffer]
            text = " ".join(f"§{n} {t}" for n, t in buffer)
            chunk_id = f"{doc.doc_id}#{numbers[0]}" + (
                f"-{numbers[-1]}" if len(numbers) > 1 else ""
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    topic=doc.topic,
                    heading=heading,
                    sections=numbers,
                    text=text,
                )
            )
            buffer = []
            words = 0

        for number, text in _clauses(section_text):
            clause_words = len(text.split())
            if buffer and words + clause_words > TARGET_WORDS:
                flush()
            buffer.append((number, text))
            words += clause_words
            if words >= MAX_WORDS:
                flush()

        flush()

    return chunks


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every markdown document in ``corpus_dir``, sorted by filename."""
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md documents found in {corpus_dir}")
    return [parse_document(p) for p in paths]


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk a list of already-loaded documents."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    """Load and chunk a directory of markdown sources."""
    return chunk_documents(load_corpus(corpus_dir))
