"""Load corpus documents from PDF.

The corpus ships as PDFs because that is the form these documents take in
practice — a policy manual is circulated as a PDF, not as markdown. Extracting
from it means dealing with what PDF text extraction actually produces: running
headers and footers repeated on every page, a metadata table flattened into
alternating key and value lines, and clause text broken across line boundaries
mid-sentence.

This module normalises all of that back into the clause structure that
``chunking.py`` already understands, so the chunker, the citation format, and
their tests are unchanged by the switch from markdown to PDF.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.rag.chunking import Document

# "§3 Deposit products" — a section heading (no minor number).
PDF_HEADING_RE = re.compile(r"^§\s?(\d+)\s{1,4}(.+?)\s*$")

# "§3.2 A guaranteed return ..." — a numbered clause.
PDF_CLAUSE_RE = re.compile(r"^§\s?(\d+)\.(\d+)\s{1,4}(.*)$")

# Lines that are page furniture rather than content.
FOOTER_RE = re.compile(r"^Page\s+\d+$")
CONTROL_KEYS = {
    "Document reference",
    "Version",
    "Effective date",
    "Approved by",
    "Review cycle",
    "Classification",
}

ATTRIBUTION_HEADING = "Sources and attribution"


def _strip_furniture(lines: list[str], doc_id: str) -> list[str]:
    """Drop running headers, footers, and the title block."""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == doc_id or FOOTER_RE.match(stripped):
            continue
        if stripped.startswith("Internal Sharia Compliance Manual"):
            continue
        if "Internal use" in stripped and "Sharia Supervisory Board" in stripped:
            continue
        out.append(stripped)
    return out


def _read_control_table(lines: list[str]) -> dict[str, str]:
    """Recover the document control metadata.

    Table cells extract as alternating key and value lines, so a key is read by
    taking the line that follows it.

    This must run against the *raw* lines, before furniture stripping. The
    running header is the bare document reference, and removing it by value also
    removes the identical value cell inside this table, which silently shifts
    every key onto the following row's value.
    """
    meta: dict[str, str] = {}
    cleaned = [line.strip() for line in lines if line.strip()]
    for i, line in enumerate(cleaned):
        if line in CONTROL_KEYS and i + 1 < len(cleaned):
            value = cleaned[i + 1]
            if value not in CONTROL_KEYS:
                meta[line] = value
    return meta


def _normalise(lines: list[str]) -> tuple[str, str]:
    """Rebuild clause structure, returning ``(body, attribution)``.

    Continuation lines are joined onto the clause they belong to, which is what
    makes a citation quotable: a clause split across a page break must come back
    as one span of text.
    """
    body: list[str] = []
    attribution: list[str] = []
    current: list[str] | None = None
    in_attribution = False

    def flush() -> None:
        nonlocal current
        if current:
            body.append(" ".join(current))
            current = None

    for line in lines:
        if line.startswith(ATTRIBUTION_HEADING):
            flush()
            in_attribution = True
            continue

        if in_attribution:
            attribution.append(line)
            continue

        clause = PDF_CLAUSE_RE.match(line)
        if clause:
            flush()
            current = [f"§{clause.group(1)}.{clause.group(2)} {clause.group(3)}".rstrip()]
            continue

        heading = PDF_HEADING_RE.match(line)
        if heading:
            flush()
            body.append(f"## §{heading.group(1)} {heading.group(2)}")
            continue

        if current is not None:
            current.append(line)

    flush()
    return "\n\n".join(body), " ".join(attribution)


def load_pdf(path: Path) -> Document:
    """Parse one corpus PDF into a ``Document``."""
    reader = PdfReader(str(path))
    meta = reader.metadata or {}

    title_raw = str(meta.get("/Title", path.stem))
    doc_id = title_raw.split()[0] if title_raw else path.stem
    title = title_raw.split(" — ", 1)[1] if " — " in title_raw else title_raw

    lines: list[str] = []
    for page in reader.pages:
        lines.extend((page.extract_text() or "").splitlines())

    control = _read_control_table(lines)
    cleaned = _strip_furniture(lines, doc_id)
    body, attribution = _normalise(cleaned)

    if not body.strip():
        raise ValueError(f"{path.name}: no clauses extracted; is the PDF text-based?")

    return Document(
        doc_id=control.get("Document reference", doc_id),
        title=title,
        topic=path.stem.split("-", 2)[-1],
        provenance=attribution or "Internal guidance; see the source PDF.",
        body=body,
        source_path=path.name,
    )


def load_pdf_corpus(pdf_dir: Path) -> list[Document]:
    """Load every PDF in ``pdf_dir``, sorted by filename."""
    paths = sorted(pdf_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"no PDF documents found in {pdf_dir}")
    return [load_pdf(p) for p in paths]
