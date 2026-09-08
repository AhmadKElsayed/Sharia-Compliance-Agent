"""Extractor for the AAOIFI Shari'ah Standards PDF.

Three problems have to be solved before this document is usable as a corpus.

**Duplicated text.** The PDF fakes bold by drawing every span twice at the same
coordinates. Naive extraction yields "Hamish JiddiyyahHamish Jiddiyyah" and
doubles every chunk. Span-level analysis showed 66 spans per page of which 33
were unique, so the fix is exact rather than heuristic: drop any span already
drawn at the same position.

**Hierarchy.** Standards number their rules ``2/1/1``, nested under headings
numbered ``2/1``, nested under sections numbered ``2.``. A rule read alone is
close to meaningless — "It is permissible for the Institution to decline" needs
its ancestors to be interpretable. The extractor rebuilds that tree so each
clause can carry its lineage into the embedding.

**Non-normative material.** Every standard ends with Appendix (A), a drafting
history, and Appendix (B), the juristic reasoning, plus adoption notes. These
are not operative rules, and indexing them alongside the rules pollutes
retrieval. They are extracted but flagged, and excluded by default.

Note on dependencies: this module imports PyMuPDF, which is AGPL. It runs only
at ingest time and is not imported by the API, so it is pinned in
``requirements-ingest.txt`` rather than in the deployed runtime.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.rag.chunking import Chunk

# "Shari'ah Standard No. (8): Murabahah" — the running header names the standard
# on every page, which is how a page is attributed to its standard.
#
# The period after "No" is optional: standards 42-44 and 46-48 are headed
# "No (44)" while the rest use "No. (8)". Requiring it made those six invisible,
# and their clauses silently inherited the preceding standard's number — so
# clauses from "Obtaining and Deploying Liquidity" were cited as "Islamic
# Reinsurance". Citation correctness depends on this one character.
STANDARD_RE = re.compile(r"Shari.ah Standard No\.?\s*\((\d+)\)\s*:\s*(.+?)\s*$")

# "2. Procedures Prior to the Contract of Murabahah"
SECTION_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")

# "2/1 The customer's expression..." or "2/1/1 The Institution may purchase..."
NUMBERED_RE = re.compile(r"^(\d+(?:/\d+){1,3})\s+(\S.*)$")

# Table-of-contents rows use dot leaders: "General rulings ......... 345".
DOT_LEADER_RE = re.compile(r"\.{4,}")

# A clause states a rule and so opens with a capital or a quote. A numbered line
# whose body starts lowercase or with punctuation is a cross-reference that
# happened to wrap onto a new line -- "as stated in item\n5/6." -- and parsing it
# as a clause would create an empty rule and orphan the real text.
CLAUSE_OPENER_RE = re.compile(r'^[A-Z“"(\[]')

MIN_CLAUSE_WORDS = 5

BODY_START = "Statement of the Standard"
NON_NORMATIVE_MARKERS = (
    "Appendix (A)",
    "Appendix (B)",
    "Adoption of the Standard",
)


@dataclass
class Clause:
    """One numbered rule, with the lineage needed to interpret it."""

    standard_no: int
    standard_title: str
    number: str
    text: str
    section: str = ""
    subsection: str = ""
    normative: bool = True
    page: int = 0

    @property
    def clause_id(self) -> str:
        """Stable identifier, e.g. ``SS-08/2/1/1``."""
        return f"SS-{self.standard_no:02d}/{self.number}"

    @property
    def citation(self) -> str:
        """What a reviewer looks up, e.g. ``Shari'ah Standard No. 8, clause 2/1/1``."""
        return f"Shari'ah Standard No. {self.standard_no} ({self.standard_title}), clause {self.number}"

    def breadcrumb(self) -> str:
        """The contextual prefix prepended before embedding.

        AAOIFI clauses are elliptical by design; without this lineage many embed
        almost identically to one another.
        """
        parts = [f"Shari'ah Standard No.({self.standard_no}) {self.standard_title}"]
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(self.subsection)
        parts.append(f"clause {self.number}")
        return " > ".join(parts)

    def embedding_text(self) -> str:
        return f"{self.breadcrumb()}\n\n{self.text}"

    def depth(self) -> int:
        return self.number.count("/") + 1

    def to_chunk(self) -> "Chunk":
        """Adapt to the pipeline's chunk type, carrying the breadcrumb prefix."""
        return Chunk(
            chunk_id=self.clause_id,
            doc_id=f"SS-{self.standard_no:02d}",
            title=self.standard_title,
            topic=self.section,
            heading=self.subsection or self.section,
            sections=[self.number],
            text=self.text,
            breadcrumb=self.breadcrumb(),
            citation_override=self.citation,
            lang="en",
        )


@dataclass
class Standard:
    """One standard and its clauses."""

    number: int
    title: str
    clauses: list[Clause] = field(default_factory=list)

    def heading_numbers(self) -> set[str]:
        """Numbers that label a group of clauses rather than stating a rule.

        ``2/1`` is a heading when ``2/1/1`` exists beneath it; its text is a
        title such as "The customer's expression of his wish", which is not a
        rule and should not be retrievable on its own. The same number in a
        standard with no deeper nesting *is* the rule, so this is decided per
        standard rather than by depth.
        """
        numbers = {c.number for c in self.clauses}
        return {n for n in numbers if any(o.startswith(f"{n}/") for o in numbers)}

    @property
    def normative_clauses(self) -> list[Clause]:
        """Operative rules: normative, and not merely group headings."""
        headings = self.heading_numbers()
        return [
            c
            for c in self.clauses
            if c.normative
            and c.number not in headings
            # A rule shorter than this is a parse fragment, not a rule.
            and len(c.text.split()) >= MIN_CLAUSE_WORDS
        ]


def page_lines(page: pymupdf.Page) -> list[str]:
    """Extract a page's text with duplicate draws removed.

    Spans are keyed on position and content; a span already drawn at that
    position is the fake-bold second pass and is discarded. Remaining spans are
    grouped into rows by vertical position and ordered horizontally.
    """
    seen: set[tuple[float, float, str]] = set()
    rows: dict[float, list[tuple[float, str]]] = defaultdict(list)

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"]
                if not text.strip():
                    continue
                x = round(span["bbox"][0], 1)
                y = round(span["bbox"][1], 1)
                key = (x, y, text)
                if key in seen:
                    continue
                seen.add(key)
                rows[round(y)].append((x, text))

    lines = []
    for y in sorted(rows):
        joined = "".join(text for _, text in sorted(rows[y])).strip()
        if joined:
            lines.append(joined)
    return lines


def _dehyphenate(parts: list[str]) -> str:
    """Join wrapped lines, repairing words split across a line break.

    ``in-`` followed by ``vitation`` is one word. A trailing hyphen before a
    lowercase continuation is treated as hyphenation; anything else keeps its
    hyphen, so "Shari'ah-" before a capitalised word survives intact.
    """
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out.endswith("-") and part[:1].islower():
            out = out[:-1] + part
        elif out:
            out = f"{out} {part}"
        else:
            out = part
    return re.sub(r"\s{2,}", " ", out).strip()


def extract_standards(pdf_path: Path) -> list[Standard]:
    """Parse the standards volume into structured clauses."""
    doc = pymupdf.open(str(pdf_path))

    standards: dict[int, Standard] = {}
    current_no: int | None = None
    current_title = ""
    section = ""
    subsection = ""
    normative = False

    pending: Clause | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal pending, buffer
        if pending is not None:
            pending.text = _dehyphenate([pending.text, *buffer])
            if pending.text:
                standards[pending.standard_no].clauses.append(pending)
        pending = None
        buffer = []

    for page_no in range(doc.page_count):
        lines = page_lines(doc[page_no])
        if not lines:
            continue

        # The running header identifies the standard and is then discarded.
        # It is not reliably the first line -- on some pages other furniture
        # precedes it -- and missing it silently attributes the page's clauses
        # to the previous standard, yielding citations to the wrong document.
        header = None
        header_idx = -1
        for idx, candidate in enumerate(lines[:4]):
            found = STANDARD_RE.search(candidate)
            if found:
                header, header_idx = found, idx
                break
        if header:
            no = int(header.group(1))
            title = header.group(2).strip()
            if no != current_no:
                flush()
                current_no, current_title = no, title
                section = subsection = ""
                normative = False
            standards.setdefault(no, Standard(number=no, title=title))
            lines = lines[:header_idx] + lines[header_idx + 1 :]

        if current_no is None:
            continue

        for raw in lines:
            line = raw.strip()
            if not line or line.isdigit():
                continue  # page number
            if DOT_LEADER_RE.search(line):
                continue  # table-of-contents row
            if STANDARD_RE.search(line):
                continue  # repeated header mid-page

            if line.startswith(BODY_START):
                flush()
                normative = True
                continue

            if any(line.startswith(m) for m in NON_NORMATIVE_MARKERS):
                flush()
                normative = False
                continue

            section_match = SECTION_RE.match(line)
            if section_match and len(section_match.group(2)) > 3:
                flush()
                section = line
                subsection = ""
                continue

            numbered = NUMBERED_RE.match(line)
            if numbered and not CLAUSE_OPENER_RE.match(numbered.group(2)):
                # A wrapped cross-reference, not a new clause: keep it as text.
                if pending is not None:
                    buffer.append(line)
                continue

            if numbered:
                flush()
                number, text = numbered.group(1), numbered.group(2)
                depth = number.count("/") + 1
                if depth == 2:
                    # A two-level item is a heading when clauses nest beneath it,
                    # but in some standards it is itself the rule. It is recorded
                    # as a clause either way, and also retained as the subsection
                    # label so any deeper clauses inherit the context.
                    subsection = f"{number} {text}"
                pending = Clause(
                    standard_no=current_no,
                    standard_title=current_title,
                    number=number,
                    text=text,
                    section=section,
                    subsection=subsection if depth > 2 else "",
                    normative=normative,
                    page=page_no,
                )
                continue

            if pending is not None:
                buffer.append(line)
            elif subsection and section:
                # A continuation of a heading that wrapped onto a second line.
                subsection = _dehyphenate([subsection, line])

    flush()
    doc.close()
    return [standards[k] for k in sorted(standards)]


def extract_chunks(pdf_path: Path, normative_only: bool = True) -> list[Chunk]:
    """Extract the volume as pipeline chunks, ready to embed."""
    return [c.to_chunk() for c in extract_clauses(pdf_path, normative_only)]


def extract_clauses(pdf_path: Path, normative_only: bool = True) -> list[Clause]:
    """Flatten the volume into clauses, excluding appendices by default.

    Clause numbers are made unique. Where extraction misses a clause opener the
    following rule inherits its predecessor's number; suffixing the repeat keeps
    it retrievable rather than letting an idempotent upsert overwrite the first.
    """
    clauses: list[Clause] = []
    for standard in extract_standards(pdf_path):
        selected = (
            standard.normative_clauses if normative_only else standard.clauses
        )
        seen: dict[str, int] = {}
        for clause in selected:
            count = seen.get(clause.number, 0) + 1
            seen[clause.number] = count
            if count > 1:
                clause.number = f"{clause.number}({count})"
            clauses.append(clause)
    return clauses
