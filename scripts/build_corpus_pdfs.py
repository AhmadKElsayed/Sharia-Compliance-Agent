#!/usr/bin/env python
"""Render the corpus markdown sources into standards-style PDF documents.

The PDFs are the corpus the pipeline actually ingests. Markdown is kept as the
editable source so content changes stay reviewable in git; this script is the
build step between the two.

Layout deliberately mirrors a real internal policy manual: a title block with
document control metadata, numbered sections, numbered clauses, running
headers and footers, and a source-attribution block. That structure is what
makes the PDF parser's job realistic rather than trivial.

Usage:
    python scripts/build_corpus_pdfs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_DIR = Path(__file__).resolve().parents[1] / "corpus" / "source"
PDF_DIR = Path(__file__).resolve().parents[1] / "corpus" / "demo"

ISSUER = "Mal Islamic Bank — Sharia Supervisory Board"
MANUAL = "Internal Sharia Compliance Manual"

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(§\d+)\s+(.+?)\s*$")
CLAUSE_RE = re.compile(r"^(§\d+\.\d+)\s+(.*)$")

NAVY = colors.HexColor("#1a3a5c")
GREY = colors.HexColor("#5a6470")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, textColor=NAVY, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10.5, leading=14, textColor=GREY, spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "SectionHeading", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=NAVY, spaceBefore=13, spaceAfter=6,
        ),
        "clause": ParagraphStyle(
            "Clause", parent=base["BodyText"], fontName="Times-Roman",
            fontSize=10.2, leading=14.5, alignment=TA_JUSTIFY,
            spaceAfter=6, firstLineIndent=0, leftIndent=13 * mm,
            bulletIndent=0,
        ),
        "note": ParagraphStyle(
            "Note", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8.4, leading=11.5, textColor=GREY, alignment=TA_JUSTIFY,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.6, leading=11.5,
        ),
    }


def inline(text: str) -> str:
    """Convert the small markdown subset used in the sources to reportlab tags."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"\[([A-Z]{3}-\d{3}[^\]]*)\]",
        r'<font color="#1a3a5c"><b>[\1]</b></font>',
        text,
    )
    return text


def header_footer(canvas, doc, meta: dict) -> None:
    canvas.saveState()
    width, height = A4

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(20 * mm, height - 12 * mm, MANUAL)
    canvas.drawRightString(width - 20 * mm, height - 12 * mm, meta["doc_id"])
    canvas.setStrokeColor(colors.HexColor("#c8ccd2"))
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, height - 14 * mm, width - 20 * mm, height - 14 * mm)

    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.drawString(20 * mm, 11 * mm, f"{ISSUER} · Internal use")
    canvas.drawRightString(width - 20 * mm, 11 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def control_table(meta: dict, st: dict) -> Table:
    rows = [
        ["Document reference", meta["doc_id"]],
        ["Version", str(meta.get("version", "1.0"))],
        ["Effective date", str(meta.get("effective_date", ""))],
        ["Approved by", ISSUER],
        ["Review cycle", str(meta.get("review_cycle", "Annual"))],
        ["Classification", "Internal — compliance reference"],
    ]
    table = Table(
        [[Paragraph(f"<b>{k}</b>", st["meta"]), Paragraph(v, st["meta"])] for k, v in rows],
        colWidths=[42 * mm, 118 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd2")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef1f5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_pdf(source: Path, out: Path) -> tuple[str, int]:
    raw = source.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{source.name}: missing frontmatter")
    meta = yaml.safe_load(match.group(1))
    body = raw[match.end() :]

    st = styles()
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(out), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=f"{meta['doc_id']} — {meta['title']}",
        author=ISSUER, subject=MANUAL,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body"
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="main", frames=[frame],
                onPage=lambda c, d: header_footer(c, d, meta),
            )
        ]
    )

    flow: list = [
        Paragraph(f"{meta['doc_id']} — {meta['title']}", st["title"]),
        Paragraph(f"{MANUAL} · {ISSUER}", st["subtitle"]),
        control_table(meta, st),
        Spacer(1, 8 * mm),
    ]

    clause_count = 0
    for line in body.splitlines():
        line = line.rstrip()
        if not line or line.startswith("# "):
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flow.append(
                Paragraph(f"{heading.group(1)}&nbsp;&nbsp;{inline(heading.group(2))}",
                          st["heading"])
            )
            continue

        clause = CLAUSE_RE.match(line)
        if clause:
            clause_count += 1
            flow.append(
                Paragraph(
                    f"<b>{clause.group(1)}</b>&nbsp;&nbsp;{inline(clause.group(2))}",
                    st["clause"],
                )
            )
            continue

        flow.append(Paragraph(inline(line), st["clause"]))

    sources = meta.get("sources", [])
    if sources:
        items = "".join(f"<br/>&nbsp;&nbsp;• {inline(str(s))}" for s in sources)
        flow.append(PageBreak())
        flow.append(Paragraph("Sources and attribution", st["heading"]))
        flow.append(
            KeepTogether(
                Paragraph(
                    "This document is internal guidance prepared by "
                    f"{ISSUER}. It paraphrases and applies principles published "
                    "in the sources below; it reproduces no proprietary standard "
                    "text and carries no authority of its own. Where this manual "
                    "and an applicable external standard differ, the external "
                    "standard governs." + items,
                    st["note"],
                )
            )
        )

    doc.build(flow)
    return meta["doc_id"], clause_count


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f"ERROR: no source directory at {SOURCE_DIR}", file=sys.stderr)
        return 2

    sources = sorted(SOURCE_DIR.glob("*.md"))
    if not sources:
        print(f"ERROR: no markdown sources in {SOURCE_DIR}", file=sys.stderr)
        return 2

    total = 0
    for source in sources:
        out = PDF_DIR / f"{source.stem}.pdf"
        doc_id, clauses = build_pdf(source, out)
        size_kb = out.stat().st_size / 1024
        print(f"  {doc_id}  {out.name:34s} {clauses:3d} clauses  {size_kb:6.1f} KB")
        total += clauses

    print(f"\nbuilt {len(sources)} PDFs, {total} clauses -> {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
