#!/usr/bin/env python
"""Render DOCUMENTATION.md to a typeset PDF.

    python scripts/build_documentation_pdf.py                  # -> DOCUMENTATION.pdf
    python scripts/build_documentation_pdf.py --out other.pdf  # choose the target

Keeps the markdown as the single source of truth so the two cannot diverge.
Supports the subset the document actually uses: ATX headings, paragraphs with
inline bold/italic/code/links, bullet and numbered lists with one nesting level,
pipe tables, fenced code blocks and horizontal rules.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path("C:/Windows/Fonts")

BODY = "SegoeUI"
MONO = "Consolas"

INK = colors.HexColor("#14171a")
MUTED = colors.HexColor("#5b6670")
ACCENT = colors.HexColor("#1a3a5c")
RULE = colors.HexColor("#c8d0d8")
CODE_BG = colors.HexColor("#f4f6f8")
HEAD_BG = colors.HexColor("#eef2f6")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# Consolas has no glyph for these; the diagrams read the same with ASCII.
GLYPH_FALLBACK = {"\u25b6": ">", "\u25c0": "<"}


def register_fonts() -> None:
    faces = [
        (BODY, "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
        (MONO, "consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"),
    ]
    for name, regular, bold, italic, bolditalic in faces:
        for suffix, filename in (
            ("", regular),
            ("-Bold", bold),
            ("-Italic", italic),
            ("-BoldItalic", bolditalic),
        ):
            path = FONT_DIR / filename
            if not path.exists():
                raise SystemExit(f"missing font: {path}")
            pdfmetrics.registerFont(TTFont(name + suffix, str(path)))
        pdfmetrics.registerFontFamily(
            name,
            normal=name,
            bold=name + "-Bold",
            italic=name + "-Italic",
            boldItalic=name + "-BoldItalic",
        )


def styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body",
        fontName=BODY,
        fontSize=8.4,
        leading=11.6,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    return {
        "body": base,
        "title": ParagraphStyle(
            "title", parent=base, fontName=BODY + "-Bold", fontSize=21, leading=25,
            textColor=ACCENT, alignment=0, spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base, fontSize=10.5, leading=15, textColor=MUTED,
            alignment=0, spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base, fontName=BODY + "-Bold", fontSize=13.8, leading=17,
            textColor=ACCENT, alignment=0, spaceBefore=3, spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base, fontName=BODY + "-Bold", fontSize=10.5, leading=13,
            textColor=ACCENT, alignment=0, spaceBefore=1, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base, fontName=BODY + "-Bold", fontSize=9.9, leading=13.5,
            textColor=INK, alignment=0, spaceBefore=6, spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base, leftIndent=11, bulletIndent=2, spaceAfter=2.5,
        ),
        "bullet2": ParagraphStyle(
            "bullet2", parent=base, leftIndent=24, bulletIndent=14, spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base, fontSize=7.7, leading=10.2, alignment=0, spaceAfter=0,
        ),
        "cellhead": ParagraphStyle(
            "cellhead", parent=base, fontName=BODY + "-Bold", fontSize=7.7,
            leading=10.2, alignment=0, spaceAfter=0, textColor=ACCENT,
        ),
        "code": ParagraphStyle(
            "code", fontName=MONO, fontSize=7.4, leading=9.6, textColor=INK,
        ),
    }


INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BARE_LINK = re.compile(r"<(https?://[^>]+)>")


def inline(text: str) -> str:
    """Convert the inline markdown subset to reportlab's mini-HTML."""
    placeholders: list[str] = []

    def stash(markup: str) -> str:
        placeholders.append(markup)
        return f"\x00{len(placeholders) - 1}\x00"

    text = MD_LINK.sub(
        lambda m: stash(
            f'<link href="{html.escape(m.group(2), quote=True)}" color="#1a3a5c">'
            f"{html.escape(m.group(1))}</link>"
        ),
        text,
    )
    text = BARE_LINK.sub(
        lambda m: stash(
            f'<link href="{html.escape(m.group(1), quote=True)}" color="#1a3a5c">'
            f"{html.escape(m.group(1))}</link>"
        ),
        text,
    )
    text = INLINE_CODE.sub(
        lambda m: stash(
            f'<font face="{MONO}" size="7.7" color="#8a3a1f">'
            f"{html.escape(m.group(1))}</font>"
        ),
        text,
    )

    text = html.escape(text)
    text = BOLD.sub(r"<b>\1</b>", text)
    text = ITALIC.sub(r"<i>\1</i>", text)

    for index, markup in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", markup)
    return text


WORD_CHARS = re.compile(r"[*`\[\]()]")


def longest_word_width(cells: list[str]) -> float:
    """Width of the widest unbreakable token, so a column never splits a word."""
    widest = 0.0
    for cell in cells:
        for word in WORD_CHARS.sub("", cell).split():
            widest = max(widest, pdfmetrics.stringWidth(word, BODY + "-Bold", 8.1))
    return widest + 11  # cell padding


def column_widths(rows: list[list[str]]) -> list[float]:
    """Weight columns by content, then guarantee each fits its longest word."""
    count = len(rows[0])
    columns = [[row[col] for row in rows if col < len(row)] for col in range(count)]

    weights = []
    for cells in columns:
        lengths = [len(c) for c in cells] or [1]
        weights.append(max(6.0, 0.45 * max(lengths) + 0.55 * sum(lengths) / len(lengths)))
    total = sum(weights)
    widths = [CONTENT_W * w / total for w in weights]

    # Grow any column that would break a word, taking the space from the widest.
    minimums = [min(longest_word_width(c), CONTENT_W * 0.3) for c in columns]
    for index, floor in enumerate(minimums):
        if widths[index] >= floor:
            continue
        deficit = floor - widths[index]
        widths[index] = floor
        donor = max(range(count), key=lambda i: widths[i] - minimums[i])
        widths[donor] = max(minimums[donor], widths[donor] - deficit)

    scale = CONTENT_W / sum(widths)
    return [w * scale for w in widths]


def build_table(rows: list[list[str]], st: dict[str, ParagraphStyle]) -> Table:
    header, *body = rows
    data = [[Paragraph(inline(c), st["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), st["cell"]) for c in row] for row in body]

    table = Table(data, colWidths=column_widths(rows), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
                ("GRID", (0, 0), (-1, -1), 0.28, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ]
        )
    )
    return table


def code_block(lines: list[str], st: dict[str, ParagraphStyle]) -> Table:
    text = "\n".join(lines)
    for bad, good in GLYPH_FALLBACK.items():
        text = text.replace(bad, good)
    inner = Preformatted(text, st["code"])
    wrapper = Table([[inner]], colWidths=[CONTENT_W], hAlign="LEFT")
    wrapper.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.28, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return wrapper


# --- the agent graph, drawn rather than spelled in dashes -----------------

NODE_FILL = {
    "llm": colors.HexColor("#dce8f4"),
    "py": colors.HexColor("#f2f4f6"),
    "io": colors.HexColor("#e8efe6"),
}
NODE_EDGE = {
    "llm": colors.HexColor("#3f6f9f"),
    "py": colors.HexColor("#98a4b0"),
    "io": colors.HexColor("#6f9066"),
}

GRAPH_NODES = [
    ("io", "query", ""),
    ("llm", "parse_query", "classify intent, extract structure   (LLM 1)"),
    ("py", "plan_retrieval", "2-4 product-anchored sub-queries"),
    ("py", "retrieve", "Qdrant top-10 x N  >  RRF K=60  >  rerank 24>8"),
    ("llm", "assess", "findings, each citing clauses   (LLM 2)"),
    ("py", "verify_citations", "drop any clause ID not actually retrieved"),
    ("py", "decide_verdict", "ordered rule table"),
    ("io", "verdict + citations", ""),
]

BOX_W = 232.0
BOX_H = 17.5
GAP = 7.0


def _arrow(drawing, x1, y1, x2, y2, colour, dash=None):
    line = Line(x1, y1, x2, y2, strokeColor=colour, strokeWidth=0.9)
    if dash:
        line.strokeDashArray = dash
    drawing.add(line)


def _head(drawing, x, y, colour, direction="down"):
    s = 3.4
    if direction == "down":
        pts = [x - s, y + s * 1.6, x + s, y + s * 1.6, x, y]
    else:
        pts = [x - s, y - s * 1.6, x + s, y - s * 1.6, x, y]
    drawing.add(Polygon(pts, fillColor=colour, strokeColor=colour))


def agent_graph(width: float) -> Drawing:
    """The compliance graph: nodes, the two conditional edges, and a legend."""
    rows = len(GRAPH_NODES)
    height = rows * BOX_H + (rows - 1) * GAP + 34
    drawing = Drawing(width, height)

    left_lane = 48.0
    box_x = 74.0
    right_lane = box_x + BOX_W + 54

    tops = [height - 20 - i * (BOX_H + GAP) for i in range(rows)]
    centres = {}

    for index, (kind, name, detail) in enumerate(GRAPH_NODES):
        top = tops[index]
        y = top - BOX_H
        centres[name] = (y + BOX_H / 2, y, top)
        drawing.add(
            Rect(
                box_x, y, BOX_W, BOX_H, rx=3, ry=3,
                fillColor=NODE_FILL[kind], strokeColor=NODE_EDGE[kind], strokeWidth=0.8,
            )
        )
        drawing.add(
            String(box_x + 9, y + (14 if detail else 9), name,
                   fontName=MONO + "-Bold", fontSize=7.4, fillColor=INK)
        )
        if detail:
            drawing.add(
                String(box_x + 9, y + 5, detail,
                       fontName=BODY, fontSize=6.3, fillColor=MUTED)
            )
        if index:
            previous = tops[index - 1] - BOX_H
            _arrow(drawing, box_x + BOX_W / 2, previous, box_x + BOX_W / 2, top + 4,
                   colors.HexColor("#6b7783"))
            _head(drawing, box_x + BOX_W / 2, top, colors.HexColor("#6b7783"))

    accent = colors.HexColor("#b4534a")
    loop = colors.HexColor("#3f6f9f")

    # Router bypass: a greeting or off-topic message skips straight to the verdict.
    _, parse_y, _ = centres["parse_query"]
    _, _, decide_top = centres["decide_verdict"]
    parse_mid = parse_y + BOX_H / 2
    decide_mid = decide_top - BOX_H / 2
    _arrow(drawing, box_x + BOX_W, parse_mid, right_lane, parse_mid, accent)
    _arrow(drawing, right_lane, parse_mid, right_lane, decide_mid, accent)
    _arrow(drawing, right_lane, decide_mid, box_x + BOX_W + 4, decide_mid, accent)
    drawing.add(Polygon(
        [box_x + BOX_W, decide_mid, box_x + BOX_W + 6, decide_mid + 3.4,
         box_x + BOX_W + 6, decide_mid - 3.4],
        fillColor=accent, strokeColor=accent,
    ))
    label_y = (parse_mid + decide_mid) / 2
    drawing.add(String(right_lane + 6, label_y + 4, "GREETING / OTHER",
                       fontName=BODY + "-Bold", fontSize=6.2, fillColor=accent))
    drawing.add(String(right_lane + 6, label_y - 4, "-> IRRELEVANT, no retrieval",
                       fontName=BODY, fontSize=6.2, fillColor=accent))

    # Broaden-and-retry: weak coverage sends control back to planning, once.
    _, retrieve_y, _ = centres["retrieve"]
    _, _, plan_top = centres["plan_retrieval"]
    retrieve_mid = retrieve_y + BOX_H / 2
    plan_mid = plan_top - BOX_H / 2
    _arrow(drawing, box_x, retrieve_mid, left_lane, retrieve_mid, loop)
    _arrow(drawing, left_lane, retrieve_mid, left_lane, plan_mid, loop)
    _arrow(drawing, left_lane, plan_mid, box_x - 4, plan_mid, loop)
    drawing.add(Polygon(
        [box_x, plan_mid, box_x - 6, plan_mid + 3.4, box_x - 6, plan_mid - 3.4],
        fillColor=loop, strokeColor=loop,
    ))
    drawing.add(String(2, plan_mid + 12, "weak",
                       fontName=BODY + "-Bold", fontSize=6.2, fillColor=loop))
    drawing.add(String(2, plan_mid + 4, "coverage:",
                       fontName=BODY, fontSize=6.2, fillColor=loop))
    drawing.add(String(2, plan_mid - 4, "broaden once",
                       fontName=BODY, fontSize=6.2, fillColor=loop))

    legend_y = 6
    for offset, (kind, label) in enumerate(
        [("llm", "LLM call"), ("py", "deterministic Python"), ("io", "in / out")]
    ):
        x = box_x + offset * 118
        drawing.add(Rect(x, legend_y, 9, 7, fillColor=NODE_FILL[kind],
                         strokeColor=NODE_EDGE[kind], strokeWidth=0.7))
        drawing.add(String(x + 13, legend_y + 1, label,
                           fontName=BODY, fontSize=6.3, fillColor=MUTED))

    return drawing


BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
NUMBER_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
TABLE_SEP = re.compile(r"^\|[\s:|-]+\|$")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md: str, st: dict[str, ParagraphStyle]) -> list:
    lines = md.splitlines()
    flow: list = []
    para: list[str] = []
    index = 0

    def flush() -> None:
        if para:
            flow.append(Paragraph(inline(" ".join(para)), st["body"]))
            para.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            info = stripped[3:].strip().lower()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            flow.append(Spacer(1, 2))
            if info == "diagram":
                flow.append(agent_graph(CONTENT_W))
            else:
                flow.append(code_block(block, st))
            flow.append(Spacer(1, 7))
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and TABLE_SEP.match(
            lines[index + 1].strip()
        ):
            flush()
            rows = [split_row(lines[index])]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_row(lines[index]))
                index += 1
            flow.append(Spacer(1, 2))
            table = build_table(rows, st)
            # A short table split across a page break strands a row under a
            # repeated header, which reads as a different table.
            flow.append(KeepTogether(table) if len(rows) <= 9 else table)
            flow.append(Spacer(1, 6))
            continue

        if not stripped:
            flush()
            index += 1
            continue

        if stripped == "---":
            flush()
            flow.append(Spacer(1, 5))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
            flow.append(Spacer(1, 9))
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            text = heading.group(2)
            if level == 1:
                node = Paragraph(inline(text), st["h1"])
            elif level == 2:
                flow.append(Spacer(1, 4))
                node = KeepTogether(
                    [
                        HRFlowable(width="100%", thickness=0.7, color=ACCENT),
                        Spacer(1, 3),
                        Paragraph(inline(text), st["h2"]),
                    ]
                )
            else:
                node = Paragraph(inline(text), st["h3"])
            # A heading alone at the foot of a page is a widow; drag it forward.
            node.keepWithNext = 1
            flow.append(node)
            index += 1
            continue

        bullet = BULLET_RE.match(line)
        if bullet:
            flush()
            depth = len(bullet.group(1))
            body_text = bullet.group(2)
            index += 1
            while index < len(lines):
                nxt = lines[index]
                if not nxt.strip() or BULLET_RE.match(nxt) or NUMBER_RE.match(nxt):
                    break
                if re.match(r"^(#{1,4})\s+", nxt.strip()) or nxt.strip().startswith("|"):
                    break
                body_text += " " + nxt.strip()
                index += 1
            style = st["bullet2"] if depth >= 2 else st["bullet"]
            flow.append(Paragraph(inline(body_text), style, bulletText="\u2022"))
            continue

        numbered = NUMBER_RE.match(line)
        if numbered:
            flush()
            depth = len(numbered.group(1))
            marker = numbered.group(2) + "."
            body_text = numbered.group(3)
            index += 1
            while index < len(lines):
                nxt = lines[index]
                if not nxt.strip() or BULLET_RE.match(nxt) or NUMBER_RE.match(nxt):
                    break
                if re.match(r"^(#{1,4})\s+", nxt.strip()) or nxt.strip().startswith("|"):
                    break
                body_text += " " + nxt.strip()
                index += 1
            style = st["bullet2"] if depth >= 2 else st["bullet"]
            flow.append(Paragraph(inline(body_text), style, bulletText=marker))
            continue

        para.append(stripped)
        index += 1

    flush()
    return flow


def title_block(md: str, st: dict[str, ParagraphStyle]) -> tuple[list, str]:
    """Lift the leading H1 and its intro paragraph into a title block."""
    lines = md.splitlines()
    title = "Technical Design"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]

    intro: list[str] = []
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].strip() and not lines[0].startswith("**"):
        intro.append(lines[0].strip())
        lines.pop(0)

    flow = [
        Paragraph(inline(title), st["title"]),
        Paragraph(inline(" ".join(intro)), st["subtitle"]),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.1, color=ACCENT),
        Spacer(1, 11),
    ]
    return flow, "\n".join(lines)


def decorate(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont(BODY, 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        MARGIN, MARGIN * 0.55, "Sharia Compliance Agent \u2014 Technical Design"
    )
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.55, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, MARGIN * 0.78, PAGE_W - MARGIN, MARGIN * 0.78)
    canvas.restoreState()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(ROOT / "DESIGN.md"))
    parser.add_argument("--out", default=str(ROOT / "DESIGN.pdf"))
    args = parser.parse_args()

    source = Path(args.src)
    if not source.exists():
        raise SystemExit(f"missing source: {source}")

    register_fonts()
    st = styles()
    md = source.read_text(encoding="utf-8")

    head, rest = title_block(md, st)
    flow = head + convert(rest, st)

    doc = BaseDocTemplate(
        args.out,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Sharia Compliance Agent - Technical Design",
        author="Ahmad K. Elsayed",
        subject="Technical design, production evolution and gaps",
    )
    frame = Frame(
        MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(flow)

    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
