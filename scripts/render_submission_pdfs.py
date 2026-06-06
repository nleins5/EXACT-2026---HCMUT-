from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ZIP_PATH = REPORTS / "EXACT2026_Submission_Package_AI_WITH_BRO.zip"
PACKAGE_MD = REPORTS / "Submission_Package.md"


@dataclass(frozen=True)
class DocumentSpec:
    markdown: Path
    pdf: Path
    pagesize: tuple[float, float]
    compact: bool = False


DOCUMENTS = [
    DocumentSpec(
        markdown=REPORTS / "Solution_Description.md",
        pdf=REPORTS / "EXACT2026_Solution_Description_AI_WITH_BRO.pdf",
        pagesize=letter,
    ),
    DocumentSpec(
        markdown=REPORTS / "Data_Disclosure.md",
        pdf=REPORTS / "EXACT2026_Data_Disclosure_AI_WITH_BRO.pdf",
        pagesize=landscape(letter),
        compact=True,
    ),
]

PALETTE = {
    "ink": colors.HexColor("#111827"),
    "muted": colors.HexColor("#4b5563"),
    "line": colors.HexColor("#d6dde8"),
    "blue": colors.HexColor("#1d4ed8"),
    "blue_soft": colors.HexColor("#eff6ff"),
    "teal": colors.HexColor("#14b8a6"),
    "amber": colors.HexColor("#f59e0b"),
    "panel": colors.HexColor("#f8fafc"),
    "white": colors.white,
}


class AccentRule(Flowable):
    def __init__(self, width: float, height: float = 5) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        return avail_width, self.height

    def draw(self) -> None:
        segment = self.width / 3
        self.canv.setFillColor(PALETTE["blue"])
        self.canv.roundRect(0, 0, self.width, self.height, 2.5, stroke=0, fill=1)
        self.canv.setFillColor(PALETTE["teal"])
        self.canv.rect(segment, 0, segment, self.height, stroke=0, fill=1)
        self.canv.setFillColor(PALETTE["amber"])
        self.canv.roundRect(segment * 2, 0, segment, self.height, 2.5, stroke=0, fill=1)


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline_markup(text: str) -> str:
    text = escape(text.strip())
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_styles(compact: bool = False) -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body_size = 8.0 if compact else 9.2
    body_leading = 9.25 if compact else 11.0
    table_size = 6.5 if compact else 7.3
    table_leading = 7.35 if compact else 8.4

    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
            fontSize=16.8 if compact else 20.5,
            leading=18.8 if compact else 23.0,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.8 if compact else 9.0,
            leading=9.2 if compact else 10.5,
            textColor=PALETTE["muted"],
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.8 if compact else 11.0,
            leading=11.0 if compact else 12.8,
            textColor=colors.HexColor("#1e3a8a"),
            backColor=PALETTE["blue_soft"],
            borderColor=PALETTE["blue"],
            borderWidth=0,
            borderPadding=(3, 5, 3, 5),
            leftIndent=0,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=body_size,
            leading=body_leading,
            textColor=PALETTE["ink"],
            spaceAfter=4 if compact else 5,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=body_size,
            leading=body_leading,
            leftIndent=13,
            firstLineIndent=-8,
            textColor=PALETTE["ink"],
            spaceAfter=2.5 if compact else 3,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.3 if compact else 9.2,
            leading=10.0 if compact else 11.3,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=0,
        ),
        "table": ParagraphStyle(
            "TableBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=table_size,
            leading=table_leading,
            textColor=PALETTE["ink"],
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=table_size,
            leading=table_leading,
            textColor=colors.HexColor("#0f172a"),
        ),
    }


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and "|" in lines[index]:
        raw = lines[index].strip()
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def table_widths(column_count: int, available_width: float, compact: bool) -> list[float]:
    if column_count == 3:
        ratios = [0.19, 0.43, 0.38]
    elif column_count == 5:
        ratios = [0.16, 0.18, 0.10, 0.31, 0.25]
    else:
        ratios = [1 / column_count] * column_count
    return [available_width * ratio for ratio in ratios]


def build_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], available_width: float, compact: bool) -> Table:
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = Table(
        data,
        colWidths=table_widths(column_count, available_width, compact),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALETTE["blue_soft"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.45, PALETTE["line"]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALETTE["white"], PALETTE["panel"]]),
            ]
        )
    )
    return table


def build_callout(text: str, styles: dict[str, ParagraphStyle], available_width: float) -> Table:
    table = Table(
        [[Paragraph(inline_markup(text), styles["quote"])]],
        colWidths=[available_width],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALETTE["panel"]),
                ("BOX", (0, 0), (-1, -1), 0.45, PALETTE["line"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def markdown_to_story(markdown: str, styles: dict[str, ParagraphStyle], available_width: float, compact: bool) -> list:
    lines = markdown.splitlines()
    story: list = [AccentRule(available_width), Spacer(1, 8)]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("> "):
            story.append(build_callout(line[2:], styles, available_width))
            story.append(Spacer(1, 5))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:]), styles["bullet"], bulletText="-"))
        elif re.match(r"^\d+\.\s+", line):
            number, rest = line.split(".", 1)
            story.append(Paragraph(inline_markup(rest.strip()), styles["bullet"], bulletText=f"{number}."))
        elif "|" in line and index + 1 < len(lines) and "|" in lines[index + 1]:
            rows, index = parse_table(lines, index)
            story.append(build_table(rows, styles, available_width, compact))
            story.append(Spacer(1, 5))
            continue
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(inline_markup(line), styles["subtitle"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["body"]))
        index += 1
    return story


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(PALETTE["line"])
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.36 * inch, doc.pagesize[0] - doc.rightMargin, 0.36 * inch)
    canvas.setFillColor(PALETTE["muted"])
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.23 * inch,
        "AI WITH BRO | EXACT 2026 | Open-source reasoning system",
    )
    canvas.restoreState()


def render_pdf(spec: DocumentSpec) -> None:
    margin_x = 0.46 * inch if not spec.compact else 0.40 * inch
    margin_y = 0.42 * inch if not spec.compact else 0.34 * inch
    doc = SimpleDocTemplate(
        str(spec.pdf),
        pagesize=spec.pagesize,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=0.50 * inch,
    )
    available_width = spec.pagesize[0] - doc.leftMargin - doc.rightMargin
    story = markdown_to_story(
        spec.markdown.read_text(encoding="utf-8"),
        make_styles(spec.compact),
        available_width,
        spec.compact,
    )
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)


def build_zip() -> None:
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for spec in DOCUMENTS:
            zf.write(spec.pdf, spec.pdf.name)
        zf.write(PACKAGE_MD, PACKAGE_MD.name)


def main() -> None:
    for spec in DOCUMENTS:
        render_pdf(spec)
        print(spec.pdf)
    build_zip()
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
