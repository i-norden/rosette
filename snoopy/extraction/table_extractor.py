"""Table extraction from PDFs using pdfplumber."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a single extracted table."""

    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str


def _clean_cell(value: object) -> str:
    """Normalise a cell value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _table_to_raw_text(headers: list[str], rows: list[list[str]]) -> str:
    """Produce a simple text representation of the table."""
    lines: list[str] = []
    if headers:
        lines.append("\t".join(headers))
        lines.append("-" * 40)
    for row in rows:
        lines.append("\t".join(row))
    return "\n".join(lines)


def extract_tables(pdf_path: str) -> list[TableInfo]:
    """Extract all tables from a PDF using *pdfplumber*.

    Each table's first row is treated as headers.  Remaining rows are
    returned as lists of strings.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of :class:`TableInfo` objects ordered by page number and
        table index within each page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    tables: list[TableInfo] = []

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_tables = page.extract_tables()
            except Exception:
                logger.warning(
                    "Failed to extract tables from page %d of %s",
                    page_num,
                    pdf_path,
                )
                continue

            if not page_tables:
                continue

            for table_idx, raw_table in enumerate(page_tables):
                if not raw_table or len(raw_table) < 1:
                    continue

                # First row is treated as headers.
                headers = [_clean_cell(c) for c in raw_table[0]]

                rows: list[list[str]] = []
                for raw_row in raw_table[1:]:
                    cleaned = [_clean_cell(c) for c in raw_row]
                    rows.append(cleaned)

                raw_text = _table_to_raw_text(headers, rows)

                tables.append(
                    TableInfo(
                        page_number=page_num,
                        table_index=table_idx,
                        headers=headers,
                        rows=rows,
                        raw_text=raw_text,
                    )
                )
    finally:
        pdf.close()

    logger.info("Extracted %d tables from %s", len(tables), pdf_path)
    return tables
