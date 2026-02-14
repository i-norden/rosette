"""PDF text and metadata extraction using PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import httpx

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Extracted text content from a single PDF page."""

    page_number: int
    text: str
    word_count: int


def extract_text(pdf_path: str) -> list[PageText]:
    """Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list of PageText objects, one per page, ordered by page number.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[PageText] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            words = text.split()
            pages.append(
                PageText(
                    page_number=page_num + 1,
                    text=text,
                    word_count=len(words),
                )
            )
    finally:
        doc.close()

    logger.info(
        "Extracted text from %d pages in %s (total words: %d)",
        len(pages),
        pdf_path,
        sum(p.word_count for p in pages),
    )
    return pages


def extract_metadata(pdf_path: str) -> dict:
    """Extract PDF metadata such as title, author, and creation date.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A dictionary containing metadata fields. Common keys include
        ``title``, ``author``, ``subject``, ``keywords``, ``creator``,
        ``producer``, ``creationDate``, ``modDate``, and ``page_count``.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    try:
        raw_meta: dict = doc.metadata or {}
        metadata: dict = {
            "title": raw_meta.get("title", ""),
            "author": raw_meta.get("author", ""),
            "subject": raw_meta.get("subject", ""),
            "keywords": raw_meta.get("keywords", ""),
            "creator": raw_meta.get("creator", ""),
            "producer": raw_meta.get("producer", ""),
            "creationDate": raw_meta.get("creationDate", ""),
            "modDate": raw_meta.get("modDate", ""),
            "page_count": len(doc),
        }
    finally:
        doc.close()

    logger.info("Extracted metadata from %s: %s", pdf_path, metadata.get("title", ""))
    return metadata


async def download_pdf(url: str, output_path: str) -> str:
    """Download a PDF from a URL and compute its SHA-256 hash.

    The file is written to *output_path*.  Intermediate directories are
    created automatically if they do not already exist.

    Args:
        url: The URL to download the PDF from.
        output_path: Local filesystem path where the PDF will be saved.

    Returns:
        The hex-encoded SHA-256 hash of the downloaded file.

    Raises:
        httpx.HTTPStatusError: If the HTTP response indicates an error.
        RuntimeError: If the download fails for any other reason.
        ValueError: If the URL scheme is not HTTPS.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("https",):
        raise ValueError(f"Only HTTPS URLs are allowed for PDF download, got: {parsed.scheme}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sha256 = hashlib.sha256()
    tmp_path = out.with_suffix(".tmp")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        fh.write(chunk)
                        sha256.update(chunk)
    except httpx.HTTPStatusError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download PDF from {url}") from exc

    # Atomic rename to final path on success
    import os

    os.rename(tmp_path, out)

    # Validate the downloaded file starts with the PDF magic bytes
    try:
        with open(out, "rb") as fh:
            magic = fh.read(5)
        if not magic.startswith(b"%PDF-"):
            out.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file from {url} is not a valid PDF (missing %%PDF- magic bytes)"
            )
    except ValueError:
        raise
    except Exception as exc:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to validate downloaded PDF from {url}") from exc

    digest = sha256.hexdigest()
    logger.info("Downloaded PDF from %s to %s (SHA-256: %s)", url, output_path, digest)
    return digest
