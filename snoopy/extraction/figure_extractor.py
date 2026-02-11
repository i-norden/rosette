"""Figure and caption extraction from PDFs using PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum image dimensions (pixels) -- anything smaller is likely an icon or logo.
_MIN_WIDTH = 50
_MIN_HEIGHT = 50

# Regex patterns for figure captions.
_FIGURE_LABEL_RE = re.compile(
    r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(
    r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?[.:]\s*.+?)(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class FigureInfo:
    """Information about a single extracted figure."""

    page_number: int
    image_path: str
    image_sha256: str
    width: int
    height: int
    figure_label: str | None = None
    caption: str | None = None


def _sha256_file(path: str) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_figures(pdf_path: str, output_dir: str) -> list[FigureInfo]:
    """Extract all embedded images from a PDF and save them to *output_dir*.

    Images smaller than 50x50 pixels are silently skipped (they are
    typically icons or logos).

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory where extracted images will be written.

    Returns:
        A list of :class:`FigureInfo` objects (without caption data).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    figures: list[FigureInfo] = []
    seen_xrefs: set[int] = set()

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    logger.warning(
                        "Could not extract image xref=%d on page %d",
                        xref,
                        page_num + 1,
                    )
                    continue

                if base_image is None:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < _MIN_WIDTH or height < _MIN_HEIGHT:
                    logger.debug(
                        "Skipping small image (%dx%d) on page %d",
                        width,
                        height,
                        page_num + 1,
                    )
                    continue

                ext = base_image.get("ext", "png")
                image_bytes = base_image.get("image", b"")
                if not image_bytes:
                    continue

                filename = f"page{page_num + 1}_img{img_index}.{ext}"
                image_path = out_dir / filename
                image_path.write_bytes(image_bytes)

                image_sha256 = hashlib.sha256(image_bytes).hexdigest()

                figures.append(
                    FigureInfo(
                        page_number=page_num + 1,
                        image_path=str(image_path),
                        image_sha256=image_sha256,
                        width=width,
                        height=height,
                    )
                )
    finally:
        doc.close()

    logger.info("Extracted %d figures from %s", len(figures), pdf_path)
    return figures


def associate_captions(
    pdf_path: str,
    figures: list[FigureInfo],
) -> list[FigureInfo]:
    """Attempt to match captions to previously extracted figures.

    For each page that contains at least one figure, the function reads
    text blocks and looks for strings matching ``Figure N`` / ``Fig. N``
    patterns.  A caption is associated with the nearest figure on the
    same page based on vertical proximity of the text block to the
    image bounding box.

    Args:
        pdf_path: Path to the PDF file.
        figures: List of :class:`FigureInfo` objects (typically produced
            by :func:`extract_figures`).

    Returns:
        The same list of :class:`FigureInfo` objects, updated in-place
        with ``figure_label`` and ``caption`` fields where matches were
        found.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    if not figures:
        return figures

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    # Group figures by page number for efficient lookup.
    figures_by_page: dict[int, list[FigureInfo]] = {}
    for fig in figures:
        figures_by_page.setdefault(fig.page_number, []).append(fig)

    try:
        for page_num, page_figures in figures_by_page.items():
            page = doc[page_num - 1]

            # Collect caption candidates from text blocks.
            # Each text block is (x0, y0, x1, y1, text, block_no, block_type).
            blocks = page.get_text("blocks")
            caption_candidates: list[tuple[float, str, str]] = []

            for block in blocks:
                if len(block) < 5:
                    continue
                # block_type 0 == text
                if len(block) >= 7 and block[6] != 0:
                    continue

                block_y_mid = (block[1] + block[3]) / 2.0
                block_text = block[4].strip() if isinstance(block[4], str) else ""

                label_match = _FIGURE_LABEL_RE.search(block_text)
                if label_match is None:
                    continue

                label = label_match.group(1).strip()

                # Try to grab a longer caption string.
                caption_match = _CAPTION_RE.search(block_text)
                caption = caption_match.group(1).strip() if caption_match else label

                caption_candidates.append((block_y_mid, label, caption))

            if not caption_candidates:
                continue

            # Retrieve image bounding boxes on this page so we can match by
            # spatial proximity.  We use the vertical midpoint of each image
            # rect returned by ``page.get_image_rects()``.
            image_list = page.get_images(full=True)
            xref_to_ymid: dict[int, float] = {}
            for img_info in image_list:
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        rect = rects[0]
                        xref_to_ymid[xref] = (rect.y0 + rect.y1) / 2.0
                except Exception:
                    pass

            # For each figure on this page, find the closest caption.
            for fig in page_figures:
                # Determine y-midpoint of this figure on the page.  If we
                # cannot resolve its position we fall back to assigning
                # captions in order.
                fig_y: float | None = None

                # Walk through image_list to find a matching xref by
                # checking if the stored image path ends with the expected
                # filename pattern.  This is a heuristic.
                for img_info in image_list:
                    xref = img_info[0]
                    if xref in xref_to_ymid:
                        # Match by filename convention used in extract_figures.
                        expected_suffix = f"page{fig.page_number}_img"
                        if expected_suffix in fig.image_path:
                            fig_y = xref_to_ymid[xref]
                            break

                if fig_y is None:
                    # Fallback: assign the first unassigned caption.
                    if caption_candidates:
                        _, label, caption = caption_candidates.pop(0)
                        fig.figure_label = label
                        fig.caption = caption
                    continue

                # Find the caption candidate closest (vertically) to this
                # figure, preferring captions *below* the image.
                best_idx: int | None = None
                best_dist = float("inf")
                for idx, (cy, _label, _caption) in enumerate(caption_candidates):
                    dist = abs(cy - fig_y)
                    # Slight preference for captions below the image.
                    if cy >= fig_y:
                        dist *= 0.9
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx is not None:
                    _, label, caption = caption_candidates.pop(best_idx)
                    fig.figure_label = label
                    fig.caption = caption
    finally:
        doc.close()

    assigned = sum(1 for f in figures if f.caption is not None)
    logger.info("Associated captions with %d / %d figures", assigned, len(figures))
    return figures
