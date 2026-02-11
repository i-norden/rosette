"""Cross-paper image/data comparison using perceptual hashing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import imagehash
from PIL import Image
from sqlalchemy import select

from snoopy.db.models import Figure
from snoopy.db.session import get_session

logger = logging.getLogger(__name__)

_PAGE_SIZE = 500  # Number of figures to fetch per DB page


@dataclass
class DuplicateMatch:
    figure_id_a: str
    figure_id_b: str
    paper_id_a: str
    paper_id_b: str
    hash_distance: int
    figure_label_a: str = ""
    figure_label_b: str = ""


@dataclass
class CrossReferenceResult:
    matches: list[DuplicateMatch] = field(default_factory=list)
    total_figures_checked: int = 0
    cross_paper_duplicates: int = 0


def compute_phash(image_path: str, hash_size: int = 16) -> str | None:
    """Compute perceptual hash for an image."""
    try:
        img = Image.open(image_path)
        h = imagehash.phash(img, hash_size=hash_size)
        return str(h)
    except Exception:
        return None


def compute_ahash(image_path: str, hash_size: int = 16) -> str | None:
    """Compute average hash for an image."""
    try:
        img = Image.open(image_path)
        h = imagehash.average_hash(img, hash_size=hash_size)
        return str(h)
    except Exception:
        return None


def hash_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two hex hash strings.

    Raises:
        ValueError: If hash strings have mismatched lengths.
    """
    if len(hash_a) != len(hash_b):
        raise ValueError(
            f"Hash length mismatch: {len(hash_a)} vs {len(hash_b)}. "
            f"Hashes must be computed with the same hash_size."
        )
    ha = imagehash.hex_to_hash(hash_a)
    hb = imagehash.hex_to_hash(hash_b)
    return ha - hb


def find_cross_paper_duplicates(
    paper_id: str,
    max_distance: int = 10,
) -> CrossReferenceResult:
    """Find figures in other papers that are perceptually similar to this paper's figures.

    Uses stored phash values from the database rather than recomputing from disk.
    Paginates through other papers' figures to avoid loading all into memory.

    Args:
        paper_id: The paper whose figures to check against the database.
        max_distance: Maximum Hamming distance to consider a match.

    Returns:
        CrossReferenceResult with any matches found.
    """
    with get_session() as session:
        # Get this paper's figures with stored hashes (lightweight query)
        target_figures = session.execute(
            select(Figure.id, Figure.paper_id, Figure.phash, Figure.figure_label)
            .where(Figure.paper_id == paper_id)
            .where(Figure.phash.isnot(None))
        ).all()

        if not target_figures:
            return CrossReferenceResult()

        target_hashes = [
            (row.id, row.paper_id, row.phash, row.figure_label or "")
            for row in target_figures
        ]

        matches = []
        total_checked = 0
        offset = 0

        # Paginate through other papers' figures
        while True:
            other_figures = session.execute(
                select(Figure.id, Figure.paper_id, Figure.phash, Figure.figure_label)
                .where(Figure.paper_id != paper_id)
                .where(Figure.phash.isnot(None))
                .offset(offset)
                .limit(_PAGE_SIZE)
            ).all()

            if not other_figures:
                break

            for other_row in other_figures:
                total_checked += 1
                for target_id, target_paper_id, target_hash, target_label in target_hashes:
                    try:
                        dist = hash_distance(target_hash, other_row.phash)
                    except ValueError:
                        continue
                    if dist <= max_distance:
                        matches.append(DuplicateMatch(
                            figure_id_a=target_id,
                            figure_id_b=other_row.id,
                            paper_id_a=paper_id,
                            paper_id_b=other_row.paper_id,
                            hash_distance=dist,
                            figure_label_a=target_label,
                            figure_label_b=other_row.figure_label or "",
                        ))

            offset += _PAGE_SIZE

        return CrossReferenceResult(
            matches=matches,
            total_figures_checked=total_checked,
            cross_paper_duplicates=len(matches),
        )


def build_hash_index() -> dict[str, list[tuple[str, str]]]:
    """Build an in-memory index of all figure perceptual hashes.

    Reads stored hashes from the database instead of recomputing from disk.

    Returns:
        Dict mapping phash string to list of (figure_id, paper_id) tuples.
    """
    index: dict[str, list[tuple[str, str]]] = {}

    with get_session() as session:
        offset = 0
        while True:
            rows = session.execute(
                select(Figure.id, Figure.paper_id, Figure.phash)
                .where(Figure.phash.isnot(None))
                .offset(offset)
                .limit(_PAGE_SIZE)
            ).all()

            if not rows:
                break

            for row in rows:
                index.setdefault(row.phash, []).append((row.id, row.paper_id))

            offset += _PAGE_SIZE

    return index
