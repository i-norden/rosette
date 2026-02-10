"""Cross-paper image/data comparison using perceptual hashing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image
from sqlalchemy import select

from snoopy.db.models import Figure
from snoopy.db.session import get_session


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
    """Compute Hamming distance between two hex hash strings."""
    if len(hash_a) != len(hash_b):
        return 999
    ha = imagehash.hex_to_hash(hash_a)
    hb = imagehash.hex_to_hash(hash_b)
    return ha - hb


def find_cross_paper_duplicates(
    paper_id: str,
    max_distance: int = 10,
) -> CrossReferenceResult:
    """Find figures in other papers that are perceptually similar to this paper's figures.

    Args:
        paper_id: The paper whose figures to check against the database.
        max_distance: Maximum Hamming distance to consider a match.

    Returns:
        CrossReferenceResult with any matches found.
    """
    with get_session() as session:
        # Get this paper's figures
        target_figures = session.execute(
            select(Figure).where(Figure.paper_id == paper_id)
        ).scalars().all()

        if not target_figures:
            return CrossReferenceResult()

        # Compute hashes for target figures
        target_hashes: list[tuple[Figure, str]] = []
        for fig in target_figures:
            if fig.image_path and Path(fig.image_path).exists():
                h = compute_phash(fig.image_path)
                if h:
                    target_hashes.append((fig, h))

        if not target_hashes:
            return CrossReferenceResult()

        # Get all other figures in the database
        other_figures = session.execute(
            select(Figure).where(Figure.paper_id != paper_id)
        ).scalars().all()

        matches = []
        for other_fig in other_figures:
            if not other_fig.image_path or not Path(other_fig.image_path).exists():
                continue
            other_hash = compute_phash(other_fig.image_path)
            if not other_hash:
                continue

            for target_fig, target_hash in target_hashes:
                dist = hash_distance(target_hash, other_hash)
                if dist <= max_distance:
                    matches.append(DuplicateMatch(
                        figure_id_a=target_fig.id,
                        figure_id_b=other_fig.id,
                        paper_id_a=paper_id,
                        paper_id_b=other_fig.paper_id,
                        hash_distance=dist,
                        figure_label_a=target_fig.figure_label or "",
                        figure_label_b=other_fig.figure_label or "",
                    ))

        return CrossReferenceResult(
            matches=matches,
            total_figures_checked=len(target_hashes),
            cross_paper_duplicates=len(matches),
        )


def build_hash_index() -> dict[str, list[tuple[str, str]]]:
    """Build an in-memory index of all figure perceptual hashes.

    Returns:
        Dict mapping phash string to list of (figure_id, paper_id) tuples.
    """
    index: dict[str, list[tuple[str, str]]] = {}

    with get_session() as session:
        figures = session.execute(select(Figure)).scalars().all()
        for fig in figures:
            if fig.image_path and Path(fig.image_path).exists():
                h = compute_phash(fig.image_path)
                if h:
                    index.setdefault(h, []).append((fig.id, fig.paper_id))

    return index
