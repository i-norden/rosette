"""Cross-paper image/data comparison using perceptual hashing and SSIM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import imagehash
import numpy as np
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
            (row.id, row.paper_id, row.phash, row.figure_label or "") for row in target_figures
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
                        matches.append(
                            DuplicateMatch(
                                figure_id_a=target_id,
                                figure_id_b=other_row.id,
                                paper_id_a=paper_id,
                                paper_id_b=other_row.paper_id,
                                hash_distance=dist,
                                figure_label_a=target_label,
                                figure_label_b=other_row.figure_label or "",
                            )
                        )

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


@dataclass
class SSIMResult:
    """Result of SSIM comparison between two images."""

    score: float
    is_duplicate: bool
    difference_map: np.ndarray | None = None


def compute_ssim(
    image_a_path: str,
    image_b_path: str,
    threshold: float = 0.95,
    return_map: bool = False,
) -> SSIMResult:
    """Compute Structural Similarity Index between two images.

    Pure numpy implementation (no scikit-image dependency). Provides a spatial
    difference map showing exactly where images differ, useful for detecting
    partial image recycling.

    Args:
        image_a_path: Path to the first image.
        image_b_path: Path to the second image.
        threshold: SSIM score above which images are considered duplicates.
        return_map: If True, include the per-pixel SSIM map in the result.

    Returns:
        SSIMResult with similarity score and optional spatial map.
    """
    try:
        img_a = np.array(Image.open(image_a_path).convert("L"), dtype=np.float64)
        img_b = np.array(Image.open(image_b_path).convert("L"), dtype=np.float64)
    except Exception:
        return SSIMResult(score=0.0, is_duplicate=False)

    # Resize to match dimensions if needed
    if img_a.shape != img_b.shape:
        min_h = min(img_a.shape[0], img_b.shape[0])
        min_w = min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:min_h, :min_w]
        img_b = img_b[:min_h, :min_w]

    if img_a.size == 0:
        return SSIMResult(score=0.0, is_duplicate=False)

    # SSIM constants (per Wang et al. 2004)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    window_size = 7

    # Use uniform window (box filter) for simplicity
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = img_a.shape
    if h < window_size or w < window_size:
        # Image too small for windowed SSIM, compute global
        mu_a = np.mean(img_a)
        mu_b = np.mean(img_b)
        sigma_a_sq = np.var(img_a)
        sigma_b_sq = np.var(img_b)
        sigma_ab = np.mean((img_a - mu_a) * (img_b - mu_b))
        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
        score = float(num / den) if den != 0 else 0.0
        return SSIMResult(score=score, is_duplicate=score >= threshold)

    # Windowed SSIM via sliding windows
    win_a = sliding_window_view(img_a, (window_size, window_size))
    win_b = sliding_window_view(img_b, (window_size, window_size))

    mu_a = np.mean(win_a, axis=(-2, -1))
    mu_b = np.mean(win_b, axis=(-2, -1))

    sigma_a_sq = np.var(win_a, axis=(-2, -1))
    sigma_b_sq = np.var(win_b, axis=(-2, -1))
    sigma_ab = np.mean(
        (win_a - mu_a[..., np.newaxis, np.newaxis]) * (win_b - mu_b[..., np.newaxis, np.newaxis]),
        axis=(-2, -1),
    )

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = np.where(den != 0, num / den, 0.0)
    score = float(np.mean(ssim_map))

    return SSIMResult(
        score=score,
        is_duplicate=score >= threshold,
        difference_map=ssim_map if return_map else None,
    )
