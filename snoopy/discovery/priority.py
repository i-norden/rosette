"""Priority scoring algorithm for ranking academic papers for investigation."""

from __future__ import annotations

from math import log10
from typing import Optional

from pydantic import BaseModel, Field

# Reference year for recency calculations
REFERENCE_YEAR = 2026

# Representative set of top-100 global research institutions
TOP_100_INSTITUTIONS: set[str] = {
    "Harvard University",
    "Massachusetts Institute of Technology",
    "Stanford University",
    "University of Oxford",
    "University of Cambridge",
    "Johns Hopkins University",
    "California Institute of Technology",
    "University of California, Berkeley",
    "University of California, San Francisco",
    "University of California, Los Angeles",
    "Columbia University",
    "Yale University",
    "Princeton University",
    "University of Chicago",
    "University of Pennsylvania",
    "Duke University",
    "Imperial College London",
    "ETH Zurich",
    "University of Toronto",
    "University of Tokyo",
}


class PaperMetadata(BaseModel):
    """Structured metadata for a paper used in priority scoring."""

    citation_count: int = Field(default=0, ge=0)
    journal_quartile: int = Field(default=4, ge=1, le=4)
    influential_citations: int = Field(default=0, ge=0)
    max_author_hindex: int = Field(default=0, ge=0)
    institution_in_top100: bool = False
    has_retraction_concern: bool = False
    year: int = Field(default=REFERENCE_YEAR)
    has_image_heavy_methods: bool = False


def _safe_log10(value: float) -> float:
    """Compute log10 safely, returning 0 for non-positive values."""
    if value <= 0:
        return 0.0
    return log10(value)


def compute_priority(paper: PaperMetadata) -> float:
    """Compute a composite priority score from 0 to 100 for a paper.

    The score is a weighted sum of normalized sub-scores:
        - 25%: citation count (log10 scaled, capped at 10,000)
        - 20%: journal quartile (Q1=1.0, Q2=0.75, Q3=0.5, Q4=0.25)
        - 15%: influential citations (log10 scaled, capped at 1,000)
        - 10%: max author h-index (linear, capped at 80)
        - 10%: institution prestige (top-100 = 1.0, else 0.3)
        - 10%: retraction concern flag (yes = 1.0, no = 0.0)
        -  5%: recency (older = higher, capped at 20 years from 2026)
        -  5%: image-heavy methods (yes = 1.0, else 0.5)

    Args:
        paper: PaperMetadata instance with all relevant fields.

    Returns:
        Float score between 0 and 100.
    """
    # --- Citation count (25%) ---
    # log10(10000) = 4.0 is the maximum
    capped_citations = min(paper.citation_count, 10_000)
    citation_score = _safe_log10(capped_citations) / 4.0 if capped_citations > 0 else 0.0
    citation_score = min(citation_score, 1.0)

    # --- Journal quartile (20%) ---
    quartile_map = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25}
    journal_score = quartile_map.get(paper.journal_quartile, 0.25)

    # --- Influential citations (15%) ---
    # log10(1000) = 3.0 is the maximum
    capped_influential = min(paper.influential_citations, 1_000)
    influential_score = (
        _safe_log10(capped_influential) / 3.0 if capped_influential > 0 else 0.0
    )
    influential_score = min(influential_score, 1.0)

    # --- Max author h-index (10%) ---
    capped_hindex = min(paper.max_author_hindex, 80)
    hindex_score = capped_hindex / 80.0

    # --- Institution prestige (10%) ---
    institution_score = 1.0 if paper.institution_in_top100 else 0.3

    # --- Retraction concern (10%) ---
    retraction_score = 1.0 if paper.has_retraction_concern else 0.0

    # --- Recency (5%) — older papers score higher (more established) ---
    age = REFERENCE_YEAR - paper.year
    capped_age = min(max(age, 0), 20)
    recency_score = capped_age / 20.0

    # --- Image-heavy methods (5%) ---
    image_score = 1.0 if paper.has_image_heavy_methods else 0.5

    # --- Weighted combination ---
    total = (
        0.25 * citation_score
        + 0.20 * journal_score
        + 0.15 * influential_score
        + 0.10 * hindex_score
        + 0.10 * institution_score
        + 0.10 * retraction_score
        + 0.05 * recency_score
        + 0.05 * image_score
    )

    # Scale to 0-100
    return round(total * 100.0, 2)


def check_institution_in_top100(institutions: list[str]) -> bool:
    """Check if any of the given institution names match the top-100 set.

    Uses case-insensitive substring matching to handle variations in
    institutional naming conventions (e.g., "Harvard Medical School"
    would match "Harvard University").

    Args:
        institutions: List of institution name strings.

    Returns:
        True if any institution matches a top-100 entry.
    """
    if not institutions:
        return False

    top100_lower = {name.lower() for name in TOP_100_INSTITUTIONS}

    for inst in institutions:
        if not inst:
            continue
        inst_lower = inst.lower().strip()

        # Exact match
        if inst_lower in top100_lower:
            return True

        # Substring match: check if any top-100 name is contained in the
        # institution string, or vice versa
        for top_name in top100_lower:
            if top_name in inst_lower or inst_lower in top_name:
                return True

    return False
