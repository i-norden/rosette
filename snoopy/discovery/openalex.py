"""OpenAlex API client for discovering and fetching academic paper metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
DEFAULT_TIMEOUT = 30.0
DEFAULT_LIMIT = 25
MAX_PER_PAGE = 200


@dataclass
class AuthorInfo:
    """Author metadata from OpenAlex."""

    name: str
    openalex_id: Optional[str] = None
    orcid: Optional[str] = None
    institution_name: Optional[str] = None
    institution_ror: Optional[str] = None


@dataclass
class PaperResult:
    """Paper metadata returned from OpenAlex."""

    openalex_id: str
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[AuthorInfo] = field(default_factory=list)
    journal: Optional[str] = None
    issn: Optional[str] = None
    citation_count: int = 0
    publication_year: Optional[int] = None
    open_access_url: Optional[str] = None


def _build_headers(email: Optional[str] = None) -> dict[str, str]:
    """Build request headers, using polite pool if email is provided."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if email:
        headers["User-Agent"] = f"snoopy/1.0 (mailto:{email})"
    return headers


def _parse_author(authorship: dict[str, Any]) -> AuthorInfo:
    """Parse a single authorship record into an AuthorInfo."""
    author_data = authorship.get("author", {})
    institution_data = {}
    institutions = authorship.get("institutions", [])
    if institutions:
        institution_data = institutions[0]

    return AuthorInfo(
        name=author_data.get("display_name", "Unknown"),
        openalex_id=author_data.get("id"),
        orcid=author_data.get("orcid"),
        institution_name=institution_data.get("display_name"),
        institution_ror=institution_data.get("ror"),
    )


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _parse_work(work: dict[str, Any]) -> PaperResult:
    """Parse a single work record from the OpenAlex API response."""
    # Extract primary location journal info
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    journal_name = source.get("display_name")
    issn_l = source.get("issn_l")

    # Extract open access URL
    oa = work.get("open_access") or {}
    oa_url = oa.get("oa_url")

    # Reconstruct abstract from inverted index
    abstract_inverted = work.get("abstract_inverted_index")
    abstract = _reconstruct_abstract(abstract_inverted) if abstract_inverted else None

    # Parse authors
    authorships = work.get("authorships", [])
    authors = [_parse_author(a) for a in authorships]

    # Normalize DOI (strip https://doi.org/ prefix if present)
    doi_raw = work.get("doi")
    doi: Optional[str] = None
    if doi_raw:
        doi = doi_raw.replace("https://doi.org/", "")

    return PaperResult(
        openalex_id=work.get("id", ""),
        doi=doi,
        title=work.get("title"),
        abstract=abstract,
        authors=authors,
        journal=journal_name,
        issn=issn_l,
        citation_count=work.get("cited_by_count", 0),
        publication_year=work.get("publication_year"),
        open_access_url=oa_url,
    )


async def search_works(
    query: str,
    field: Optional[str] = None,
    min_citations: Optional[int] = None,
    limit: int = DEFAULT_LIMIT,
    email: Optional[str] = None,
) -> list[PaperResult]:
    """Search OpenAlex for works matching a query and optional filters.

    Args:
        query: Free-text search query.
        field: Optional concept/field filter (e.g. "computer science").
        min_citations: Optional minimum citation count filter.
        limit: Maximum number of results to return (max 200).
        email: Email for polite pool (higher rate limits).

    Returns:
        List of PaperResult dataclass instances.
    """
    params: dict[str, Any] = {
        "search": query,
        "per_page": min(limit, MAX_PER_PAGE),
    }

    # Build filter string
    filters: list[str] = []
    if field:
        filters.append(f"concepts.display_name.search:{field}")
    if min_citations is not None and min_citations > 0:
        filters.append(f"cited_by_count:>{min_citations}")
    if filters:
        params["filter"] = ",".join(filters)

    headers = _build_headers(email)

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}/works",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "OpenAlex search HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return []
    except httpx.HTTPError as exc:
        logger.error("OpenAlex search request failed: %s", exc)
        return []

    results_raw = data.get("results", [])
    return [_parse_work(w) for w in results_raw]


async def get_work(
    openalex_id: str,
    email: Optional[str] = None,
) -> Optional[PaperResult]:
    """Fetch a single work by its OpenAlex ID.

    Args:
        openalex_id: Full OpenAlex ID URL or short ID (e.g. "W2741809807").
        email: Email for polite pool.

    Returns:
        PaperResult or None on failure.
    """
    # Normalize ID: accept both full URL and short form
    if openalex_id.startswith("https://"):
        url = openalex_id
    else:
        url = f"{BASE_URL}/works/{openalex_id}"

    headers = _build_headers(email)

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "OpenAlex get_work HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return None
    except httpx.HTTPError as exc:
        logger.error("OpenAlex get_work request failed: %s", exc)
        return None

    return _parse_work(data)
