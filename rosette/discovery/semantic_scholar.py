"""Semantic Scholar API client for fetching paper and author metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = 30.0

# Fields to request for paper endpoints
PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,citationCount,"
    "influentialCitationCount,authors,authors.authorId,"
    "authors.name,authors.hIndex"
)

# Fields to request for author endpoints
AUTHOR_FIELDS = "authorId,name,hIndex,citationCount,paperCount"


@dataclass
class S2AuthorInfo:
    """Author metadata from Semantic Scholar."""

    author_id: Optional[str] = None
    name: str = "Unknown"
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    paper_count: Optional[int] = None


@dataclass
class S2PaperResult:
    """Paper metadata from Semantic Scholar."""

    paper_id: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    influential_citation_count: int = 0
    authors: list[S2AuthorInfo] = field(default_factory=list)


def _build_headers(api_key: Optional[str] = None) -> dict[str, str]:
    """Build request headers, adding API key if provided."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _parse_author(author_data: dict[str, Any]) -> S2AuthorInfo:
    """Parse a single author record from the API response."""
    return S2AuthorInfo(
        author_id=author_data.get("authorId"),
        name=author_data.get("name", "Unknown"),
        h_index=author_data.get("hIndex"),
        citation_count=author_data.get("citationCount"),
        paper_count=author_data.get("paperCount"),
    )


def _parse_paper(data: dict[str, Any]) -> S2PaperResult:
    """Parse a paper API response into an S2PaperResult."""
    external_ids = data.get("externalIds") or {}
    doi = external_ids.get("DOI")

    authors_raw = data.get("authors", [])
    authors = [_parse_author(a) for a in authors_raw]

    return S2PaperResult(
        paper_id=data.get("paperId"),
        doi=doi,
        title=data.get("title"),
        abstract=data.get("abstract"),
        year=data.get("year"),
        citation_count=data.get("citationCount", 0),
        influential_citation_count=data.get("influentialCitationCount", 0),
        authors=authors,
    )


async def get_paper(
    doi_or_id: str,
    api_key: Optional[str] = None,
) -> Optional[S2PaperResult]:
    """Fetch paper metadata by DOI or Semantic Scholar paper ID.

    Args:
        doi_or_id: A DOI (e.g. "10.1038/nature12373") or Semantic Scholar
            paper ID. DOIs are automatically prefixed with "DOI:" for the API.
        api_key: Optional Semantic Scholar API key for higher rate limits.

    Returns:
        S2PaperResult or None on failure.
    """
    # Determine if the input looks like a DOI (contains a slash)
    if "/" in doi_or_id and not doi_or_id.startswith("DOI:"):
        paper_ref = f"DOI:{doi_or_id}"
    else:
        paper_ref = doi_or_id

    url = f"{BASE_URL}/paper/{paper_ref}"
    headers = _build_headers(api_key)
    params = {"fields": PAPER_FIELDS}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.debug("Semantic Scholar: paper not found: %s", doi_or_id)
        elif exc.response.status_code == 429:
            logger.warning("Semantic Scholar: rate limited on paper request")
        else:
            logger.error(
                "Semantic Scholar paper HTTP error %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
        return None
    except httpx.HTTPError as exc:
        logger.error("Semantic Scholar paper request failed: %s", exc)
        return None

    return _parse_paper(data)


async def get_author(
    author_id: str,
    api_key: Optional[str] = None,
) -> Optional[S2AuthorInfo]:
    """Fetch author metadata by Semantic Scholar author ID.

    Args:
        author_id: Semantic Scholar author ID string.
        api_key: Optional API key for higher rate limits.

    Returns:
        S2AuthorInfo or None on failure.
    """
    if not author_id:
        return None

    url = f"{BASE_URL}/author/{author_id}"
    headers = _build_headers(api_key)
    params = {"fields": AUTHOR_FIELDS}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.debug("Semantic Scholar: author not found: %s", author_id)
        elif exc.response.status_code == 429:
            logger.warning("Semantic Scholar: rate limited on author request")
        else:
            logger.error(
                "Semantic Scholar author HTTP error %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
        return None
    except httpx.HTTPError as exc:
        logger.error("Semantic Scholar author request failed: %s", exc)
        return None

    return S2AuthorInfo(
        author_id=data.get("authorId"),
        name=data.get("name", "Unknown"),
        h_index=data.get("hIndex"),
        citation_count=data.get("citationCount"),
        paper_count=data.get("paperCount"),
    )
