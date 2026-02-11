"""Retraction Watch / Crossref retraction integration.

Cross-references papers against retraction databases to provide context about
whether a paper or its authors have been associated with retractions.
Uses the Crossref API to check retraction status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_CROSSREF_API = "https://api.crossref.org"
_USER_AGENT = "snoopy/0.1 (academic research tool; mailto:research@example.com)"
_TIMEOUT = 30.0


@dataclass
class RetractionInfo:
    """Information about a paper's retraction status."""

    is_retracted: bool
    retraction_doi: str | None = None
    retraction_date: str | None = None
    retraction_reason: str | None = None
    has_expression_of_concern: bool = False
    has_correction: bool = False
    update_type: str | None = None


@dataclass
class AuthorRetractionHistory:
    """Retraction history for a specific author."""

    author_name: str
    total_retractions: int = 0
    total_expressions_of_concern: int = 0
    retracted_dois: list[str] | None = None


async def check_retraction_status(doi: str) -> RetractionInfo:
    """Check whether a paper has been retracted via the Crossref API.

    Args:
        doi: The DOI of the paper to check.

    Returns:
        RetractionInfo with retraction status and details.
    """
    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    ) as client:
        try:
            # Check the paper's Crossref metadata for update-to relations
            resp = await client.get(f"{_CROSSREF_API}/works/{doi}")
            if resp.status_code != 200:
                return RetractionInfo(is_retracted=False)

            data = resp.json().get("message", {})

            # Check for retraction notices in the "relation" field
            updates = data.get("update-to", [])
            for update in updates:
                update_type = update.get("type", "")
                if update_type == "retraction":
                    return RetractionInfo(
                        is_retracted=True,
                        retraction_doi=update.get("DOI"),
                        retraction_date=update.get("updated", {}).get("date-time"),
                        update_type="retraction",
                    )
                elif update_type == "expression_of_concern":
                    return RetractionInfo(
                        is_retracted=False,
                        has_expression_of_concern=True,
                        retraction_doi=update.get("DOI"),
                        update_type="expression_of_concern",
                    )
                elif update_type == "correction":
                    return RetractionInfo(
                        is_retracted=False,
                        has_correction=True,
                        retraction_doi=update.get("DOI"),
                        update_type="correction",
                    )

            # Also check if this paper IS a retraction notice
            item_type = data.get("type", "")
            if item_type == "retraction":
                return RetractionInfo(
                    is_retracted=True,
                    update_type="retraction_notice",
                )

        except httpx.HTTPError as e:
            logger.warning("Crossref API error for DOI %s: %s", doi, e)
        except Exception as e:
            logger.warning("Error checking retraction status for %s: %s", doi, e)

    return RetractionInfo(is_retracted=False)


async def check_author_retractions(
    author_name: str, limit: int = 50
) -> AuthorRetractionHistory:
    """Search for retractions associated with an author name.

    Args:
        author_name: The author's name to search for.
        limit: Maximum number of results to check.

    Returns:
        AuthorRetractionHistory with retraction counts and DOIs.
    """
    retracted_dois = []
    eoc_count = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    ) as client:
        try:
            # Search for retraction notices by this author
            params = {
                "query.author": author_name,
                "filter": "type:retraction",
                "rows": limit,
                "select": "DOI,title,type",
            }
            resp = await client.get(f"{_CROSSREF_API}/works", params=params)
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                for item in items:
                    doi = item.get("DOI", "")
                    if doi:
                        retracted_dois.append(doi)

            # Also search for expressions of concern
            params["filter"] = "type:expression-of-concern"
            resp = await client.get(f"{_CROSSREF_API}/works", params=params)
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                eoc_count = len(items)

        except httpx.HTTPError as e:
            logger.warning("Crossref API error for author %s: %s", author_name, e)
        except Exception as e:
            logger.warning("Error checking author retractions for %s: %s", author_name, e)

    return AuthorRetractionHistory(
        author_name=author_name,
        total_retractions=len(retracted_dois),
        total_expressions_of_concern=eoc_count,
        retracted_dois=retracted_dois if retracted_dois else None,
    )
