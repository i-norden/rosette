"""CrossRef API client for fetching work metadata and checking retractions."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org"
DEFAULT_TIMEOUT = 30.0


def _normalize_doi(doi: str) -> str:
    """Strip common DOI URL prefixes to get the raw DOI."""
    doi = doi.strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


async def get_work(
    doi: str,
    mailto: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Fetch work metadata from CrossRef by DOI.

    Args:
        doi: The DOI of the work.
        mailto: Optional email for the polite pool (faster responses).

    Returns:
        Metadata dict from the CrossRef "message" field, or None on failure.
        The dict contains keys like "title", "author", "container-title",
        "published-print", "is-referenced-by-count", "ISSN", "type",
        "update-to", etc.
    """
    doi = _normalize_doi(doi)
    if not doi:
        logger.warning("get_work called with empty DOI")
        return None

    url = f"{BASE_URL}/works/{doi}"
    headers: dict[str, str] = {"Accept": "application/json"}
    params: dict[str, str] = {}
    if mailto:
        params["mailto"] = mailto

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.debug("CrossRef: DOI not found: %s", doi)
        else:
            logger.error(
                "CrossRef HTTP error %s for DOI %s: %s",
                exc.response.status_code,
                doi,
                exc.response.text[:500],
            )
        return None
    except httpx.HTTPError as exc:
        logger.error("CrossRef request failed for DOI %s: %s", doi, exc)
        return None

    message = data.get("message")
    if not isinstance(message, dict):
        logger.warning("CrossRef: unexpected response structure for DOI %s", doi)
        return None

    return message


async def check_retraction(
    doi: str,
    mailto: Optional[str] = None,
) -> bool:
    """Check if a work has been retracted via CrossRef update-to metadata.

    A work is considered retracted if it has an "update-to" entry with
    type "retraction".

    Args:
        doi: The DOI of the work to check.
        mailto: Optional email for polite pool.

    Returns:
        True if a retraction notice is found, False otherwise (including
        on API errors).
    """
    work = await get_work(doi, mailto=mailto)
    if work is None:
        return False

    # Check the "update-to" field for retraction entries
    updates = work.get("update-to", [])
    if not isinstance(updates, list):
        return False

    for update in updates:
        if not isinstance(update, dict):
            continue
        update_type = update.get("type", "")
        if update_type.lower() == "retraction":
            logger.info("CrossRef: retraction found for DOI %s", doi)
            return True

    # Also check if this work itself is a retraction notice pointing back
    # (some publishers use "relation" instead)
    relation = work.get("relation", {})
    if isinstance(relation, dict):
        is_retraction_of = relation.get("is-retraction-of", [])
        if is_retraction_of:
            logger.info(
                "CrossRef: work DOI %s is itself a retraction notice", doi
            )
            return True

    return False
