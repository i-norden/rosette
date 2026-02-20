"""Unpaywall API client for resolving open-access PDF URLs from DOIs."""

from __future__ import annotations

import logging
import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unpaywall.org/v2"
DEFAULT_TIMEOUT = 30.0


async def get_pdf_url(
    doi: str,
    email: str,
) -> str | None:
    """Look up an open-access PDF URL for a given DOI via Unpaywall.

    Args:
        doi: The DOI of the paper (e.g. "10.1038/nature12373").
        email: Your email address, required by the Unpaywall API.

    Returns:
        URL string pointing to an OA PDF, or None if no OA version found.
    """
    if not doi or not email:
        logger.warning("get_pdf_url called with empty doi or email")
        return None

    # Strip any DOI URL prefix
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/") :]
    elif doi.startswith("http://dx.doi.org/"):
        doi = doi[len("http://dx.doi.org/") :]

    url = f"{BASE_URL}/{doi}"
    params = {"email": email}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.debug("Unpaywall: DOI not found: %s", doi)
        else:
            logger.error(
                "Unpaywall HTTP error %s for DOI %s: %s",
                exc.response.status_code,
                doi,
                exc.response.text[:500],
            )
        return None
    except httpx.HTTPError as exc:
        logger.error("Unpaywall request failed for DOI %s: %s", doi, exc)
        return None

    # Try the best OA location first
    best_oa = data.get("best_oa_location")
    if best_oa:
        pdf_url = best_oa.get("url_for_pdf")
        if pdf_url:
            return pdf_url
        # Fall back to landing page URL if no direct PDF
        landing_url = best_oa.get("url_for_landing_page")
        if landing_url:
            logger.debug(
                "Unpaywall: no direct PDF for DOI %s, landing page: %s",
                doi,
                landing_url,
            )

    # Check all OA locations for a PDF URL
    oa_locations = data.get("oa_locations", [])
    for location in oa_locations:
        pdf_url = location.get("url_for_pdf")
        if pdf_url:
            return pdf_url

    logger.debug("Unpaywall: no OA PDF found for DOI %s", doi)
    return None
