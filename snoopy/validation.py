"""Shared validation utilities."""

from __future__ import annotations

import re

_DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+$")

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:", "DOI:")


def validate_doi(doi: str) -> str:
    """Validate and normalize a DOI string.

    Strips common URL prefixes and checks the format matches the
    standard DOI pattern (10.XXXX/...).

    Args:
        doi: The DOI string to validate.

    Returns:
        The normalized DOI string.

    Raises:
        ValueError: If the DOI format is invalid.
    """
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    doi = doi.strip()
    if not _DOI_PATTERN.match(doi):
        raise ValueError(f"Invalid DOI format: {doi!r}. Expected 10.XXXX/...")
    return doi
