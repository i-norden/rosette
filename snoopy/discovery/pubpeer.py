"""PubPeer integration for checking existing community concerns.

Queries the PubPeer API to determine whether a paper already has public
comments or concerns raised by the scientific community.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_PUBPEER_API = "https://pubpeer.com/api/v1"
_USER_AGENT = "snoopy/0.1 (academic research tool)"
_TIMEOUT = 15.0


@dataclass
class PubPeerComment:
    """A single PubPeer comment."""

    author: str
    content_snippet: str
    created_at: str
    url: str


@dataclass
class PubPeerResult:
    """Result of PubPeer lookup for a paper."""

    has_comments: bool
    total_comments: int = 0
    comments: list[PubPeerComment] = field(default_factory=list)
    pubpeer_url: str | None = None
    error: str | None = None


async def check_pubpeer(doi: str) -> PubPeerResult:
    """Check PubPeer for comments on a paper.

    Args:
        doi: The DOI of the paper to check.

    Returns:
        PubPeerResult with comment information.
    """
    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    ) as client:
        try:
            resp = await client.get(
                f"{_PUBPEER_API}/publications",
                params={"doi": doi},
            )

            if resp.status_code == 404:
                return PubPeerResult(has_comments=False)

            if resp.status_code != 200:
                return PubPeerResult(
                    has_comments=False,
                    error=f"PubPeer API returned status {resp.status_code}",
                )

            data = resp.json()
            publications = data.get("data", [])

            if not publications:
                return PubPeerResult(has_comments=False)

            pub = publications[0] if isinstance(publications, list) else publications
            total_comments = pub.get("total_comments", 0)
            pubpeer_url = pub.get("url")

            comments = []
            raw_comments = pub.get("comments", [])
            for c in raw_comments[:10]:  # Limit to 10 most recent
                comments.append(PubPeerComment(
                    author=c.get("author", "Anonymous"),
                    content_snippet=str(c.get("content", ""))[:300],
                    created_at=c.get("created_at", ""),
                    url=c.get("url", ""),
                ))

            return PubPeerResult(
                has_comments=total_comments > 0,
                total_comments=total_comments,
                comments=comments,
                pubpeer_url=pubpeer_url,
            )

        except httpx.HTTPError as e:
            logger.warning("PubPeer API error for DOI %s: %s", doi, e)
            return PubPeerResult(
                has_comments=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            logger.warning("Error checking PubPeer for %s: %s", doi, e)
            return PubPeerResult(
                has_comments=False,
                error=str(e),
            )
