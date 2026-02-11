"""Webhook notification service for delivering analysis results."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def _is_safe_url(url: str) -> bool:
    """Validate that a URL does not point to private/internal IP ranges.

    Returns True if the URL is safe to request, False otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("https",):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return False
    except (socket.gaierror, ValueError):
        return False

    return True

# Default retry configuration
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 2.0  # seconds
_DEFAULT_TIMEOUT = 10.0  # seconds


@dataclass
class WebhookNotifier:
    """Manages webhook URL registrations and delivers finding notifications.

    Attributes:
        urls: Set of registered webhook URLs.
        max_retries: Maximum number of delivery attempts per URL.
        retry_delay: Base delay in seconds between retries (exponential backoff).
        timeout: HTTP request timeout in seconds.
    """

    urls: set[str] = field(default_factory=set)
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_delay: float = _DEFAULT_RETRY_DELAY
    timeout: float = _DEFAULT_TIMEOUT

    def register_url(self, url: str) -> None:
        """Register a webhook URL to receive notifications.

        Args:
            url: The URL to register.

        Raises:
            ValueError: If the URL is not a safe HTTPS URL or resolves
                to a private/internal IP address.
        """
        if not _is_safe_url(url):
            raise ValueError(
                f"Rejected webhook URL {url!r}: only HTTPS URLs "
                "resolving to public IP addresses are allowed"
            )
        self.urls.add(url)
        logger.info("Registered webhook URL: %s", url)

    def unregister_url(self, url: str) -> None:
        """Remove a webhook URL from the notification list.

        Args:
            url: The URL to unregister.
        """
        self.urls.discard(url)
        logger.info("Unregistered webhook URL: %s", url)

    async def notify(self, payload: dict) -> dict[str, bool]:
        """POST a findings summary JSON payload to all registered webhook URLs.

        Includes retry logic with exponential backoff for failed deliveries.

        Args:
            payload: The JSON-serializable payload to send.

        Returns:
            A dict mapping each URL to True (success) or False (failure).
        """
        results: dict[str, bool] = {}
        tasks = [self._deliver(url, payload) for url in self.urls]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for url, outcome in zip(self.urls, outcomes):
            if isinstance(outcome, Exception):
                results[url] = False
                logger.error("Webhook delivery to %s raised exception: %s", url, outcome)
            else:
                results[url] = outcome

        return results

    async def _deliver(self, url: str, payload: dict) -> bool:
        """Deliver a payload to a single URL with retry logic.

        Uses exponential backoff: delay * 2^attempt for each retry.

        Args:
            url: The target webhook URL.
            payload: The JSON payload to POST.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(url, json=payload)
                    if 200 <= response.status_code < 300:
                        logger.info(
                            "Webhook delivery to %s succeeded (status %d)",
                            url,
                            response.status_code,
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook delivery to %s returned status %d (attempt %d/%d)",
                            url,
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                        )
                except httpx.HTTPError as e:
                    logger.warning(
                        "Webhook delivery to %s failed (attempt %d/%d): %s",
                        url,
                        attempt + 1,
                        self.max_retries,
                        e,
                    )

                # Exponential backoff before retry
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        logger.error(
            "Webhook delivery to %s failed after %d attempts", url, self.max_retries
        )
        return False
