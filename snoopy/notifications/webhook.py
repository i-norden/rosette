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


def _resolve_safe_ips(url: str) -> list[str]:
    """Validate that a URL does not point to private/internal IP ranges.

    Returns a list of safe resolved IP addresses, or an empty list if the URL
    is unsafe or cannot be resolved.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return []

    if parsed.scheme not in ("https",):
        return []

    hostname = parsed.hostname
    if not hostname:
        return []

    try:
        infos = socket.getaddrinfo(hostname, None)
        safe_ips: list[str] = []
        for info in infos:
            ip_str = str(info[4][0])
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return []
            safe_ips.append(ip_str)
        return safe_ips
    except (socket.gaierror, ValueError):
        return []


def _is_safe_url(url: str) -> bool:
    """Validate that a URL does not point to private/internal IP ranges.

    Returns True if the URL is safe to request, False otherwise.
    """
    return len(_resolve_safe_ips(url)) > 0


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
    _pinned_ips: dict[str, list[str]] = field(default_factory=dict, repr=False)
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_delay: float = _DEFAULT_RETRY_DELAY
    timeout: float = _DEFAULT_TIMEOUT

    def register_url(self, url: str) -> None:
        """Register a webhook URL to receive notifications.

        Resolves the URL's hostname and pins the resulting IPs to prevent
        DNS rebinding attacks between registration and delivery.

        Args:
            url: The URL to register.

        Raises:
            ValueError: If the URL is not a safe HTTPS URL or resolves
                to a private/internal IP address.
        """
        safe_ips = _resolve_safe_ips(url)
        if not safe_ips:
            raise ValueError(
                f"Rejected webhook URL {url!r}: only HTTPS URLs "
                "resolving to public IP addresses are allowed"
            )
        self.urls.add(url)
        self._pinned_ips[url] = safe_ips
        logger.info("Registered webhook URL: %s (pinned IPs: %s)", url, safe_ips)

    def unregister_url(self, url: str) -> None:
        """Remove a webhook URL from the notification list.

        Args:
            url: The URL to unregister.
        """
        self.urls.discard(url)
        self._pinned_ips.pop(url, None)
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
                results[url] = bool(outcome)

        return results

    async def _deliver(self, url: str, payload: dict) -> bool:
        """Deliver a payload to a single URL with retry logic.

        Uses the IP address pinned at registration time to prevent DNS
        rebinding attacks.  Uses exponential backoff: delay * 2^attempt
        for each retry.

        Args:
            url: The target webhook URL.
            payload: The JSON payload to POST.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        pinned = self._pinned_ips.get(url)
        if not pinned:
            logger.error("Webhook URL %s has no pinned IPs (not properly registered)", url)
            return False

        # Re-validate the pinned IPs are still safe (they shouldn't change,
        # but defend against manual mutation of _pinned_ips)
        for ip_str in pinned:
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                logger.error("Pinned IP %s for webhook URL %s is not public", ip_str, url)
                return False

        # Build the delivery URL using the pinned IP and set the Host header
        parsed = urlparse(url)
        original_hostname = parsed.hostname or ""
        # Reconstruct netloc with pinned IP, preserving port if present
        if parsed.port:
            new_netloc = f"{pinned[0]}:{parsed.port}"
        else:
            new_netloc = pinned[0]
        delivery_url = parsed._replace(netloc=new_netloc).geturl()
        headers = {"Host": original_hostname}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(delivery_url, json=payload, headers=headers)
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
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        logger.error("Webhook delivery to %s failed after %d attempts", url, self.max_retries)
        return False
