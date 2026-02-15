"""Tests for the webhook notification service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from snoopy.notifications.webhook import WebhookNotifier, _is_safe_url, _resolve_safe_ips


class TestIsSafeUrl:
    def test_rejects_http_scheme(self):
        assert _is_safe_url("http://example.com/hook") is False

    def test_rejects_localhost(self):
        assert _is_safe_url("https://localhost/hook") is False

    def test_rejects_loopback(self):
        assert _is_safe_url("https://127.0.0.1/hook") is False

    def test_rejects_link_local(self):
        assert _is_safe_url("https://169.254.169.254/metadata") is False

    def test_rejects_empty_url(self):
        assert _is_safe_url("") is False

    def test_rejects_no_hostname(self):
        assert _is_safe_url("https:///path") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_accepts_public_https(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        assert _is_safe_url("https://example.com/hook") is True


class TestSSRFProtection:
    """Tests for private/internal IP rejection (SSRF protection)."""

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_private_10_network(self, mock_getaddrinfo):
        """10.0.0.0/8 private range should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        assert _is_safe_url("https://internal.example.com/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_private_172_network(self, mock_getaddrinfo):
        """172.16.0.0/12 private range should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("172.16.0.1", 443)),
        ]
        assert _is_safe_url("https://internal.example.com/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_private_192_network(self, mock_getaddrinfo):
        """192.168.0.0/16 private range should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 443)),
        ]
        assert _is_safe_url("https://internal.example.com/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_aws_metadata_ip(self, mock_getaddrinfo):
        """AWS metadata endpoint IP (169.254.169.254) should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 443)),
        ]
        assert _is_safe_url("https://metadata.internal/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_loopback_127(self, mock_getaddrinfo):
        """127.0.0.1 loopback should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        assert _is_safe_url("https://loopback.example.com/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_ipv6_loopback(self, mock_getaddrinfo):
        """IPv6 loopback (::1) should be rejected."""
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::1", 443, 0, 0)),
        ]
        assert _is_safe_url("https://ipv6-loopback.example.com/hook") is False

    def test_rejects_ftp_scheme(self):
        """Non-HTTPS schemes should be rejected."""
        assert _is_safe_url("ftp://example.com/hook") is False

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_rejects_mixed_public_and_private(self, mock_getaddrinfo):
        """If any resolved IP is private, the URL should be rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        assert _is_safe_url("https://mixed.example.com/hook") is False


class TestResolveSafeIps:
    """Tests for _resolve_safe_ips which returns pinned IPs."""

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_returns_ips_for_public_url(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        ips = _resolve_safe_ips("https://example.com/hook")
        assert ips == ["93.184.216.34"]

    @patch("snoopy.notifications.webhook.socket.getaddrinfo")
    def test_returns_empty_for_private_ip(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        ips = _resolve_safe_ips("https://internal.example.com/hook")
        assert ips == []

    def test_returns_empty_for_bad_scheme(self):
        ips = _resolve_safe_ips("http://example.com/hook")
        assert ips == []


class TestWebhookRegistration:
    def test_register_url(self):
        notifier = WebhookNotifier()
        with patch(
            "snoopy.notifications.webhook._resolve_safe_ips", return_value=["93.184.216.34"]
        ):
            notifier.register_url("https://example.com/hook")
        assert "https://example.com/hook" in notifier.urls
        assert notifier._pinned_ips["https://example.com/hook"] == ["93.184.216.34"]

    def test_register_unsafe_url_raises(self):
        notifier = WebhookNotifier()
        with pytest.raises(ValueError, match="Rejected webhook URL"):
            notifier.register_url("http://localhost/hook")

    def test_unregister_url(self):
        notifier = WebhookNotifier()
        notifier.urls.add("https://example.com/hook")
        notifier.unregister_url("https://example.com/hook")
        assert "https://example.com/hook" not in notifier.urls

    def test_unregister_nonexistent_url(self):
        notifier = WebhookNotifier()
        notifier.unregister_url("https://example.com/nonexistent")
        assert len(notifier.urls) == 0


class TestWebhookDelivery:
    @pytest.mark.asyncio
    async def test_successful_delivery(self):
        notifier = WebhookNotifier(max_retries=1, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}
        notifier._pinned_ips = {"https://example.com/hook": ["93.184.216.34"]}

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await notifier.notify({"paper_id": "123", "risk": "high"})

        assert results["https://example.com/hook"] is True

    @pytest.mark.asyncio
    async def test_failed_delivery_retries(self):
        notifier = WebhookNotifier(max_retries=2, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}
        notifier._pinned_ips = {"https://example.com/hook": ["93.184.216.34"]}

        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 500
        mock_response_ok = AsyncMock()
        mock_response_ok.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [mock_response_fail, mock_response_ok]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await notifier.notify({"paper_id": "123"})

        assert results["https://example.com/hook"] is True

    @pytest.mark.asyncio
    async def test_delivery_exhausts_retries(self):
        notifier = WebhookNotifier(max_retries=2, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}
        notifier._pinned_ips = {"https://example.com/hook": ["93.184.216.34"]}

        mock_response = AsyncMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await notifier.notify({"paper_id": "123"})

        assert results["https://example.com/hook"] is False

    @pytest.mark.asyncio
    async def test_delivery_handles_http_error(self):
        notifier = WebhookNotifier(max_retries=1, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}
        notifier._pinned_ips = {"https://example.com/hook": ["93.184.216.34"]}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPError("connection failed")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await notifier.notify({"paper_id": "123"})

        assert results["https://example.com/hook"] is False

    @pytest.mark.asyncio
    async def test_no_urls_returns_empty(self):
        notifier = WebhookNotifier()
        results = await notifier.notify({"paper_id": "123"})
        assert results == {}

    @pytest.mark.asyncio
    async def test_multiple_urls(self):
        notifier = WebhookNotifier(max_retries=1, retry_delay=0.01)
        notifier.urls = {"https://a.com/hook", "https://b.com/hook"}
        notifier._pinned_ips = {
            "https://a.com/hook": ["1.2.3.4"],
            "https://b.com/hook": ["5.6.7.8"],
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await notifier.notify({"paper_id": "123"})

        assert len(results) == 2
        assert all(v is True for v in results.values())

    @pytest.mark.asyncio
    async def test_delivery_rejects_unregistered_url(self):
        """A URL without pinned IPs should fail delivery."""
        notifier = WebhookNotifier(max_retries=1, retry_delay=0.01)
        notifier.urls = {"https://unknown.com/hook"}
        # No pinned IPs set

        results = await notifier.notify({"paper_id": "123"})
        assert results["https://unknown.com/hook"] is False


class TestExponentialBackoff:
    @pytest.mark.asyncio
    async def test_backoff_increases_delay(self):
        """Verify that retry delay doubles each attempt."""
        notifier = WebhookNotifier(max_retries=3, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}
        notifier._pinned_ips = {"https://example.com/hook": ["93.184.216.34"]}

        mock_response = AsyncMock()
        mock_response.status_code = 500

        delays_observed = []

        async def _mock_sleep(delay):
            delays_observed.append(delay)

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            patch("asyncio.sleep", side_effect=_mock_sleep),
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await notifier.notify({"paper_id": "123"})

        # With max_retries=3, we get 2 sleep calls (between attempts)
        assert len(delays_observed) == 2
        # Backoff: delay * 2^0 = 0.01, delay * 2^1 = 0.02
        assert delays_observed[0] == pytest.approx(0.01)
        assert delays_observed[1] == pytest.approx(0.02)
