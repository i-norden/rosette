"""Tests for the webhook notification service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from snoopy.notifications.webhook import WebhookNotifier, _is_safe_url


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


class TestWebhookRegistration:
    def test_register_url(self):
        notifier = WebhookNotifier()
        with patch("snoopy.notifications.webhook._is_safe_url", return_value=True):
            notifier.register_url("https://example.com/hook")
        assert "https://example.com/hook" in notifier.urls

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


class TestExponentialBackoff:
    @pytest.mark.asyncio
    async def test_backoff_increases_delay(self):
        """Verify that retry delay doubles each attempt."""
        notifier = WebhookNotifier(max_retries=3, retry_delay=0.01)
        notifier.urls = {"https://example.com/hook"}

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
