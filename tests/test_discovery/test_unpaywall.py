"""Tests for Unpaywall discovery provider."""

from __future__ import annotations

import pytest

from snoopy.discovery.unpaywall import get_pdf_url


class TestGetPdfUrl:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_doi(self) -> None:
        result = await get_pdf_url("", "test@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_email(self) -> None:
        result = await get_pdf_url("10.1234/test", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("10.1234/test", "test@example.com")
        assert result is None
