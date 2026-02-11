"""Tests for CrossRef discovery provider."""

from __future__ import annotations

import pytest

from snoopy.discovery.crossref import _normalize_doi, check_retraction, get_work


class TestNormalizeDoi:
    def test_strips_https_prefix(self) -> None:
        assert _normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"

    def test_strips_http_prefix(self) -> None:
        assert _normalize_doi("http://doi.org/10.1234/test") == "10.1234/test"

    def test_strips_dx_prefix(self) -> None:
        assert _normalize_doi("https://dx.doi.org/10.1234/test") == "10.1234/test"

    def test_strips_whitespace(self) -> None:
        assert _normalize_doi("  10.1234/test  ") == "10.1234/test"

    def test_no_prefix(self) -> None:
        assert _normalize_doi("10.1234/test") == "10.1234/test"


class TestGetWork:
    @pytest.mark.asyncio
    async def test_returns_none_on_empty_doi(self) -> None:
        result = await get_work("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_work("10.1234/test")
        assert result is None


class TestCheckRetraction:
    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await check_retraction("10.1234/test")
        assert result is False
