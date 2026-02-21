"""Tests for Unpaywall discovery provider."""

from __future__ import annotations

import pytest

from rosette.discovery.unpaywall import get_pdf_url


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


class TestGetPdfUrlHappyPath:
    @pytest.mark.asyncio
    async def test_returns_best_oa_pdf_url(self, monkeypatch) -> None:
        """Happy path: returns best_oa_location.url_for_pdf."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "best_oa_location": {
                    "url_for_pdf": "https://example.com/paper.pdf",
                    "url_for_landing_page": "https://example.com/paper",
                },
                "oa_locations": [],
            },
            request=httpx.Request("GET", "https://api.unpaywall.org/v2/10.1234/test"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("10.1234/test", "test@example.com")
        assert result == "https://example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_falls_back_to_oa_locations(self, monkeypatch) -> None:
        """Falls back to oa_locations when best_oa_location has no PDF URL."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "best_oa_location": {
                    "url_for_landing_page": "https://example.com/paper",
                },
                "oa_locations": [
                    {"url_for_pdf": "https://fallback.com/paper.pdf"},
                ],
            },
            request=httpx.Request("GET", "https://api.unpaywall.org/v2/10.1234/test"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("10.1234/test", "test@example.com")
        assert result == "https://fallback.com/paper.pdf"


class TestGetPdfUrlDoiPrefixStripping:
    @pytest.mark.asyncio
    async def test_strips_https_doi_org_prefix(self, monkeypatch) -> None:
        """DOI with https://doi.org/ prefix is handled correctly."""
        import httpx

        called_urls = []

        async def _mock_get(self, url, **kwargs):
            called_urls.append(str(url))
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"},
                    "oa_locations": [],
                },
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("https://doi.org/10.1234/test", "test@example.com")
        assert result == "https://example.com/paper.pdf"
        # Verify the prefix was stripped (URL should contain the raw DOI, not the full URL)
        assert any("10.1234/test" in u for u in called_urls)

    @pytest.mark.asyncio
    async def test_strips_dx_doi_org_prefix(self, monkeypatch) -> None:
        """DOI with http://dx.doi.org/ prefix is handled correctly."""
        import httpx

        async def _mock_get(self, url, **kwargs):
            return httpx.Response(
                200,
                json={
                    "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"},
                    "oa_locations": [],
                },
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("http://dx.doi.org/10.1234/test", "test@example.com")
        assert result == "https://example.com/paper.pdf"


class TestGetPdfUrl404:
    @pytest.mark.asyncio
    async def test_returns_none_on_404(self, monkeypatch) -> None:
        """404 response returns None without raising."""
        import httpx

        async def _mock_get(self, url, **kwargs):
            response = httpx.Response(
                404,
                request=httpx.Request("GET", url),
            )
            raise httpx.HTTPStatusError("Not found", request=response.request, response=response)

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("10.9999/nonexistent", "test@example.com")
        assert result is None


class TestGetPdfUrlNoOALocation:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_oa_available(self, monkeypatch) -> None:
        """Paper exists but has no OA locations -> returns None."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={"best_oa_location": None, "oa_locations": []},
            request=httpx.Request("GET", "https://api.unpaywall.org/v2/10.1234/test"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_pdf_url("10.1234/test", "test@example.com")
        assert result is None
