"""Tests for CrossRef discovery provider."""

from __future__ import annotations

import pytest

from rosette.discovery.crossref import _normalize_doi, check_retraction, get_work


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


class TestGetWorkHappyPath:
    @pytest.mark.asyncio
    async def test_returns_message_on_success(self, monkeypatch) -> None:
        """Successful API call returns the message dict."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": "10.1234/test",
                    "title": ["Test Paper"],
                    "author": [{"given": "Alice", "family": "Smith"}],
                    "is-referenced-by-count": 42,
                    "type": "journal-article",
                },
            },
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1234/test"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_work("10.1234/test")
        assert result is not None
        assert result["DOI"] == "10.1234/test"
        assert result["is-referenced-by-count"] == 42

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self, monkeypatch) -> None:
        """404 response returns None without raising."""
        import httpx

        async def _mock_get(self, url, **kwargs):
            response = httpx.Response(404, request=httpx.Request("GET", url))
            raise httpx.HTTPStatusError("Not found", request=response.request, response=response)

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_work("10.9999/nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_mailto_param(self, monkeypatch) -> None:
        """mailto parameter is passed to the request."""
        import httpx

        captured_kwargs: list[dict] = []

        async def _mock_get(self, url, **kwargs):
            captured_kwargs.append(kwargs)
            return httpx.Response(
                200,
                json={"status": "ok", "message": {"DOI": "10.1234/test"}},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        await get_work("10.1234/test", mailto="test@example.com")
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("params", {}).get("mailto") == "test@example.com"


class TestCheckRetractionHappyPath:
    @pytest.mark.asyncio
    async def test_returns_true_for_retracted_paper(self, monkeypatch) -> None:
        """Paper with retraction in update-to returns True."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": "10.1234/retracted",
                    "update-to": [
                        {"type": "retraction", "DOI": "10.1234/retraction-notice"},
                    ],
                },
            },
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1234/retracted"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await check_retraction("10.1234/retracted")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_retraction_notice(self, monkeypatch) -> None:
        """Paper that IS a retraction notice (via relation) returns True."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": "10.1234/retraction-notice",
                    "update-to": [],
                    "relation": {
                        "is-retraction-of": [{"id": "10.1234/original"}],
                    },
                },
            },
            request=httpx.Request(
                "GET", "https://api.crossref.org/works/10.1234/retraction-notice"
            ),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await check_retraction("10.1234/retraction-notice")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_clean_paper(self, monkeypatch) -> None:
        """Paper with no retractions returns False."""
        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": "10.1234/clean",
                    "update-to": [],
                    "relation": {},
                },
            },
            request=httpx.Request("GET", "https://api.crossref.org/works/10.1234/clean"),
        )

        async def _mock_get(self, url, **kwargs):
            return mock_response

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await check_retraction("10.1234/clean")
        assert result is False
