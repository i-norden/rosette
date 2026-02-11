"""Tests for Semantic Scholar discovery provider."""

from __future__ import annotations

import pytest

from snoopy.discovery.semantic_scholar import (
    S2PaperResult,
    _parse_author,
    _parse_paper,
    get_author,
    get_paper,
)


class TestParseAuthor:
    def test_parse_full_author(self) -> None:
        data = {
            "authorId": "12345",
            "name": "Jane Doe",
            "hIndex": 42,
            "citationCount": 5000,
            "paperCount": 100,
        }
        author = _parse_author(data)
        assert author.author_id == "12345"
        assert author.name == "Jane Doe"
        assert author.h_index == 42
        assert author.citation_count == 5000
        assert author.paper_count == 100

    def test_parse_minimal_author(self) -> None:
        data = {}
        author = _parse_author(data)
        assert author.name == "Unknown"
        assert author.author_id is None


class TestParsePaper:
    def test_parse_full_paper(self) -> None:
        data = {
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1234/test"},
            "title": "A Great Paper",
            "abstract": "We did stuff.",
            "year": 2024,
            "citationCount": 100,
            "influentialCitationCount": 10,
            "authors": [
                {"authorId": "A1", "name": "Author One", "hIndex": 20},
            ],
        }
        paper = _parse_paper(data)
        assert isinstance(paper, S2PaperResult)
        assert paper.paper_id == "abc123"
        assert paper.doi == "10.1234/test"
        assert paper.title == "A Great Paper"
        assert paper.year == 2024
        assert paper.citation_count == 100
        assert paper.influential_citation_count == 10
        assert len(paper.authors) == 1

    def test_parse_paper_no_external_ids(self) -> None:
        data = {"paperId": "xyz"}
        paper = _parse_paper(data)
        assert paper.doi is None


class TestGetPaper:
    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_paper("10.1234/test")
        assert result is None


class TestGetAuthor:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_id(self) -> None:
        result = await get_author("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, monkeypatch) -> None:
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        result = await get_author("12345")
        assert result is None
