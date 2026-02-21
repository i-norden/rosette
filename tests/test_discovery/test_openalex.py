"""Tests for OpenAlex discovery provider."""

from __future__ import annotations

import pytest

from rosette.discovery.openalex import (
    PaperResult,
    _parse_work,
    _reconstruct_abstract,
    search_works,
)


class TestReconstructAbstract:
    def test_empty_index(self) -> None:
        assert _reconstruct_abstract({}) == ""

    def test_simple_reconstruction(self) -> None:
        inverted = {"hello": [0], "world": [1]}
        assert _reconstruct_abstract(inverted) == "hello world"

    def test_order_preserved(self) -> None:
        inverted = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
        result = _reconstruct_abstract(inverted)
        assert result == "the cat sat the mat"


class TestParseWork:
    def test_parse_minimal_work(self) -> None:
        work = {"id": "W123", "title": "Test Paper", "cited_by_count": 42}
        result = _parse_work(work)
        assert isinstance(result, PaperResult)
        assert result.openalex_id == "W123"
        assert result.title == "Test Paper"
        assert result.citation_count == 42

    def test_parse_doi_strips_prefix(self) -> None:
        work = {"id": "W456", "doi": "https://doi.org/10.1234/test"}
        result = _parse_work(work)
        assert result.doi == "10.1234/test"

    def test_parse_work_with_authors(self) -> None:
        work = {
            "id": "W789",
            "authorships": [
                {
                    "author": {"display_name": "Jane Doe", "id": "A1", "orcid": "0000-0001"},
                    "institutions": [{"display_name": "MIT", "ror": "https://ror.org/042nb2s44"}],
                },
            ],
        }
        result = _parse_work(work)
        assert len(result.authors) == 1
        assert result.authors[0].name == "Jane Doe"
        assert result.authors[0].institution_name == "MIT"

    def test_parse_work_with_oa_url(self) -> None:
        work = {
            "id": "W101",
            "open_access": {"oa_url": "https://example.com/paper.pdf"},
        }
        result = _parse_work(work)
        assert result.open_access_url == "https://example.com/paper.pdf"


class TestSearchWorks:
    @pytest.mark.asyncio
    async def test_search_returns_empty_on_http_error(self, monkeypatch) -> None:
        """search_works returns empty list on HTTP error."""
        import httpx

        async def _mock_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        results = await search_works("test query")
        assert results == []
