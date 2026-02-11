"""Tests for PubPeer discovery provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from snoopy.discovery.pubpeer import check_pubpeer


def _make_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    """Create a mock httpx.Response with the given status code and JSON data."""
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _make_client(get_return=None, get_side_effect=None) -> AsyncMock:
    """Create a mock httpx.AsyncClient as an async context manager."""
    mock_client = AsyncMock()
    if get_return is not None:
        mock_client.get = AsyncMock(return_value=get_return)
    if get_side_effect is not None:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    return mock_client


class TestCheckPubpeer:
    @pytest.mark.asyncio
    async def test_paper_with_comments(self):
        response = _make_response(200, {
            "data": [
                {
                    "total_comments": 3,
                    "url": "https://pubpeer.com/publications/ABC123",
                    "comments": [
                        {
                            "author": "Reviewer A",
                            "content": "Concerns about Figure 2",
                            "created_at": "2024-01-15",
                            "url": "https://pubpeer.com/comment/1",
                        },
                        {
                            "author": "Anonymous",
                            "content": "Data inconsistency noted",
                            "created_at": "2024-02-01",
                            "url": "https://pubpeer.com/comment/2",
                        },
                    ],
                }
            ]
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/test.paper")

        assert result.has_comments is True
        assert result.total_comments == 3
        assert len(result.comments) == 2
        assert result.comments[0].author == "Reviewer A"
        assert result.pubpeer_url == "https://pubpeer.com/publications/ABC123"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_paper_with_no_comments(self):
        response = _make_response(200, {"data": []})
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/clean.paper")

        assert result.has_comments is False
        assert result.total_comments == 0
        assert result.comments == []

    @pytest.mark.asyncio
    async def test_paper_not_found(self):
        response = _make_response(404)
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/unknown")

        assert result.has_comments is False

    @pytest.mark.asyncio
    async def test_api_error_status(self):
        response = _make_response(500)
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/server.error")

        assert result.has_comments is False
        assert "status 500" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self):
        mock_client = _make_client(
            get_side_effect=httpx.HTTPError("connection failed")
        )

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/conn.fail")

        assert result.has_comments is False
        assert "HTTP error" in result.error

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        mock_client = _make_client(get_side_effect=ValueError("unexpected"))

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/unexpected")

        assert result.has_comments is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_comments_truncated_to_10(self):
        comments = [
            {
                "author": f"Author {i}",
                "content": f"Comment {i}",
                "created_at": "2024-01-01",
                "url": f"https://pubpeer.com/comment/{i}",
            }
            for i in range(15)
        ]
        response = _make_response(200, {
            "data": [
                {
                    "total_comments": 15,
                    "url": "https://pubpeer.com/publications/XYZ",
                    "comments": comments,
                }
            ]
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/many.comments")

        assert result.total_comments == 15
        assert len(result.comments) == 10

    @pytest.mark.asyncio
    async def test_content_snippet_truncated(self):
        long_content = "A" * 500
        response = _make_response(200, {
            "data": [
                {
                    "total_comments": 1,
                    "comments": [
                        {
                            "author": "Test",
                            "content": long_content,
                            "created_at": "2024-01-01",
                            "url": "https://pubpeer.com/comment/1",
                        }
                    ],
                }
            ]
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.pubpeer.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_pubpeer("10.1234/long.content")

        assert len(result.comments[0].content_snippet) <= 300
