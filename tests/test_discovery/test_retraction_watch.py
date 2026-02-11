"""Tests for the Retraction Watch / Crossref retraction integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from snoopy.discovery.retraction_watch import (
    check_author_retractions,
    check_retraction_status,
)


def _make_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _make_client(get_return=None, get_side_effect=None) -> AsyncMock:
    """Create a mock httpx.AsyncClient."""
    mock_client = AsyncMock()
    if get_return is not None:
        mock_client.get = AsyncMock(return_value=get_return)
    if get_side_effect is not None:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    return mock_client


class TestCheckRetractionStatus:
    @pytest.mark.asyncio
    async def test_retracted_paper(self):
        response = _make_response(200, {
            "message": {
                "update-to": [
                    {
                        "type": "retraction",
                        "DOI": "10.1234/retraction.notice",
                        "updated": {"date-time": "2024-03-15"},
                    }
                ],
            }
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/retracted")

        assert result.is_retracted is True
        assert result.retraction_doi == "10.1234/retraction.notice"
        assert result.update_type == "retraction"

    @pytest.mark.asyncio
    async def test_expression_of_concern(self):
        response = _make_response(200, {
            "message": {
                "update-to": [
                    {
                        "type": "expression_of_concern",
                        "DOI": "10.1234/eoc.notice",
                    }
                ],
            }
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/eoc")

        assert result.is_retracted is False
        assert result.has_expression_of_concern is True
        assert result.update_type == "expression_of_concern"

    @pytest.mark.asyncio
    async def test_correction(self):
        response = _make_response(200, {
            "message": {
                "update-to": [
                    {
                        "type": "correction",
                        "DOI": "10.1234/correction",
                    }
                ],
            }
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/corrected")

        assert result.is_retracted is False
        assert result.has_correction is True

    @pytest.mark.asyncio
    async def test_clean_paper(self):
        response = _make_response(200, {
            "message": {
                "type": "journal-article",
            }
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/clean")

        assert result.is_retracted is False
        assert result.has_expression_of_concern is False
        assert result.has_correction is False

    @pytest.mark.asyncio
    async def test_retraction_notice_type(self):
        response = _make_response(200, {
            "message": {
                "type": "retraction",
            }
        })
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/retraction.notice")

        assert result.is_retracted is True
        assert result.update_type == "retraction_notice"

    @pytest.mark.asyncio
    async def test_api_404(self):
        response = _make_response(404)
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/unknown")

        assert result.is_retracted is False

    @pytest.mark.asyncio
    async def test_http_error(self):
        mock_client = _make_client(
            get_side_effect=httpx.HTTPError("timeout")
        )

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_retraction_status("10.1234/timeout")

        assert result.is_retracted is False


class TestCheckAuthorRetractions:
    @pytest.mark.asyncio
    async def test_author_with_retractions(self):
        call_count = 0

        async def _mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(200, {
                    "message": {
                        "items": [
                            {"DOI": "10.1234/retracted.1", "title": "Retracted Paper 1"},
                            {"DOI": "10.1234/retracted.2", "title": "Retracted Paper 2"},
                        ]
                    }
                })
            else:
                return _make_response(200, {
                    "message": {
                        "items": [
                            {"DOI": "10.1234/eoc.1"},
                        ]
                    }
                })

        mock_client = _make_client(get_side_effect=_mock_get)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_author_retractions("John Doe")

        assert result.author_name == "John Doe"
        assert result.total_retractions == 2
        assert result.total_expressions_of_concern == 1
        assert result.retracted_dois == ["10.1234/retracted.1", "10.1234/retracted.2"]

    @pytest.mark.asyncio
    async def test_author_with_no_retractions(self):
        response = _make_response(200, {"message": {"items": []}})
        mock_client = _make_client(get_return=response)

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_author_retractions("Jane Clean")

        assert result.total_retractions == 0
        assert result.retracted_dois is None

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self):
        mock_client = _make_client(
            get_side_effect=httpx.HTTPError("network error")
        )

        with patch("snoopy.discovery.retraction_watch.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await check_author_retractions("Network Error Author")

        assert result.total_retractions == 0
        assert result.total_expressions_of_concern == 0
