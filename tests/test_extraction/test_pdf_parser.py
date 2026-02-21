"""Tests for PDF text and metadata extraction."""

from __future__ import annotations

import pytest

from rosette.extraction.pdf_parser import (
    PageText,
    download_pdf,
    extract_metadata,
    extract_text,
)


class TestExtractText:
    def test_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_text(str(tmp_path / "nonexistent.pdf"))

    def test_raises_runtime_error_on_bad_pdf(self, tmp_path, monkeypatch) -> None:
        """A file that exists but cannot be parsed should raise RuntimeError."""
        import fitz

        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real PDF")

        def _raise(*args, **kwargs):
            raise Exception("parse error")

        monkeypatch.setattr(fitz, "open", _raise)

        with pytest.raises(RuntimeError, match="Failed to open PDF"):
            extract_text(str(bad_pdf))

    def test_returns_page_text_objects(self, tmp_path, monkeypatch) -> None:
        import fitz

        class _FakePage:
            def get_text(self, mode):
                return "Hello world from page one."

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def close(self):
                pass

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        pages = extract_text(str(pdf_path))
        assert len(pages) == 1
        assert isinstance(pages[0], PageText)
        assert pages[0].page_number == 1
        assert pages[0].word_count == 5
        assert "Hello" in pages[0].text


class TestExtractMetadata:
    def test_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_metadata(str(tmp_path / "nonexistent.pdf"))

    def test_returns_metadata_dict(self, tmp_path, monkeypatch) -> None:
        import fitz

        class _FakeDoc:
            metadata = {
                "title": "Test Paper",
                "author": "Jane Doe",
            }

            def __len__(self):
                return 3

            def close(self):
                pass

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        meta = extract_metadata(str(pdf_path))
        assert meta["title"] == "Test Paper"
        assert meta["author"] == "Jane Doe"
        assert meta["page_count"] == 3
        # Missing keys should default to empty string
        assert meta["subject"] == ""


class TestDownloadPdf:
    @pytest.mark.asyncio
    async def test_download_raises_on_network_error(self, tmp_path) -> None:
        from unittest.mock import AsyncMock, patch

        import httpx

        mock_client = AsyncMock()
        mock_client.stream.side_effect = httpx.HTTPError("connection failed")

        with patch("rosette.extraction.pdf_parser.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="Failed to download PDF"):
                await download_pdf("https://example.com/paper.pdf", str(tmp_path / "out.pdf"))

    @pytest.mark.asyncio
    async def test_download_raises_on_http_status_error(self, tmp_path) -> None:
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        import httpx

        @asynccontextmanager
        async def _mock_stream(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404",
                request=httpx.Request("GET", "https://example.com/paper.pdf"),
                response=httpx.Response(404),
            )
            yield resp

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream

        with patch("rosette.extraction.pdf_parser.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.HTTPStatusError):
                await download_pdf("https://example.com/paper.pdf", str(tmp_path / "out.pdf"))

    @pytest.mark.asyncio
    async def test_download_rejects_http_url(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Only HTTPS URLs"):
            await download_pdf("http://example.com/paper.pdf", str(tmp_path / "out.pdf"))
