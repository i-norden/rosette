"""Tests for demo fixture download and generation utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from snoopy.demo.fixtures import (
    CLEAN_PAPERS,
    RETRACTED_PAPERS,
    RETRACTION_WATCH_PAPERS,
    RSIIL_BASE,
    RSIIL_CLEAN_IMAGES,
    RSIIL_FORGERY_IMAGES,
    SURVEY_PAPER,
    _count_files,
    _download_file,
    _pmc_pdf_url,
    download_pmc_papers,
    generate_synthetic_forgeries,
)


class TestPmcPdfUrl:
    def test_constructs_europepmc_url(self):
        url = _pmc_pdf_url("PMC1234567")
        assert "europepmc.org" in url
        assert "PMC1234567" in url

    def test_different_pmcids(self):
        url1 = _pmc_pdf_url("PMC111")
        url2 = _pmc_pdf_url("PMC222")
        assert url1 != url2
        assert "PMC111" in url1
        assert "PMC222" in url2


class TestFixtureConstants:
    def test_retracted_papers_have_required_keys(self):
        for paper in RETRACTED_PAPERS:
            assert "pmcid" in paper
            assert "filename" in paper
            assert "desc" in paper
            assert paper["pmcid"].startswith("PMC")

    def test_clean_papers_have_required_keys(self):
        for paper in CLEAN_PAPERS:
            assert "pmcid" in paper
            assert "filename" in paper
            assert paper["pmcid"].startswith("PMC")

    def test_rsiil_images_are_tuples(self):
        for remote_path, local_name in RSIIL_FORGERY_IMAGES:
            assert isinstance(remote_path, str)
            assert isinstance(local_name, str)
            assert local_name.endswith(".png")

        for remote_path, local_name in RSIIL_CLEAN_IMAGES:
            assert isinstance(remote_path, str)
            assert isinstance(local_name, str)

    def test_rsiil_base_url(self):
        assert RSIIL_BASE.startswith("https://")
        assert "github" in RSIIL_BASE

    def test_survey_paper_format(self):
        assert "pmcid" in SURVEY_PAPER
        assert "filename" in SURVEY_PAPER


class TestDownloadFile:
    def test_skips_existing_file(self, tmp_path):
        dest = tmp_path / "existing.pdf"
        dest.write_bytes(b"%PDF-1.4 test content")
        client = MagicMock()
        result = _download_file("https://example.com/test.pdf", dest, client)
        assert result is False
        client.get.assert_not_called()

    def test_downloads_new_file(self, tmp_path):
        dest = tmp_path / "new.pdf"
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-1.4 test content"
        mock_resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.get.return_value = mock_resp
        result = _download_file("https://example.com/test.pdf", dest, client)
        assert result is True
        assert dest.exists()

    def test_handles_download_failure(self, tmp_path):
        import httpx

        dest = tmp_path / "fail.pdf"
        client = MagicMock()
        client.get.side_effect = httpx.RequestError("connection failed")
        with patch("snoopy.demo.fixtures.console"):
            result = _download_file("https://example.com/test.pdf", dest, client)
        assert result is False


class TestCountFiles:
    def test_counts_files_in_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")
        # Non-recursive count
        assert _count_files(tmp_path) == 2

    def test_returns_zero_for_missing_directory(self):
        assert _count_files(Path("/nonexistent/path")) == 0


class TestGenerateSyntheticForgeries:
    def test_generates_images(self, tmp_path):
        with patch("snoopy.demo.fixtures.FIXTURES_DIR", tmp_path):
            with patch("snoopy.demo.fixtures.create_progress") as mock_progress:
                mock_progress.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(add_task=MagicMock(return_value=0), advance=MagicMock())
                )
                mock_progress.return_value.__exit__ = MagicMock(return_value=False)
                count = generate_synthetic_forgeries()
        assert count == 10
        synthetic_dir = tmp_path / "synthetic"
        assert synthetic_dir.exists()
        png_files = list(synthetic_dir.glob("*.png"))
        assert len(png_files) == 10


class TestDownloadPmcPapers:
    def test_download_pmc_papers(self, tmp_path):
        papers = [{"pmcid": "PMC1234", "filename": "PMC1234.pdf", "desc": "test"}]
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-1.4 content"
        mock_resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.get.return_value = mock_resp

        with patch("snoopy.demo.fixtures.FIXTURES_DIR", tmp_path):
            with patch("snoopy.demo.fixtures.create_progress") as mock_progress:
                mock_progress.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(add_task=MagicMock(return_value=0), advance=MagicMock())
                )
                mock_progress.return_value.__exit__ = MagicMock(return_value=False)
                count = download_pmc_papers(papers, "test_cat", client)

        assert count == 1
        assert (tmp_path / "test_cat" / "PMC1234.pdf").exists()
