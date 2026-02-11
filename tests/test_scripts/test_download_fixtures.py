"""Tests for snoopy.demo.fixtures (fixture download/generation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx

from snoopy.demo.fixtures import (
    _count_files,
    _download_file,
    _download_streaming,
    _pmc_pdf_url,
    generate_synthetic_forgeries,
    sample_rsiil_images,
)
import snoopy.demo.fixtures as fixtures_mod


class TestPmcPdfUrl:
    def test_standard_pmcid(self) -> None:
        url = _pmc_pdf_url("PMC1234567")
        assert url == "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC1234567&blobtype=pdf"

    def test_another_pmcid(self) -> None:
        url = _pmc_pdf_url("PMC4941872")
        assert url == "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4941872&blobtype=pdf"


class TestCountFiles:
    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert _count_files(tmp_path / "nonexistent") == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert _count_files(d) == 0

    def test_directory_with_files(self, tmp_path: Path) -> None:
        d = tmp_path / "files"
        d.mkdir()
        (d / "a.txt").write_text("a")
        (d / "b.txt").write_text("b")
        assert _count_files(d) == 2

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.txt").write_text("a")
        (d / "subdir").mkdir()
        assert _count_files(d) == 1


class TestDownloadFile:
    def test_skips_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "existing.pdf"
        dest.write_bytes(b"content")
        client = MagicMock()
        result = _download_file("https://example.com/file.pdf", dest, client)
        assert result is False
        client.get.assert_not_called()

    def test_skips_empty_existing_file(self, tmp_path: Path) -> None:
        """An empty file should be re-downloaded."""
        dest = tmp_path / "empty.pdf"
        dest.write_bytes(b"")
        client = MagicMock()
        resp = MagicMock()
        resp.content = b"pdf-data"
        resp.raise_for_status = MagicMock()
        client.get.return_value = resp
        result = _download_file("https://example.com/file.pdf", dest, client)
        assert result is True
        assert dest.read_bytes() == b"pdf-data"

    def test_downloads_new_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "new.pdf"
        client = MagicMock()
        resp = MagicMock()
        resp.content = b"pdf-data-here"
        resp.raise_for_status = MagicMock()
        client.get.return_value = resp
        result = _download_file("https://example.com/new.pdf", dest, client)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"pdf-data-here"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "file.pdf"
        client = MagicMock()
        resp = MagicMock()
        resp.content = b"data"
        resp.raise_for_status = MagicMock()
        client.get.return_value = resp
        _download_file("https://example.com/file.pdf", dest, client)
        assert dest.parent.exists()

    def test_handles_http_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "fail.pdf"
        client = MagicMock()
        client.get.side_effect = httpx.RequestError("connection failed")
        result = _download_file("https://example.com/fail.pdf", dest, client)
        assert result is False
        assert not dest.exists()

    def test_handles_http_status_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "404.pdf"
        client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        client.get.return_value = resp
        result = _download_file("https://example.com/404.pdf", dest, client)
        assert result is False


class TestGenerateSyntheticForgeries:
    def test_creates_ten_images(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        count = generate_synthetic_forgeries()
        assert count == 10
        out_dir = tmp_path / "synthetic"
        assert out_dir.exists()
        images = list(out_dir.glob("*.png"))
        assert len(images) == 10

    def test_idempotent_second_run(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        first = generate_synthetic_forgeries()
        assert first == 10
        second = generate_synthetic_forgeries()
        assert second == 0  # all already exist

    def test_copymove_images_created(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        generate_synthetic_forgeries()
        out_dir = tmp_path / "synthetic"
        copymove = [f for f in out_dir.iterdir() if "copymove" in f.name]
        assert len(copymove) == 7

    def test_spliced_image_created(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        generate_synthetic_forgeries()
        out_dir = tmp_path / "synthetic"
        spliced = [f for f in out_dir.iterdir() if "spliced" in f.name]
        assert len(spliced) == 1

    def test_retouched_images_created(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        generate_synthetic_forgeries()
        out_dir = tmp_path / "synthetic"
        retouched = [f for f in out_dir.iterdir() if "retouched" in f.name]
        assert len(retouched) == 2

    def test_images_are_valid_pngs(self, tmp_path: Path, monkeypatch) -> None:
        from PIL import Image
        monkeypatch.setattr(fixtures_mod, "FIXTURES_DIR", tmp_path)
        generate_synthetic_forgeries()
        out_dir = tmp_path / "synthetic"
        for img_path in out_dir.glob("*.png"):
            img = Image.open(img_path)
            assert img.size == (512, 512)
            assert img.mode == "RGB"


class TestDownloadStreaming:
    def test_skips_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "existing.zip"
        dest.write_bytes(b"content")
        client = MagicMock()
        result = _download_streaming("https://example.com/file.zip", dest, client)
        assert result is False
        client.stream.assert_not_called()

    def test_skips_empty_existing_file(self, tmp_path: Path) -> None:
        """An empty file should be re-downloaded."""
        dest = tmp_path / "empty.zip"
        dest.write_bytes(b"")
        # Build a mock streaming response
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "11"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = [b"hello", b" world"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        client = MagicMock()
        client.stream.return_value = mock_resp
        result = _download_streaming("https://example.com/file.zip", dest, client)
        assert result is True
        assert dest.read_bytes() == b"hello world"

    def test_downloads_new_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "new.zip"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "4"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = [b"data"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        client = MagicMock()
        client.stream.return_value = mock_resp
        result = _download_streaming("https://example.com/new.zip", dest, client)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"data"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "file.zip"
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "3"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = [b"abc"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        client = MagicMock()
        client.stream.return_value = mock_resp
        _download_streaming("https://example.com/file.zip", dest, client)
        assert dest.parent.exists()

    def test_handles_request_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "fail.zip"
        client = MagicMock()
        client.stream.side_effect = httpx.RequestError("connection failed")
        result = _download_streaming("https://example.com/fail.zip", dest, client)
        assert result is False
        assert not dest.exists()


class TestSampleRsiilImages:
    def _create_rsiil_data(self, tmp_path: Path) -> Path:
        """Create a fake RSIIL data directory structure with images."""
        rsiil_dir = tmp_path / "data" / "rsiil"
        pristine = rsiil_dir / "pristine"
        test = rsiil_dir / "test"
        for d in (pristine, test):
            d.mkdir(parents=True)
        # Create pristine images
        for i in range(50):
            (pristine / f"pristine_{i:03d}.png").write_bytes(b"img")
        # Create test (tampered) images
        for i in range(40):
            (test / f"test_{i:03d}.png").write_bytes(b"img")
        return rsiil_dir

    def test_returns_correct_counts(self, tmp_path: Path, monkeypatch) -> None:
        rsiil_dir = self._create_rsiil_data(tmp_path)
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", rsiil_dir)
        pristine, tampered = sample_rsiil_images(30)
        assert len(pristine) == 30
        assert len(tampered) == 30

    def test_returns_empty_when_no_data(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", tmp_path / "nonexistent")
        pristine, tampered = sample_rsiil_images(30)
        assert pristine == []
        assert tampered == []

    def test_handles_fewer_images_than_sample_size(self, tmp_path: Path, monkeypatch) -> None:
        rsiil_dir = tmp_path / "data" / "rsiil"
        pristine_dir = rsiil_dir / "pristine"
        test_dir = rsiil_dir / "test"
        for d in (pristine_dir, test_dir):
            d.mkdir(parents=True)
        # Only 5 pristine and 3 tampered
        for i in range(5):
            (pristine_dir / f"p_{i}.png").write_bytes(b"img")
        for i in range(3):
            (test_dir / f"t_{i}.png").write_bytes(b"img")
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", rsiil_dir)
        pristine, tampered = sample_rsiil_images(30)
        assert len(pristine) == 5
        assert len(tampered) == 3

    def test_deterministic_sampling(self, tmp_path: Path, monkeypatch) -> None:
        rsiil_dir = self._create_rsiil_data(tmp_path)
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", rsiil_dir)
        p1, t1 = sample_rsiil_images(20, seed=42)
        p2, t2 = sample_rsiil_images(20, seed=42)
        assert p1 == p2
        assert t1 == t2

    def test_different_seed_gives_different_sample(self, tmp_path: Path, monkeypatch) -> None:
        rsiil_dir = self._create_rsiil_data(tmp_path)
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", rsiil_dir)
        p1, t1 = sample_rsiil_images(20, seed=42)
        p2, t2 = sample_rsiil_images(20, seed=99)
        # Very unlikely to be the same with different seeds
        assert p1 != p2 or t1 != t2

    def test_ignores_non_image_files(self, tmp_path: Path, monkeypatch) -> None:
        rsiil_dir = tmp_path / "data" / "rsiil"
        pristine_dir = rsiil_dir / "pristine"
        test_dir = rsiil_dir / "test"
        for d in (pristine_dir, test_dir):
            d.mkdir(parents=True)
        (pristine_dir / "readme.txt").write_text("not an image")
        (pristine_dir / "data.csv").write_text("1,2,3")
        (pristine_dir / "image.png").write_bytes(b"img")
        monkeypatch.setattr(fixtures_mod, "RSIIL_DATA_DIR", rsiil_dir)
        pristine, tampered = sample_rsiil_images(30)
        assert len(pristine) == 1
        assert pristine[0].name == "image.png"
