"""Tests for figure extraction from PDFs."""

from __future__ import annotations

import pytest

from snoopy.extraction.figure_extractor import (
    FigureInfo,
    _sha256_file,
    associate_captions,
    extract_figures,
)


class TestExtractFigures:
    def test_raises_file_not_found(self, tmp_path) -> None:
        missing = str(tmp_path / "nonexistent.pdf")
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_figures(missing, str(tmp_path / "out"))

    def test_output_dir_created(self, tmp_path, monkeypatch) -> None:
        """extract_figures should create the output directory if it does
        not exist, even when the PDF contains no images."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return []

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def close(self):
                pass

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        out_dir = tmp_path / "figures_out"
        result = extract_figures(str(pdf_path), str(out_dir))
        assert result == []
        assert out_dir.exists()


class TestAssociateCaptions:
    def test_empty_figures_returns_empty(self, tmp_path) -> None:
        result = associate_captions(str(tmp_path / "any.pdf"), [])
        assert result == []

    def test_raises_file_not_found(self, tmp_path) -> None:
        fig = FigureInfo(
            page_number=1,
            image_path="dummy.png",
            image_sha256="abc",
            width=100,
            height=100,
        )
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            associate_captions(str(tmp_path / "missing.pdf"), [fig])


class TestSha256File:
    def test_computes_correct_hash(self, tmp_path) -> None:
        import hashlib

        content = b"hello world"
        path = tmp_path / "test.bin"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _sha256_file(str(path)) == expected

    def test_empty_file(self, tmp_path) -> None:
        import hashlib

        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(str(path)) == expected
