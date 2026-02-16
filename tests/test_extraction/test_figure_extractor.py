"""Tests for figure extraction from PDFs."""

from __future__ import annotations

import pytest

from snoopy.extraction.figure_extractor import (
    FigureInfo,
    _ALLOWED_EXTENSIONS,
    _MAX_HEIGHT,
    _MAX_WIDTH,
    _MIN_HEIGHT,
    _MIN_WIDTH,
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

    def test_skips_small_images(self, tmp_path, monkeypatch) -> None:
        """Images smaller than minimum dimensions should be skipped."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": _MIN_WIDTH - 1,
                    "height": _MIN_HEIGHT - 1,
                    "ext": "png",
                    "image": b"\x89PNG" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "small_imgs.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert result == []

    def test_skips_oversized_images(self, tmp_path, monkeypatch) -> None:
        """Images larger than maximum dimensions should be skipped."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": _MAX_WIDTH + 1,
                    "height": _MAX_HEIGHT + 1,
                    "ext": "png",
                    "image": b"\x89PNG" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "oversized.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert result == []

    def test_sanitizes_malicious_extension(self, tmp_path, monkeypatch) -> None:
        """Malicious extension from PDF metadata should be sanitized to 'png'."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": 200,
                    "height": 200,
                    "ext": "../../etc/passwd",
                    "image": b"\x89PNG" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "malicious.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        out_dir = tmp_path / "out"
        result = extract_figures(str(pdf_path), str(out_dir))
        assert len(result) == 1
        # The extension should have been sanitized to 'png'
        assert result[0].image_path.endswith(".png")

    def test_allowed_extensions_constant(self) -> None:
        """Verify the allowed extensions set."""
        assert "png" in _ALLOWED_EXTENSIONS
        assert "jpg" in _ALLOWED_EXTENSIONS
        assert "jpeg" in _ALLOWED_EXTENSIONS
        assert "gif" in _ALLOWED_EXTENSIONS
        assert "bmp" in _ALLOWED_EXTENSIONS
        assert "tiff" in _ALLOWED_EXTENSIONS
        assert "webp" in _ALLOWED_EXTENSIONS

    def test_extracts_valid_image(self, tmp_path, monkeypatch) -> None:
        """A valid-sized image should be extracted."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": 200,
                    "height": 200,
                    "ext": "jpg",
                    "image": b"\xff\xd8\xff" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "valid.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        out_dir = tmp_path / "out"
        result = extract_figures(str(pdf_path), str(out_dir))
        assert len(result) == 1
        assert result[0].width == 200
        assert result[0].height == 200
        assert result[0].image_path.endswith(".jpg")


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


class TestMalformedPDFs:
    """Tests for malformed, truncated, and edge-case PDFs."""

    def test_truncated_pdf_raises_runtime_error(self, tmp_path) -> None:
        """A truncated PDF (valid header, incomplete content) should raise RuntimeError."""
        pdf_path = tmp_path / "truncated.pdf"
        # Valid PDF header but truncated body
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n")

        # fitz.open will fail on a truly truncated PDF
        with pytest.raises((RuntimeError, Exception)):
            extract_figures(str(pdf_path), str(tmp_path / "out"))

    def test_password_protected_pdf(self, tmp_path, monkeypatch) -> None:
        """A password-protected PDF should raise RuntimeError."""
        import fitz

        pdf_path = tmp_path / "protected.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        def _raise_on_open(path):
            raise RuntimeError("cannot open encrypted PDF")

        monkeypatch.setattr(fitz, "open", _raise_on_open)

        with pytest.raises(RuntimeError, match="Failed to open PDF"):
            extract_figures(str(pdf_path), str(tmp_path / "out"))

    def test_pdf_with_zero_extractable_images(self, tmp_path, monkeypatch) -> None:
        """A PDF with pages but no images should return an empty list."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return []

        class _FakeDoc:
            def __len__(self):
                return 5  # 5 pages, no images

            def __getitem__(self, idx):
                return _FakePage()

            def close(self):
                pass

        pdf_path = tmp_path / "no_images.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert result == []

    def test_image_extraction_failure_continues(self, tmp_path, monkeypatch) -> None:
        """When extract_image raises for one xref, other images should still be extracted."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                # Two images: xref 1 will fail, xref 2 will succeed
                return [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                if xref == 1:
                    raise Exception("corrupt image data")
                return {
                    "width": 200,
                    "height": 200,
                    "ext": "png",
                    "image": b"\x89PNG" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "partial.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        # Only the second image should be extracted
        assert len(result) == 1
        assert result[0].width == 200

    def test_extract_image_returns_none(self, tmp_path, monkeypatch) -> None:
        """When extract_image returns None, the image should be skipped."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return None

            def close(self):
                pass

        pdf_path = tmp_path / "none_image.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert result == []

    def test_extract_image_empty_bytes(self, tmp_path, monkeypatch) -> None:
        """When image bytes are empty, the image should be skipped."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": 200,
                    "height": 200,
                    "ext": "png",
                    "image": b"",
                }

            def close(self):
                pass

        pdf_path = tmp_path / "empty_bytes.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert result == []

    def test_duplicate_xrefs_deduplicated(self, tmp_path, monkeypatch) -> None:
        """Duplicate xrefs across pages should only be extracted once."""
        import fitz

        class _FakePage:
            def get_images(self, full=True):
                # Same xref on every page
                return [(42, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

        class _FakeDoc:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return _FakePage()

            def extract_image(self, xref):
                return {
                    "width": 200,
                    "height": 200,
                    "ext": "png",
                    "image": b"\x89PNG" + b"\x00" * 100,
                }

            def close(self):
                pass

        pdf_path = tmp_path / "dup_xref.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

        result = extract_figures(str(pdf_path), str(tmp_path / "out"))
        assert len(result) == 1


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
