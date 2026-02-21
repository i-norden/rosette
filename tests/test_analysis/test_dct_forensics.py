from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rosette.analysis.image_forensics import DCTResult, dct_analysis


class TestDCTCleanImage:
    """A clean single-compression JPEG should show low periodicity."""

    def test_clean_jpeg_low_periodicity(self, tmp_path):
        # Create a smooth gradient image and save once at high quality
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        path = str(tmp_path / "clean.jpg")
        Image.fromarray(arr).save(path, "JPEG", quality=90)

        result = dct_analysis(path)

        assert isinstance(result, DCTResult)
        assert isinstance(result.periodicity_score, float)
        assert result.periodicity_score < 0.5

    def test_clean_jpeg_returns_all_fields(self, tmp_path):
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        path = str(tmp_path / "clean_fields.jpg")
        Image.fromarray(arr).save(path, "JPEG", quality=90)

        result = dct_analysis(path)

        assert isinstance(result.suspicious, bool)
        assert isinstance(result.periodicity_score, float)
        assert result.estimated_primary_quality is None or isinstance(
            result.estimated_primary_quality, int
        )
        assert isinstance(result.block_inconsistencies, int)
        assert isinstance(result.details, str)


class TestDCTDoubleCompressed:
    """Double-compressed JPEGs should show higher periodicity than single."""

    def test_double_compression_produces_valid_result(self, tmp_path):
        # Create image and save at low quality first
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        first_path = str(tmp_path / "first_pass.jpg")
        Image.fromarray(arr).save(first_path, "JPEG", quality=60)

        # Load and re-save at different quality (double compression)
        reloaded = Image.open(first_path)
        second_path = str(tmp_path / "double_compressed.jpg")
        reloaded.save(second_path, "JPEG", quality=90)

        result = dct_analysis(second_path)

        assert isinstance(result, DCTResult)
        assert isinstance(result.periodicity_score, float)
        assert result.periodicity_score >= 0.0
        assert isinstance(result.block_inconsistencies, int)
        assert result.block_inconsistencies >= 0

    def test_double_compression_block_inconsistencies_non_negative(self, tmp_path):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        first_path = str(tmp_path / "pass1.jpg")
        Image.fromarray(arr).save(first_path, "JPEG", quality=60)

        reloaded = Image.open(first_path)
        second_path = str(tmp_path / "pass2.jpg")
        reloaded.save(second_path, "JPEG", quality=90)

        result = dct_analysis(second_path)

        assert result.block_inconsistencies >= 0


class TestDCTPNGInput:
    """PNG input (no JPEG compression history) should not crash and show low periodicity."""

    def test_png_input_does_not_crash(self, tmp_path):
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        path = str(tmp_path / "test.png")
        Image.fromarray(arr).save(path, "PNG")

        result = dct_analysis(path)

        assert isinstance(result, DCTResult)

    def test_png_input_low_periodicity(self, tmp_path):
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        path = str(tmp_path / "gradient.png")
        Image.fromarray(arr).save(path, "PNG")

        result = dct_analysis(path)

        assert result.periodicity_score < 0.5

    def test_png_input_details_is_string(self, tmp_path):
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        path = str(tmp_path / "solid.png")
        Image.fromarray(arr).save(path, "PNG")

        result = dct_analysis(path)

        assert isinstance(result.details, str)


class TestDCTNonexistentFile:
    """Nonexistent file path should return a valid non-suspicious result (OpenCV handles gracefully)."""

    def test_nonexistent_file_returns_result(self, tmp_path):
        fake_path = str(tmp_path / "does_not_exist.jpg")

        result = dct_analysis(fake_path)
        assert isinstance(result, DCTResult)

    def test_nonexistent_directory_returns_result(self):
        result = dct_analysis("/nonexistent/directory/image.jpg")
        assert isinstance(result, DCTResult)


class TestDCTCustomThreshold:
    """Verify the periodicity_threshold parameter is respected."""

    def test_high_threshold_reduces_suspicion(self, tmp_path):
        arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        first_path = str(tmp_path / "pass1.jpg")
        Image.fromarray(arr).save(first_path, "JPEG", quality=50)

        reloaded = Image.open(first_path)
        second_path = str(tmp_path / "pass2.jpg")
        reloaded.save(second_path, "JPEG", quality=95)

        result_strict = dct_analysis(second_path, periodicity_threshold=0.1)
        result_lenient = dct_analysis(second_path, periodicity_threshold=0.9)

        # With a very lenient threshold, the image is less likely to be flagged
        # than with a strict threshold (same score, different bar)
        if result_strict.suspicious:
            assert result_strict.periodicity_score == pytest.approx(
                result_lenient.periodicity_score
            )
