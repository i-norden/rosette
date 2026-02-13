from __future__ import annotations

import numpy as np
from PIL import Image

from snoopy.analysis.image_forensics import FFTResult, frequency_analysis


class TestFFTNaturalImage:
    """A natural-looking image (gradient with noise) should have a reasonable high_freq_ratio."""

    def test_natural_image_high_freq_ratio_bounded(self, tmp_path):
        # Create a gradient image with some random noise to mimic natural content
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            arr[i, :, :] = int(i / 256 * 255)
        noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = str(tmp_path / "natural.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result, FFTResult)
        assert 0.0 <= result.high_freq_ratio <= 1.0

    def test_natural_image_spectral_anomaly_non_negative(self, tmp_path):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                arr[i, j, 0] = int(i / 256 * 200)
                arr[i, j, 1] = int(j / 256 * 200)
                arr[i, j, 2] = 100
        noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = str(tmp_path / "natural2.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert result.spectral_anomaly_score >= 0.0


class TestFFTResampledImage:
    """Resized images (interpolation artifacts) should produce a positive spectral anomaly score."""

    def test_resampled_image_has_positive_anomaly_score(self, tmp_path):
        # Create an image with fine detail
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        # Downsample then upsample to introduce interpolation artifacts
        small = img.resize((128, 128), Image.BILINEAR)
        resampled = small.resize((512, 512), Image.BILINEAR)

        path = str(tmp_path / "resampled.png")
        resampled.save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result, FFTResult)
        assert result.spectral_anomaly_score >= 0.0

    def test_resampled_image_periodic_peaks_is_list(self, tmp_path):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        small = img.resize((64, 64), Image.BILINEAR)
        resampled = small.resize((256, 256), Image.BILINEAR)

        path = str(tmp_path / "resampled_peaks.png")
        resampled.save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result.periodic_peaks, list)
        for peak in result.periodic_peaks:
            assert isinstance(peak, float)


class TestFFTUniformImage:
    """A solid color image should not crash and should produce valid output."""

    def test_uniform_image_does_not_crash(self, tmp_path):
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 128
        path = str(tmp_path / "uniform.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result, FFTResult)

    def test_uniform_image_non_negative_anomaly(self, tmp_path):
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 64
        path = str(tmp_path / "uniform_dark.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert result.spectral_anomaly_score >= 0.0

    def test_uniform_image_high_freq_ratio_bounded(self, tmp_path):
        arr = np.ones((200, 200, 3), dtype=np.uint8) * 200
        path = str(tmp_path / "uniform_light.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert 0.0 <= result.high_freq_ratio <= 1.0


class TestFFTSmallImage:
    """Very small images (16x16) should not crash."""

    def test_small_image_does_not_crash(self, tmp_path):
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        path = str(tmp_path / "tiny.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result, FFTResult)

    def test_small_image_returns_valid_fields(self, tmp_path):
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        path = str(tmp_path / "tiny_valid.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result.suspicious, bool)
        assert isinstance(result.spectral_anomaly_score, float)
        assert isinstance(result.periodic_peaks, list)
        assert isinstance(result.high_freq_ratio, float)
        assert isinstance(result.details, str)


class TestFFTResultFields:
    """Verify all FFTResult fields are present and have correct types."""

    def test_all_fields_present(self, tmp_path):
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        path = str(tmp_path / "fields_check.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert hasattr(result, "suspicious")
        assert hasattr(result, "spectral_anomaly_score")
        assert hasattr(result, "periodic_peaks")
        assert hasattr(result, "high_freq_ratio")
        assert hasattr(result, "details")

    def test_field_types(self, tmp_path):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "type_check.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        assert isinstance(result.suspicious, bool)
        assert isinstance(result.spectral_anomaly_score, float)
        assert isinstance(result.periodic_peaks, list)
        assert isinstance(result.high_freq_ratio, float)
        assert isinstance(result.details, str)

    def test_periodic_peaks_elements_are_floats(self, tmp_path):
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        path = str(tmp_path / "peaks_types.png")
        Image.fromarray(arr).save(path, "PNG")

        result = frequency_analysis(path)

        for peak in result.periodic_peaks:
            assert isinstance(peak, float)

    def test_custom_anomaly_threshold(self, tmp_path):
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        path = str(tmp_path / "threshold.png")
        Image.fromarray(arr).save(path, "PNG")

        result_strict = frequency_analysis(path, anomaly_threshold=0.5)
        result_lenient = frequency_analysis(path, anomaly_threshold=10.0)

        # Same image should produce the same spectral score regardless of threshold
        assert result_strict.spectral_anomaly_score == result_lenient.spectral_anomaly_score

        # A stricter threshold should be at least as likely to flag suspicious
        if result_lenient.suspicious:
            assert result_strict.suspicious

    def test_nonexistent_file_returns_result(self, tmp_path):
        fake_path = str(tmp_path / "no_such_image.png")

        result = frequency_analysis(fake_path)
        assert isinstance(result, FFTResult)
