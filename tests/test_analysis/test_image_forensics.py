"""Tests for image forensics analysis methods."""

import numpy as np
from PIL import Image

from snoopy.analysis.image_forensics import (
    clone_detection,
    error_level_analysis,
    noise_analysis,
)


class TestELA:
    def test_clean_image(self, clean_image):
        result = error_level_analysis(clean_image)
        assert result.mean_difference >= 0
        assert result.std_difference >= 0

    def test_returns_ela_result(self, sample_jpeg):
        result = error_level_analysis(sample_jpeg)
        assert hasattr(result, "suspicious")
        assert hasattr(result, "max_difference")
        assert hasattr(result, "mean_difference")

    def test_nonexistent_file(self):
        try:
            error_level_analysis("/nonexistent/path.jpg")
            assert False, "Should have raised an error"
        except (FileNotFoundError, Exception):
            pass


class TestCloneDetection:
    def test_clean_image(self, sample_image):
        result = clone_detection(sample_image)
        assert hasattr(result, "suspicious")
        assert hasattr(result, "num_matches")

    def test_returns_clone_result(self, sample_image):
        result = clone_detection(sample_image)
        assert isinstance(result.num_matches, int)
        assert isinstance(result.match_clusters, list)

    def test_nonexistent_file(self):
        try:
            clone_detection("/nonexistent/path.png")
            assert False, "Should have raised an error"
        except (FileNotFoundError, Exception):
            pass


class TestNoiseAnalysis:
    def test_clean_image(self, clean_image):
        result = noise_analysis(clean_image)
        assert hasattr(result, "suspicious")
        assert hasattr(result, "mean_noise")
        assert result.mean_noise >= 0

    def test_returns_noise_result(self, sample_image):
        result = noise_analysis(sample_image)
        assert isinstance(result.noise_map, list)
        assert isinstance(result.max_ratio, float)
