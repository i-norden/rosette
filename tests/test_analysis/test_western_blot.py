"""Tests for western blot analysis module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from snoopy.analysis.western_blot import (
    BandProfile,
    WesternBlotResult,
    _compare_profiles,
    _detect_lanes,
    analyze_western_blot,
)


@pytest.fixture
def synthetic_western_blot(tmp_path) -> str:
    """Create a synthetic western blot image with distinct lanes.

    Generates a dark background with bright vertical bands (lanes) separated
    by dark gaps, mimicking the structure of a real western blot.
    """
    width, height = 400, 200
    arr = np.zeros((height, width), dtype=np.uint8)

    # Create 5 distinct lanes with bright bands
    lane_positions = [(30, 70), (90, 130), (150, 190), (210, 250), (270, 310)]
    for i, (start, end) in enumerate(lane_positions):
        # Each lane has a different intensity pattern
        for y in range(height):
            intensity = int(100 + 80 * np.sin(y / 20.0 + i * 0.5))
            arr[y, start:end] = np.clip(intensity, 0, 255)

        # Add some bright bands at different positions per lane
        band_y = 50 + i * 15
        arr[band_y : band_y + 10, start:end] = 220

    # Convert to RGB and save
    rgb = np.stack([arr, arr, arr], axis=2)
    img = Image.fromarray(rgb)
    path = str(tmp_path / "western_blot.png")
    img.save(path)
    return path


class TestAnalyzeWesternBlot:
    def test_analyze_western_blot_with_clean_image(self, clean_image: str) -> None:
        """Clean gradient image should not flag suspicious on western blot analysis.

        A gradient image may or may not be detected as having lanes, but
        it should not produce spurious high-confidence duplication findings.
        """
        result = analyze_western_blot(clean_image)
        assert isinstance(result, WesternBlotResult)
        # A clean gradient image should not have duplicate lanes
        assert len(result.duplicate_lanes) == 0


class TestDetectLanes:
    def test_detect_lanes(self, synthetic_western_blot: str) -> None:
        """Verify lane detection on a synthetic western blot image."""
        import cv2

        img = cv2.imread(synthetic_western_blot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lanes = _detect_lanes(gray, min_lane_width=10)
        # We should detect at least some lanes from our synthetic image
        assert len(lanes) >= 2

        # Each lane should be a (start, end) tuple with valid coordinates
        for start, end in lanes:
            assert start < end
            assert start >= 0
            assert end <= gray.shape[1]


class TestCompareProfiles:
    def test_compare_profiles_identical(self) -> None:
        """Correlation should be near 1.0 for identical profiles."""
        intensities = [float(x) for x in range(100)]
        profile_a = BandProfile(
            lane_index=0,
            x_start=0,
            x_end=50,
            mean_intensities=intensities,
            peak_positions=[25, 50, 75],
            peak_intensities=[25.0, 50.0, 75.0],
        )
        profile_b = BandProfile(
            lane_index=1,
            x_start=50,
            x_end=100,
            mean_intensities=intensities,
            peak_positions=[25, 50, 75],
            peak_intensities=[25.0, 50.0, 75.0],
        )

        correlation = _compare_profiles(profile_a, profile_b)
        assert abs(correlation - 1.0) < 0.01

    def test_compare_profiles_different(self) -> None:
        """Correlation should be lower for different profiles."""
        ascending = [float(x) for x in range(100)]
        descending = [float(99 - x) for x in range(100)]

        profile_a = BandProfile(
            lane_index=0,
            x_start=0,
            x_end=50,
            mean_intensities=ascending,
            peak_positions=[],
            peak_intensities=[],
        )
        profile_b = BandProfile(
            lane_index=1,
            x_start=50,
            x_end=100,
            mean_intensities=descending,
            peak_positions=[],
            peak_intensities=[],
        )

        correlation = _compare_profiles(profile_a, profile_b)
        # Perfectly negatively correlated
        assert correlation < 0
