"""Western blot specific analysis for detecting manipulation.

Western blot manipulation is the single most common form of image fraud in
biomedical papers (per Bik et al. 2016). This module provides dedicated
analysis techniques that exploit the structure of western blot images:
band intensity profiling, background continuity analysis, and lane alignment
checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BandProfile:
    """Intensity profile for a single lane in a western blot."""

    lane_index: int
    x_start: int
    x_end: int
    mean_intensities: list[float]
    peak_positions: list[int]
    peak_intensities: list[float]


@dataclass
class SpliceBoundary:
    """Detected splice boundary between lanes."""

    x_position: int
    left_lane: int
    right_lane: int
    background_discontinuity: float
    noise_discontinuity: float
    confidence: float


@dataclass
class WesternBlotResult:
    """Result of western blot specific analysis."""

    suspicious: bool
    band_profiles: list[BandProfile] = field(default_factory=list)
    duplicate_lanes: list[tuple[int, int, float]] = field(default_factory=list)
    splice_boundaries: list[SpliceBoundary] = field(default_factory=list)
    lane_count: int = 0
    uniform_profiles: bool = False
    details: str = ""


def _detect_lanes(gray: np.ndarray, min_lane_width: int = 15) -> list[tuple[int, int]]:
    """Detect lane boundaries in a western blot image.

    Projects the image vertically and finds lanes as regions between intensity
    valleys (dark separators between bright lanes).

    Returns:
        List of (x_start, x_end) tuples for each detected lane.
    """
    h, w = gray.shape
    # Vertical projection: average intensity per column
    projection = np.mean(gray, axis=0)

    # Smooth the projection to reduce noise
    kernel_size = max(5, w // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(projection.reshape(1, -1), (kernel_size, 1), 0).flatten()

    # Find valleys (lane boundaries) using adaptive thresholding.
    # Use the larger of mean*multiplier and 25th percentile to handle low-contrast images.
    threshold = max(np.mean(smoothed) * 0.7, np.percentile(smoothed, 25))
    in_lane = smoothed > threshold

    lanes = []
    lane_start = None
    for i in range(len(in_lane)):
        if in_lane[i] and lane_start is None:
            lane_start = i
        elif not in_lane[i] and lane_start is not None:
            if i - lane_start >= min_lane_width:
                lanes.append((lane_start, i))
            lane_start = None

    if lane_start is not None and len(smoothed) - lane_start >= min_lane_width:
        lanes.append((lane_start, len(smoothed)))

    return lanes


def _extract_band_profile(
    gray: np.ndarray, x_start: int, x_end: int, lane_index: int
) -> BandProfile:
    """Extract the horizontal intensity profile for a single lane.

    Computes mean intensity at each row across the lane width, then finds
    peaks (bands) in the profile.
    """
    lane_strip = gray[:, x_start:x_end]
    mean_intensities = np.mean(lane_strip, axis=1).tolist()

    # Find peaks (bands) — local maxima above mean intensity
    intensities = np.array(mean_intensities)
    mean_val = np.mean(intensities)
    peak_positions = []
    peak_intensities = []

    for i in range(1, len(intensities) - 1):
        if (
            intensities[i] > intensities[i - 1]
            and intensities[i] > intensities[i + 1]
            and intensities[i] > mean_val
        ):
            peak_positions.append(i)
            peak_intensities.append(float(intensities[i]))

    return BandProfile(
        lane_index=lane_index,
        x_start=x_start,
        x_end=x_end,
        mean_intensities=mean_intensities,
        peak_positions=peak_positions,
        peak_intensities=peak_intensities,
    )


def _compare_profiles(profile_a: BandProfile, profile_b: BandProfile) -> float:
    """Compute similarity between two lane profiles.

    Returns a correlation coefficient between -1 and 1, where values near 1
    indicate suspiciously identical lanes.
    """
    a = np.array(profile_a.mean_intensities)
    b = np.array(profile_b.mean_intensities)

    # Resize to same length if needed
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    if len(a) < 5:
        return 0.0

    # Normalized cross-correlation
    a_norm = a - np.mean(a)
    b_norm = b - np.mean(b)
    denom = np.std(a_norm) * np.std(b_norm) * len(a)
    if denom == 0:
        return 0.0

    return float(np.sum(a_norm * b_norm) / denom)


def _detect_splice_boundaries(
    gray: np.ndarray, lanes: list[tuple[int, int]]
) -> list[SpliceBoundary]:
    """Detect splice boundaries between lanes by analyzing background discontinuities.

    Examines the border region between adjacent lanes for abrupt changes in
    background level and noise characteristics that indicate splicing.
    """
    boundaries = []
    h, w = gray.shape

    for i in range(len(lanes) - 1):
        _, x_end_left = lanes[i]
        x_start_right, _ = lanes[i + 1]

        # Analyze the gap between lanes
        gap_start = max(0, x_end_left - 3)
        gap_end = min(w, x_start_right + 3)

        if gap_end - gap_start < 2:
            continue

        # Background discontinuity: compare mean intensity on each side
        left_border = gray[:, max(0, gap_start - 5) : gap_start]
        right_border = gray[:, gap_end : min(w, gap_end + 5)]

        if left_border.size == 0 or right_border.size == 0:
            continue

        bg_left = float(np.mean(left_border))
        bg_right = float(np.mean(right_border))
        bg_discontinuity = abs(bg_left - bg_right)

        # Noise discontinuity: compare noise variance on each side
        noise_left = float(np.var(cv2.Laplacian(left_border, cv2.CV_64F)))
        noise_right = float(np.var(cv2.Laplacian(right_border, cv2.CV_64F)))
        noise_disc = abs(noise_left - noise_right) / max(noise_left, noise_right, 1.0)

        # Confidence based on magnitude of discontinuities
        confidence = min(
            (bg_discontinuity / 30.0 + noise_disc) / 2.0,
            1.0,
        )

        if confidence > 0.3:
            boundaries.append(
                SpliceBoundary(
                    x_position=(gap_start + gap_end) // 2,
                    left_lane=i,
                    right_lane=i + 1,
                    background_discontinuity=bg_discontinuity,
                    noise_discontinuity=noise_disc,
                    confidence=confidence,
                )
            )

    return boundaries


def analyze_western_blot(image_path: str) -> WesternBlotResult:
    """Perform western blot specific analysis on an image.

    Detects:
    - Suspiciously identical lane profiles (band duplication)
    - Background discontinuities at splice boundaries
    - Lane alignment issues

    Args:
        image_path: Path to the western blot image.

    Returns:
        WesternBlotResult with detailed findings.
    """
    img = cv2.imread(image_path)
    if img is None:
        return WesternBlotResult(suspicious=False, details="Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Detect lanes
    lanes = _detect_lanes(gray)
    if len(lanes) < 2:
        return WesternBlotResult(
            suspicious=False,
            lane_count=len(lanes),
            details=f"Only {len(lanes)} lane(s) detected; need >= 2 for comparison",
        )

    # Step 2: Extract band profiles for each lane
    profiles = []
    for idx, (x_start, x_end) in enumerate(lanes):
        profile = _extract_band_profile(gray, x_start, x_end, idx)
        profiles.append(profile)

    # Step 3: Compare all lane pairs for suspicious similarity
    duplicate_lanes = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            correlation = _compare_profiles(profiles[i], profiles[j])
            if correlation > 0.95:
                duplicate_lanes.append((i, j, correlation))

    # Step 4: Check for suspiciously uniform profiles
    # (fabricated blots often have nearly identical loading controls)
    all_correlations = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            all_correlations.append(_compare_profiles(profiles[i], profiles[j]))
    uniform_profiles = bool(len(all_correlations) > 0 and np.mean(all_correlations) > 0.85)

    # Step 5: Detect splice boundaries
    splice_boundaries = _detect_splice_boundaries(gray, lanes)

    # Determine overall suspicion
    details_parts = []
    suspicious = False

    if duplicate_lanes:
        suspicious = True
        for i, j, corr in duplicate_lanes:
            details_parts.append(
                f"Lanes {i} and {j} have suspiciously similar profiles (correlation={corr:.3f})"
            )

    if uniform_profiles:
        suspicious = True
        details_parts.append(f"All {len(lanes)} lanes show suspiciously uniform intensity profiles")

    high_conf_splices = [s for s in splice_boundaries if s.confidence > 0.6]
    if high_conf_splices:
        suspicious = True
        for s in high_conf_splices:
            details_parts.append(
                f"Splice boundary detected between lanes {s.left_lane} and "
                f"{s.right_lane} at x={s.x_position} "
                f"(background discontinuity={s.background_discontinuity:.1f}, "
                f"confidence={s.confidence:.2f})"
            )

    return WesternBlotResult(
        suspicious=suspicious,
        band_profiles=profiles,
        duplicate_lanes=duplicate_lanes,
        splice_boundaries=splice_boundaries,
        lane_count=len(lanes),
        uniform_profiles=uniform_profiles,
        details="; ".join(details_parts) if details_parts else "No anomalies detected",
    )
