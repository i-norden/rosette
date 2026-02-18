"""Traditional CV-based image forensics analysis methods."""

import io
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class ELAResult:
    """Result of Error Level Analysis."""

    suspicious: bool
    max_difference: float
    mean_difference: float
    std_difference: float
    ela_image_path: str | None


@dataclass
class CloneResult:
    """Result of copy-move clone detection."""

    suspicious: bool
    num_matches: int
    match_clusters: list[dict]
    inlier_ratio: float


@dataclass
class NoiseResult:
    """Result of noise inconsistency analysis."""

    suspicious: bool
    noise_map: list[dict]
    mean_noise: float
    noise_std: float
    max_ratio: float


def error_level_analysis(
    image_path: str,
    quality: int = 80,
    min_max_diff: float = 15.0,
    output_dir: str | None = None,
) -> ELAResult:
    """Perform Error Level Analysis on an image.

    Re-saves the image at a given JPEG quality and computes the pixel-level
    difference between the original and the resaved version. Regions that have
    been manipulated tend to show higher error levels than the rest of the
    image.

    Args:
        image_path: Path to the image file.
        quality: JPEG quality level for resaving (default 95).
        min_max_diff: Absolute minimum max-pixel-difference (0-255) required
            to flag suspicious.  Prevents trivial differences (e.g. 3/255)
            from triggering on clean images (default 15).
        output_dir: Directory for saving ELA artifact images. Required when
            the result is suspicious and an artifact image should be saved.

    Returns:
        ELAResult with suspicion flag and difference statistics.
    """
    original = Image.open(image_path).convert("RGB")

    # Re-save at the specified JPEG quality into memory
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    original_arr = np.array(original, dtype=np.float64)
    resaved_arr = np.array(resaved, dtype=np.float64)

    diff = np.abs(original_arr - resaved_arr)

    # Per-channel statistics
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    max_diff = float(np.max(diff))

    # Flag suspicious if the max difference exceeds both an absolute floor and
    # the statistical outlier threshold (mean + 2*std).
    suspicious = max_diff >= min_max_diff and max_diff > (mean_diff + 2.0 * std_diff)

    ela_image_path = None
    if suspicious and output_dir:
        # Save the difference image for visual inspection
        ela_visual = np.clip(diff * (255.0 / max(max_diff, 1.0)), 0, 255).astype(np.uint8)
        ela_img = Image.fromarray(ela_visual)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ela_image_path = os.path.join(output_dir, f"{base_name}_ela.png")
        ela_img.save(ela_image_path)

    return ELAResult(
        suspicious=suspicious,
        max_difference=max_diff,
        mean_difference=mean_diff,
        std_difference=std_diff,
        ela_image_path=ela_image_path,
    )


def clone_detection(
    image_path: str,
    min_matches: int = 30,
    min_inlier_ratio: float = 0.15,
    feature_extractor: str = "sift",
) -> CloneResult:
    """Detect copy-move (clone) regions within an image using feature matching.

    Extracts keypoints with SIFT (default) or ORB, performs self-matching, and
    filters for geometrically consistent matches that indicate duplicated
    regions.

    Args:
        image_path: Path to the image file.
        min_matches: Minimum RANSAC inlier count to consider suspicious
            (default 30).
        min_inlier_ratio: Minimum ratio of RANSAC inliers to total filtered
            matches (default 0.15).  Low ratios indicate random feature
            coincidences rather than genuine cloned regions.
        feature_extractor: Feature extraction method, either "sift" (default)
            or "orb".  SIFT produces floating-point descriptors matched with
            L2 norm; ORB produces binary descriptors matched with Hamming
            distance.

    Returns:
        CloneResult with suspicion flag and match cluster information.
    """
    img = cv2.imread(image_path)
    if img is None:
        return CloneResult(suspicious=False, num_matches=0, match_clusters=[], inlier_ratio=0.0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if feature_extractor == "sift":
        detector = cv2.SIFT_create(nfeatures=5000)  # type: ignore[attr-defined]
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000)  # type: ignore[attr-defined]
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        norm_type = cv2.NORM_HAMMING

    if descriptors is None or len(keypoints) < 2:
        return CloneResult(suspicious=False, num_matches=0, match_clusters=[], inlier_ratio=0.0)

    # Self-match using brute force with the appropriate distance norm
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors, descriptors, k=5)

    # Filter matches: remove self-matches and require spatial separation > 20px
    filtered_src = []
    filtered_dst = []
    for match_group in raw_matches:
        for m in match_group:
            if m.queryIdx == m.trainIdx:
                continue
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if dist > 20.0:
                filtered_src.append(pt1)
                filtered_dst.append(pt2)

    if len(filtered_src) < min_matches:
        return CloneResult(
            suspicious=False,
            num_matches=len(filtered_src),
            match_clusters=[],
            inlier_ratio=0.0,
        )

    src_pts = np.array(filtered_src, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(filtered_dst, dtype=np.float32).reshape(-1, 1, 2)

    # Use RANSAC to find geometrically consistent matches
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return CloneResult(
            suspicious=False,
            num_matches=len(filtered_src),
            match_clusters=[],
            inlier_ratio=0.0,
        )

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = inlier_count / len(filtered_src) if filtered_src else 0.0

    # Cluster inlier destination points to identify cloned regions
    inlier_pts = np.array(filtered_dst)[inlier_mask]
    match_clusters = []
    if len(inlier_pts) > 0:
        # Simple clustering: use connected components via distance threshold
        used = np.zeros(len(inlier_pts), dtype=bool)
        cluster_radius = 50.0
        for i in range(len(inlier_pts)):
            if used[i]:
                continue
            cluster_pts = [inlier_pts[i]]
            used[i] = True
            for j in range(i + 1, len(inlier_pts)):
                if used[j]:
                    continue
                if np.linalg.norm(inlier_pts[i] - inlier_pts[j]) < cluster_radius:
                    cluster_pts.append(inlier_pts[j])
                    used[j] = True
            cluster_arr = np.array(cluster_pts)
            center = np.mean(cluster_arr, axis=0)
            max_dist = (
                float(np.max(np.linalg.norm(cluster_arr - center, axis=1)))
                if len(cluster_arr) > 1
                else 0.0
            )
            match_clusters.append(
                {
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "radius": max_dist,
                    "num_points": len(cluster_pts),
                }
            )

    suspicious = inlier_count >= min_matches and inlier_ratio >= min_inlier_ratio
    return CloneResult(
        suspicious=suspicious,
        num_matches=inlier_count,
        match_clusters=match_clusters,
        inlier_ratio=inlier_ratio,
    )


def noise_analysis(
    image_path: str,
    block_size: int = 64,
    intensity_bin_width: int = 32,
    suspicious_threshold: float = 25.0,
) -> NoiseResult:
    """Analyse noise level inconsistencies across image blocks.

    Divides the image into blocks and computes the variance of the Laplacian
    for each block. Manipulated regions often exhibit different noise
    characteristics compared to the rest of the image.

    Args:
        image_path: Path to the image file.
        block_size: Size of the square blocks in pixels (default 64).

    Returns:
        NoiseResult with suspicion flag and per-block noise map.
    """
    img = cv2.imread(image_path)
    if img is None:
        return NoiseResult(
            suspicious=False, noise_map=[], mean_noise=0.0, noise_std=0.0, max_ratio=0.0
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    noise_map: list[dict] = []
    noise_values: list[float] = []
    block_info: list[tuple[float, float]] = []  # (mean_intensity, noise_level)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y : y + block_size, x : x + block_size]
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            noise_level = float(np.var(laplacian))
            mean_intensity = float(np.mean(block))

            noise_map.append(
                {
                    "block_x": x,
                    "block_y": y,
                    "noise_level": noise_level,
                }
            )
            noise_values.append(noise_level)
            block_info.append((mean_intensity, noise_level))

    if not noise_values:
        return NoiseResult(
            suspicious=False, noise_map=[], mean_noise=0.0, noise_std=0.0, max_ratio=0.0
        )

    noise_arr = np.array(noise_values)
    mean_noise = float(np.mean(noise_arr))
    noise_std = float(np.std(noise_arr))

    # Group blocks by similar mean intensity and check if max noise variance
    # exceeds 3 * min noise variance within groups.
    max_ratio = 0.0
    intensity_bins: dict[int, list[float]] = {}
    for mean_int, nlevel in block_info:
        bin_idx = int(mean_int // intensity_bin_width)
        intensity_bins.setdefault(bin_idx, []).append(nlevel)

    for bin_idx, levels in intensity_bins.items():
        if len(levels) < 2:
            continue
        min_level = min(levels)
        max_level = max(levels)
        if min_level == 0 and max_level > 0:
            # Zero noise in one block alongside non-zero = highly suspicious (synthetic)
            max_ratio = float("inf")
        elif min_level > 0:
            ratio = max_level / min_level
            max_ratio = max(max_ratio, ratio)

    suspicious = max_ratio > suspicious_threshold

    return NoiseResult(
        suspicious=suspicious,
        noise_map=noise_map,
        mean_noise=mean_noise,
        noise_std=noise_std,
        max_ratio=max_ratio,
    )


# ---------------------------------------------------------------------------
# DCT coefficient analysis
# ---------------------------------------------------------------------------


@dataclass
class DCTResult:
    """Result of DCT coefficient analysis for double JPEG compression detection."""

    suspicious: bool
    periodicity_score: float
    estimated_primary_quality: int | None
    block_inconsistencies: int
    details: str


def dct_analysis(
    image_path: str,
    periodicity_threshold: float = 0.3,
) -> DCTResult:
    """Analyse DCT coefficients to detect double JPEG compression.

    Divides the image into 8x8 blocks, computes the 2D DCT for each block,
    and builds histograms of quantized coefficients per DCT mode.  Periodic
    "comb" patterns in these histograms indicate that the image was JPEG
    compressed, decompressed, edited, and then re-compressed -- a strong sign
    of manipulation.

    Args:
        image_path: Path to the image file.
        periodicity_threshold: Minimum periodicity score (0-1) to flag the
            image as suspicious (default 0.3).

    Returns:
        DCTResult with suspicion flag and periodicity statistics.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return DCTResult(
            suspicious=False,
            periodicity_score=0.0,
            estimated_primary_quality=None,
            block_inconsistencies=0,
            details="Failed to read image.",
        )

    img_float = np.float32(img)
    h, w = img_float.shape  # type: ignore[misc]

    # Trim dimensions to multiples of 8
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    if h8 == 0 or w8 == 0:
        return DCTResult(
            suspicious=False,
            periodicity_score=0.0,
            estimated_primary_quality=None,
            block_inconsistencies=0,
            details="Image too small for 8x8 block DCT analysis.",
        )

    img_float = img_float[:h8, :w8]  # type: ignore[index]

    # Number of DCT modes to inspect (skip DC at [0,0])
    # We look at modes (0,1)..(7,7) = 63 AC modes
    num_modes = 63
    # Histogram bins: quantized coefficient values in [-127, 128] -> 256 bins
    num_bins = 256
    mode_histograms = np.zeros((num_modes, num_bins), dtype=np.float64)

    # Collect per-block first-AC energy for inconsistency detection
    block_energies: list[float] = []

    mode_coords: list[tuple[int, int]] = []
    for u in range(8):
        for v in range(8):
            if u == 0 and v == 0:
                continue
            mode_coords.append((u, v))

    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = img_float[y : y + 8, x : x + 8]
            dct_block = cv2.dct(block)

            # Sum of squared AC coefficients as block energy
            ac_energy = float(np.sum(dct_block**2) - dct_block[0, 0] ** 2)
            block_energies.append(ac_energy)

            for idx, (u, v) in enumerate(mode_coords):
                coeff = dct_block[u, v]
                # Quantize to integer and clamp to histogram range
                q = int(round(float(coeff)))
                q = max(-127, min(128, q))
                bin_index = q + 127  # map -127..128 -> 0..255
                mode_histograms[idx, bin_index] += 1

    # Detect periodicity in each mode histogram via autocorrelation.
    # A "comb" pattern from double compression shows periodic peaks whose
    # spacing corresponds to the original quantization table entry.
    periodicity_scores: list[float] = []
    best_periods: list[int] = []

    for idx in range(num_modes):
        hist = mode_histograms[idx]
        if np.sum(hist) == 0:
            periodicity_scores.append(0.0)
            best_periods.append(0)
            continue

        # Normalize histogram
        hist_norm = hist / np.sum(hist)
        # Remove DC (mean)
        hist_norm = hist_norm - np.mean(hist_norm)

        # Compute autocorrelation via FFT
        fft_hist = np.fft.fft(hist_norm)
        power = np.abs(fft_hist) ** 2
        autocorr = np.real(np.fft.ifft(power))

        # Normalize autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Search for peaks at lags 2..20 (typical quantization step sizes)
        max_score = 0.0
        best_period = 0
        for lag in range(2, min(21, len(autocorr) // 2)):
            score = float(autocorr[lag])
            if score > max_score:
                max_score = score
                best_period = lag

        periodicity_scores.append(max(0.0, max_score))
        best_periods.append(best_period)

    # Overall periodicity score: mean of top-10 mode periodicity scores
    sorted_scores = sorted(periodicity_scores, reverse=True)
    top_n = min(10, len(sorted_scores))
    overall_periodicity = float(np.mean(sorted_scores[:top_n])) if top_n > 0 else 0.0

    # Estimate primary quality from the most common best-period across modes.
    # The period approximates the quantization step; higher step = lower quality.
    estimated_quality: int | None = None
    if overall_periodicity >= periodicity_threshold and best_periods:
        # Filter to modes with meaningful periodicity
        significant_periods = [
            p
            for p, s in zip(best_periods, periodicity_scores)
            if s >= periodicity_threshold and p > 0
        ]
        if significant_periods:
            # Mode (most common) period
            from collections import Counter

            period_counts = Counter(significant_periods)
            dominant_period = period_counts.most_common(1)[0][0]
            # Rough mapping: quant_step ~ period, quality ~ 100 - step * 1.5
            estimated_quality = max(1, min(100, int(round(100 - dominant_period * 1.5))))

    # Detect block-level inconsistencies in AC energy.
    # Blocks that were edited after first compression will have different
    # energy distributions than untouched blocks.
    block_inconsistencies = 0
    if block_energies:
        energy_arr = np.array(block_energies)
        energy_mean = float(np.mean(energy_arr))
        energy_std = float(np.std(energy_arr))
        if energy_std > 0:
            # Blocks deviating by more than 3 sigma are inconsistent
            outlier_mask = np.abs(energy_arr - energy_mean) > 3.0 * energy_std
            block_inconsistencies = int(np.sum(outlier_mask))

    suspicious = overall_periodicity >= periodicity_threshold

    details_parts = [
        f"Periodicity score: {overall_periodicity:.4f} (threshold: {periodicity_threshold}).",
    ]
    if estimated_quality is not None:
        details_parts.append(f"Estimated primary JPEG quality: ~{estimated_quality}.")
    if block_inconsistencies > 0:
        details_parts.append(
            f"{block_inconsistencies} blocks show inconsistent AC energy "
            f"(possible localized editing after first compression)."
        )
    if suspicious:
        details_parts.append(
            "DCT histograms show periodic comb patterns consistent with double JPEG compression."
        )
    else:
        details_parts.append("No significant double-compression artifacts detected.")

    return DCTResult(
        suspicious=suspicious,
        periodicity_score=overall_periodicity,
        estimated_primary_quality=estimated_quality,
        block_inconsistencies=block_inconsistencies,
        details=" ".join(details_parts),
    )


# ---------------------------------------------------------------------------
# JPEG ghost detection
# ---------------------------------------------------------------------------


@dataclass
class JPEGGhostResult:
    """Result of JPEG ghost detection."""

    suspicious: bool
    ghost_regions: list[dict]
    quality_map: list[dict]
    dominant_quality: int
    quality_variance: float
    details: str


def jpeg_ghost_detection(
    image_path: str,
    quality_range: tuple[int, int] = (50, 95),
    step: int = 5,
    block_size: int = 64,
) -> JPEGGhostResult:
    """Detect JPEG ghost artifacts revealing mixed compression histories.

    Re-saves the image at multiple JPEG quality levels and computes per-block
    absolute differences between the original and each resaved version.  For
    each block the quality level that minimises the difference is recorded.
    If different blocks minimise at significantly different quality levels, the
    image likely contains regions with mixed compression histories -- a strong
    indicator of splicing or compositing.

    Args:
        image_path: Path to the image file.
        quality_range: (min_quality, max_quality) inclusive range of JPEG
            quality levels to test (default (50, 95)).
        step: Quality level increment (default 5).
        block_size: Side length of square blocks in pixels (default 64).

    Returns:
        JPEGGhostResult with suspicion flag, ghost region info, and quality
        distribution statistics.
    """
    original = Image.open(image_path).convert("RGB")
    original_arr = np.array(original, dtype=np.float64)
    h, w = original_arr.shape[:2]

    # Generate quality levels to test
    q_min, q_max = quality_range
    quality_levels = list(range(q_min, q_max + 1, step))
    if not quality_levels:
        return JPEGGhostResult(
            suspicious=False,
            ghost_regions=[],
            quality_map=[],
            dominant_quality=0,
            quality_variance=0.0,
            details="Empty quality range.",
        )

    # Trim to block-aligned dimensions
    h_blocks = h // block_size
    w_blocks = w // block_size
    if h_blocks == 0 or w_blocks == 0:
        return JPEGGhostResult(
            suspicious=False,
            ghost_regions=[],
            quality_map=[],
            dominant_quality=0,
            quality_variance=0.0,
            details="Image too small for block-based ghost analysis.",
        )

    h_trim = h_blocks * block_size
    w_trim = w_blocks * block_size
    original_trimmed = original_arr[:h_trim, :w_trim]

    # For each quality level, resave and compute per-block mean absolute diff
    # Shape: (num_qualities, h_blocks, w_blocks)
    diff_volumes = np.zeros((len(quality_levels), h_blocks, w_blocks), dtype=np.float64)

    for qi, quality in enumerate(quality_levels):
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")
        resaved_arr = np.array(resaved, dtype=np.float64)[:h_trim, :w_trim]

        block_diff = np.abs(original_trimmed - resaved_arr)

        for by in range(h_blocks):
            for bx in range(w_blocks):
                y0 = by * block_size
                x0 = bx * block_size
                block_region = block_diff[y0 : y0 + block_size, x0 : x0 + block_size]
                diff_volumes[qi, by, bx] = float(np.mean(block_region))

    # For each block, find the quality that minimises the difference
    best_quality_indices = np.argmin(diff_volumes, axis=0)  # (h_blocks, w_blocks)
    best_qualities = np.array(
        [
            [quality_levels[best_quality_indices[by, bx]] for bx in range(w_blocks)]
            for by in range(h_blocks)
        ],
        dtype=np.int32,
    )

    # Build the quality map
    quality_map: list[dict] = []
    all_best_qualities: list[int] = []
    for by in range(h_blocks):
        for bx in range(w_blocks):
            q = int(best_qualities[by, bx])
            min_diff = float(diff_volumes[best_quality_indices[by, bx], by, bx])
            quality_map.append(
                {
                    "block_x": bx * block_size,
                    "block_y": by * block_size,
                    "best_quality": q,
                    "min_difference": min_diff,
                }
            )
            all_best_qualities.append(q)

    quality_arr = np.array(all_best_qualities, dtype=np.float64)
    dominant_quality = int(np.median(quality_arr))
    quality_variance = float(np.var(quality_arr))

    # Identify ghost regions: blocks whose best quality deviates significantly
    # from the dominant quality (more than 1 step away)
    ghost_regions: list[dict] = []
    deviation_threshold = step * 2  # at least 2 steps away from dominant
    for entry in quality_map:
        if abs(entry["best_quality"] - dominant_quality) >= deviation_threshold:
            ghost_regions.append(
                {
                    "block_x": entry["block_x"],
                    "block_y": entry["block_y"],
                    "block_quality": entry["best_quality"],
                    "dominant_quality": dominant_quality,
                    "quality_deviation": abs(entry["best_quality"] - dominant_quality),
                }
            )

    # Suspicious if there are ghost regions and quality variance is high
    ghost_fraction = len(ghost_regions) / max(len(quality_map), 1)
    suspicious = len(ghost_regions) > 0 and (quality_variance > (step**2) or ghost_fraction > 0.05)

    details_parts = [
        f"Tested {len(quality_levels)} quality levels from {q_min} to {q_max}.",
        f"Dominant compression quality: {dominant_quality}.",
        f"Quality variance across blocks: {quality_variance:.2f}.",
        f"Ghost regions detected: {len(ghost_regions)} / {len(quality_map)} blocks.",
    ]
    if suspicious:
        details_parts.append(
            "Significant quality-level disagreement across blocks indicates "
            "mixed compression history (possible splicing or compositing)."
        )
    else:
        details_parts.append("Block quality levels are consistent; no ghost artifacts detected.")

    return JPEGGhostResult(
        suspicious=suspicious,
        ghost_regions=ghost_regions,
        quality_map=quality_map,
        dominant_quality=dominant_quality,
        quality_variance=quality_variance,
        details=" ".join(details_parts),
    )


# ---------------------------------------------------------------------------
# FFT frequency domain analysis
# ---------------------------------------------------------------------------


@dataclass
class FFTResult:
    """Result of FFT-based frequency domain analysis."""

    suspicious: bool
    spectral_anomaly_score: float
    periodic_peaks: list[float]
    high_freq_ratio: float
    details: str


def frequency_analysis(
    image_path: str,
    anomaly_threshold: float = 2.5,
) -> FFTResult:
    """Analyse the frequency spectrum of an image to detect manipulation.

    Computes the 2D FFT, builds a radial magnitude profile, and looks for:

    1. **Resampling artifacts** -- periodic peaks in the spectrum caused by
       interpolation during scaling, rotation, or warping.
    2. **Spectral anomalies** -- abnormal drops in high-frequency energy that
       suggest the image was synthesised by a GAN or diffusion model (which
       tend to under-represent high frequencies).

    Args:
        image_path: Path to the image file.
        anomaly_threshold: Z-score threshold for spectral peak detection
            (default 2.5).

    Returns:
        FFTResult with suspicion flag, anomaly score, detected peaks, and
        high-frequency energy ratio.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return FFTResult(
            suspicious=False,
            spectral_anomaly_score=0.0,
            periodic_peaks=[],
            high_freq_ratio=0.0,
            details="Failed to read image.",
        )

    img_float = np.float64(img)
    h, w = img_float.shape  # type: ignore[misc]

    # Compute 2D FFT and shift zero-frequency to centre
    f_transform = np.fft.fft2(img_float)
    f_shifted = np.fft.fftshift(f_transform)

    # Magnitude spectrum in log scale (add 1 to avoid log(0))
    magnitude = np.log1p(np.abs(f_shifted))

    cy, cx = h // 2, w // 2
    max_radius = int(min(cy, cx))

    if max_radius < 2:
        return FFTResult(
            suspicious=False,
            spectral_anomaly_score=0.0,
            periodic_peaks=[],
            high_freq_ratio=0.0,
            details="Image too small for meaningful frequency analysis.",
        )

    # Build radial profile: mean magnitude at each integer radius
    # Pre-compute radius map for efficiency
    y_coords, x_coords = np.ogrid[:h, :w]
    radius_map = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    radial_profile = np.zeros(max_radius, dtype=np.float64)
    radial_counts = np.zeros(max_radius, dtype=np.float64)

    # Bin each pixel by its integer radius
    radius_int = np.clip(radius_map.astype(np.int64), 0, max_radius - 1)
    for r in range(max_radius):
        mask = radius_int == r
        pixels_at_r = magnitude[mask]
        if len(pixels_at_r) > 0:
            radial_profile[r] = float(np.mean(pixels_at_r))
            radial_counts[r] = len(pixels_at_r)

    # --- Detect resampling artifacts: periodic peaks in radial profile ---
    # Detrend the profile (remove smooth baseline) with a moving average
    if max_radius > 10:
        kernel_size = max(3, max_radius // 20)
        kernel = np.ones(kernel_size) / kernel_size
        smooth_profile = np.convolve(radial_profile, kernel, mode="same")
        detrended = radial_profile - smooth_profile
    else:
        detrended = radial_profile - np.mean(radial_profile)

    # Find peaks via FFT of the detrended radial profile
    profile_fft = np.fft.fft(detrended)
    profile_power = np.abs(profile_fft[: max_radius // 2])

    # Skip DC component (index 0) and very low frequencies (index 1)
    if len(profile_power) > 2:
        search_power = profile_power[2:]
        power_mean = float(np.mean(search_power))
        power_std = float(np.std(search_power))

        periodic_peaks: list[float] = []
        if power_std > 0:
            z_scores = (search_power - power_mean) / power_std
            for i, z in enumerate(z_scores):
                if z > anomaly_threshold:
                    # Convert index to frequency (cycles per max_radius)
                    freq = float(i + 2) / max_radius
                    periodic_peaks.append(round(freq, 4))
    else:
        periodic_peaks = []

    # --- Detect high-frequency anomalies (GAN/diffusion signatures) ---
    # Split the radial profile into low-freq (inner 1/3) and high-freq (outer 1/3)
    boundary_low = max_radius // 3
    boundary_high = 2 * max_radius // 3

    low_freq_energy = (
        float(np.mean(radial_profile[1 : boundary_low + 1])) if boundary_low > 1 else 0.0
    )
    high_freq_energy = (
        float(np.mean(radial_profile[boundary_high:])) if boundary_high < max_radius else 0.0
    )

    # Ratio of high-freq to low-freq energy
    high_freq_ratio = high_freq_energy / max(low_freq_energy, 1e-10)

    # Compute spectral anomaly score.
    # Natural images follow an approximate 1/f power law.  We measure how much
    # the actual profile deviates from the expected falloff.
    # Fit a simple 1/f model: expected[r] = A / (r + 1)
    radii = np.arange(1, max_radius, dtype=np.float64)
    actual = radial_profile[1:]
    if len(radii) > 0 and np.sum(actual) > 0:
        # Least-squares fit: A = sum(actual * (r+1)) / sum(1)
        # More robust: fit in log-log space
        log_r = np.log1p(radii)
        log_actual = np.log1p(np.maximum(actual, 0))

        # Expected: log(mag) = a - b * log(r) for natural images
        # Fit linear regression in log-log
        valid = log_actual > 0
        if np.sum(valid) > 2:
            X = np.column_stack([np.ones(int(np.sum(valid))), log_r[valid]])
            y_fit = log_actual[valid]
            # Normal equation: beta = (X^T X)^-1 X^T y
            try:
                beta = np.linalg.lstsq(X, y_fit, rcond=None)[0]
                predicted = X @ beta
                residuals = y_fit - predicted
                spectral_anomaly_score = float(np.sqrt(np.mean(residuals**2)))
            except np.linalg.LinAlgError:
                spectral_anomaly_score = 0.0
        else:
            spectral_anomaly_score = 0.0
    else:
        spectral_anomaly_score = 0.0

    # Determine suspicion
    has_resampling_peaks = len(periodic_peaks) > 0
    # Very low high_freq_ratio suggests synthetic origin (GAN/diffusion)
    has_hf_dropout = high_freq_ratio < 0.1
    suspicious = has_resampling_peaks or has_hf_dropout

    details_parts = [
        f"Spectral anomaly score (RMSE of 1/f fit): {spectral_anomaly_score:.4f}.",
        f"High-frequency energy ratio: {high_freq_ratio:.4f}.",
    ]
    if has_resampling_peaks:
        details_parts.append(
            f"Detected {len(periodic_peaks)} periodic peak(s) in the radial "
            f"spectrum at normalised frequencies: {periodic_peaks}. "
            "This may indicate resampling (resize/rotate) artifacts."
        )
    if has_hf_dropout:
        details_parts.append(
            "Abnormally low high-frequency energy detected. This spectral "
            "profile is consistent with GAN or diffusion-model generated images."
        )
    if not suspicious:
        details_parts.append("Frequency spectrum is consistent with natural photographic content.")

    return FFTResult(
        suspicious=suspicious,
        spectral_anomaly_score=spectral_anomaly_score,
        periodic_peaks=periodic_peaks,
        high_freq_ratio=high_freq_ratio,
        details=" ".join(details_parts),
    )
