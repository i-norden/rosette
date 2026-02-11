"""Traditional CV-based image forensics analysis methods."""

import atexit
import io
import os
import shutil
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Managed temporary directory for ELA artifacts — cleaned up on process exit.
_temp_dir: str | None = None


def _get_temp_dir() -> str:
    """Return a temporary directory for ELA artifacts, creating it if needed."""
    global _temp_dir
    if _temp_dir is None or not os.path.isdir(_temp_dir):
        _temp_dir = tempfile.mkdtemp(prefix="snoopy_ela_")
        atexit.register(lambda: shutil.rmtree(_temp_dir, ignore_errors=True))
    return _temp_dir


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
    quality: int = 95,
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
    if suspicious:
        # Save the difference image for visual inspection
        ela_visual = np.clip(diff * (255.0 / max(max_diff, 1.0)), 0, 255).astype(np.uint8)
        ela_img = Image.fromarray(ela_visual)

        if output_dir:
            # Write to specified directory (persistent)
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ela_image_path = os.path.join(output_dir, f"{base_name}_ela.png")
            ela_img.save(ela_image_path)
        else:
            # Write to a managed temp directory that is cleaned up on process exit
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ela_image_path = os.path.join(
                _get_temp_dir(), f"{base_name}_ela.png"
            )
            ela_img.save(ela_image_path)

    return ELAResult(
        suspicious=suspicious,
        max_difference=max_diff,
        mean_difference=mean_diff,
        std_difference=std_diff,
        ela_image_path=ela_image_path,
    )


def clone_detection(
    image_path: str, min_matches: int = 30, min_inlier_ratio: float = 0.15,
) -> CloneResult:
    """Detect copy-move (clone) regions within an image using ORB features.

    Extracts keypoints with ORB, performs self-matching, and filters for
    geometrically consistent matches that indicate duplicated regions.

    Args:
        image_path: Path to the image file.
        min_matches: Minimum RANSAC inlier count to consider suspicious
            (default 30).
        min_inlier_ratio: Minimum ratio of RANSAC inliers to total filtered
            matches (default 0.15).  Low ratios indicate random feature
            coincidences rather than genuine cloned regions.

    Returns:
        CloneResult with suspicion flag and match cluster information.
    """
    img = cv2.imread(image_path)
    if img is None:
        return CloneResult(suspicious=False, num_matches=0, match_clusters=[], inlier_ratio=0.0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 2:
        return CloneResult(suspicious=False, num_matches=0, match_clusters=[], inlier_ratio=0.0)

    # Self-match using brute force with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
            max_dist = float(np.max(np.linalg.norm(cluster_arr - center, axis=1))) if len(cluster_arr) > 1 else 0.0
            match_clusters.append({
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "radius": max_dist,
                "num_points": len(cluster_pts),
            })

    suspicious = inlier_count >= min_matches and inlier_ratio >= min_inlier_ratio
    return CloneResult(
        suspicious=suspicious,
        num_matches=inlier_count,
        match_clusters=match_clusters,
        inlier_ratio=inlier_ratio,
    )


def noise_analysis(image_path: str, block_size: int = 64, intensity_bin_width: int = 32) -> NoiseResult:
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
        return NoiseResult(suspicious=False, noise_map=[], mean_noise=0.0, noise_std=0.0, max_ratio=0.0)

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

            noise_map.append({
                "block_x": x,
                "block_y": y,
                "noise_level": noise_level,
            })
            noise_values.append(noise_level)
            block_info.append((mean_intensity, noise_level))

    if not noise_values:
        return NoiseResult(suspicious=False, noise_map=[], mean_noise=0.0, noise_std=0.0, max_ratio=0.0)

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
        if min_level > 0:
            ratio = max_level / min_level
            max_ratio = max(max_ratio, ratio)

    suspicious = max_ratio > 10.0

    return NoiseResult(
        suspicious=suspicious,
        noise_map=noise_map,
        mean_noise=mean_noise,
        noise_std=noise_std,
        max_ratio=max_ratio,
    )
