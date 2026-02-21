"""Shared analysis functions used by both production orchestrator and demo runner.

Provides config-driven wrappers around the underlying analysis methods that
return standardized finding dicts. This eliminates the duplication between
``rosette.pipeline.orchestrator`` and ``rosette.demo.runner``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rosette.analysis.cross_reference import hash_distance
from rosette.analysis.image_forensics import (
    BlockCloneResult,
    CloneResult,
    DCTResult,
    ELAResult,
    FFTResult,
    JPEGGhostResult,
    NoiseResult,
    block_clone_detection,
    clone_detection,
    dct_analysis,
    error_level_analysis,
    frequency_analysis,
    jpeg_ghost_detection,
    noise_analysis,
)
from rosette.config import AnalysisConfig

logger = logging.getLogger(__name__)


def _default_config() -> AnalysisConfig:
    """Return a default AnalysisConfig for standalone use."""
    return AnalysisConfig()


def run_image_forensics(
    image_path: str,
    figure_id: str = "",
    config: AnalysisConfig | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """Run all image forensics methods on a single image.

    Returns a list of standardized finding dicts.
    """
    cfg = config or _default_config()
    findings: list[dict] = []
    path_str = str(image_path)
    fig_id = figure_id or Path(image_path).name

    # Pre-load image data once to avoid redundant file reads.
    # Two grayscale variants are needed because cv2.IMREAD_GRAYSCALE
    # (direct Y-channel decode) produces different pixel values than
    # cv2.cvtColor(bgr, COLOR_BGR2GRAY) for JPEG images due to
    # intermediate rounding in the YCbCr→RGB→Gray chain.
    cv2_bgr = cv2.imread(path_str)
    cv2_gray_cvt: np.ndarray | None = None  # for clone_detection, noise_analysis
    cv2_gray_direct: np.ndarray | None = None  # for block_clone, dct, frequency
    pil_rgb: Image.Image | None = None
    if cv2_bgr is not None:
        cv2_gray_cvt = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2GRAY)
    cv2_gray_direct = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    try:
        pil_rgb = Image.open(path_str).convert("RGB")
    except Exception:
        pass

    # Define each analysis step as a closure returning a list of findings.
    # Each closure has its own error handling for isolation.

    def _run_ela() -> list[dict]:
        results: list[dict] = []
        try:
            ela: ELAResult = error_level_analysis(
                path_str,
                quality=cfg.ela.quality,
                output_dir=output_dir,
                _preloaded_pil=pil_rgb,
            )
            if ela.suspicious:
                max_diff = ela.max_difference
                mean_diff = ela.mean_difference
                std_diff = ela.std_difference

                epsilon = 1e-6
                z_score = (max_diff - mean_diff) / (std_diff + epsilon)

                if max_diff >= cfg.ela.high_threshold and max_diff > mean_diff + 3 * std_diff:
                    severity = "high"
                    confidence = min(max(z_score / 15.0, max_diff / 100.0), 0.85)
                elif max_diff >= cfg.ela.medium_threshold and max_diff > mean_diff + 3 * std_diff:
                    severity = "medium"
                    confidence = min(max(z_score / 20.0, max_diff / 70.0), 0.75)
                elif max_diff >= cfg.ela.low_threshold and max_diff > mean_diff + 2 * std_diff:
                    severity = "low"
                    confidence = min(max(z_score / 25.0, max_diff / 50.0), 0.65)
                else:
                    severity = "low"
                    confidence = min(max(z_score / 30.0, max_diff / 60.0), 0.50)

                results.append(
                    {
                        "title": "ELA anomaly detected",
                        "analysis_type": "ela",
                        "method": "ela",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"Max diff={ela.max_difference:.1f}, "
                            f"mean={ela.mean_difference:.1f}, std={ela.std_difference:.1f}"
                        ),
                        "figure_id": fig_id,
                        "evidence": {
                            "max_difference": round(ela.max_difference, 2),
                            "mean_difference": round(ela.mean_difference, 2),
                            "std_difference": round(ela.std_difference, 2),
                            "ela_image_path": ela.ela_image_path,
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("ELA failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("ELA unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_clone() -> list[dict]:
        results: list[dict] = []
        try:
            clone: CloneResult = clone_detection(
                path_str,
                min_matches=cfg.clone.min_matches,
                min_inlier_ratio=cfg.clone.min_inlier_ratio,
                _preloaded_gray=cv2_gray_cvt,
            )
            if clone.suspicious:
                inliers = clone.num_matches
                ratio = clone.inlier_ratio

                if inliers >= cfg.clone.high_inliers and ratio >= cfg.clone.high_ratio:
                    severity = "high"
                    confidence = min(ratio, 1.0) * 0.85
                elif inliers >= cfg.clone.medium_inliers and ratio >= cfg.clone.medium_ratio:
                    severity = "medium"
                    confidence = min(ratio, 1.0) * 0.7
                elif inliers >= cfg.clone.low_inliers and ratio >= cfg.clone.low_ratio:
                    severity = "low"
                    confidence = min(ratio, 1.0) * 0.5
                else:
                    severity = "low"
                    confidence = min(ratio, 1.0) * 0.4

                results.append(
                    {
                        "title": "Copy-move cloning detected",
                        "analysis_type": "clone_detection",
                        "method": "clone_detection",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"{clone.num_matches} matches, "
                            f"{len(clone.match_clusters)} clusters, "
                            f"inlier ratio={clone.inlier_ratio:.2f}"
                        ),
                        "figure_id": fig_id,
                        "evidence": {
                            "num_matches": clone.num_matches,
                            "num_clusters": len(clone.match_clusters),
                            "inlier_ratio": round(clone.inlier_ratio, 3),
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("Clone detection failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("Clone detection unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_block_clone() -> list[dict]:
        results: list[dict] = []
        try:
            block_clone: BlockCloneResult = block_clone_detection(
                path_str, _preloaded_gray=cv2_gray_direct
            )
            if block_clone.suspicious:
                cons = block_clone.consistency
                area = block_clone.clone_area_px
                psim = block_clone.pixel_similarity

                if cons >= 0.7 and psim >= 0.3:
                    severity = "high"
                    confidence = min(0.65 + cons * 0.2, 0.85)
                elif cons >= 0.5 and psim >= 0.15:
                    severity = "medium"
                    confidence = min(0.45 + cons * 0.3, 0.75)
                else:
                    severity = "low"
                    confidence = min(0.30 + cons * 0.3, 0.60)

                results.append(
                    {
                        "title": "Block-level copy-move detected",
                        "analysis_type": "clone_detection",
                        "method": "clone_detection",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"Block copy-move: {block_clone.num_matching_blocks} matching blocks, "
                            f"consistency={cons:.2f}, displacement={block_clone.displacement}, "
                            f"clone area={area}px², pixel similarity={psim:.2f}"
                        ),
                        "figure_id": fig_id,
                        "evidence": {
                            "num_matching_blocks": block_clone.num_matching_blocks,
                            "consistency": round(cons, 3),
                            "displacement": list(block_clone.displacement),
                            "clone_area_px": area,
                            "pixel_similarity": round(psim, 3),
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("Block clone detection failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("Block clone detection unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_noise() -> list[dict]:
        results: list[dict] = []
        try:
            noise_result: NoiseResult = noise_analysis(
                path_str,
                block_size=cfg.noise.block_size,
                intensity_bin_width=cfg.noise.intensity_bin_width,
                suspicious_threshold=cfg.noise.max_ratio_threshold,
                _preloaded_gray=cv2_gray_cvt,
            )
            if noise_result.suspicious:
                max_ratio = noise_result.max_ratio

                if max_ratio > cfg.noise.high_ratio:
                    severity = "high"
                    confidence = min(max_ratio / 80.0, 0.85)
                elif max_ratio > cfg.noise.medium_ratio:
                    severity = "medium"
                    confidence = min(max_ratio / 60.0, 0.7)
                elif max_ratio > cfg.noise.low_ratio:
                    severity = "low"
                    confidence = min(max_ratio / 40.0, 0.5)
                else:
                    severity = "low"
                    confidence = min(max_ratio / 40.0, 0.4)

                results.append(
                    {
                        "title": "Noise inconsistency detected",
                        "analysis_type": "noise_analysis",
                        "method": "noise_analysis",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"Max noise ratio={noise_result.max_ratio:.1f}, "
                            f"mean={noise_result.mean_noise:.1f}, std={noise_result.noise_std:.1f}"
                        ),
                        "figure_id": fig_id,
                        "evidence": {
                            "max_ratio": round(noise_result.max_ratio, 2),
                            "mean_noise": round(noise_result.mean_noise, 2),
                            "noise_std": round(noise_result.noise_std, 2),
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("Noise analysis failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("Noise analysis unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_jpeg_ghost() -> list[dict]:
        results: list[dict] = []
        try:
            ghost_result: JPEGGhostResult = jpeg_ghost_detection(
                path_str,
                quality_range=(
                    cfg.jpeg_ghost_quality_range_start,
                    cfg.jpeg_ghost_quality_range_end,
                ),
                step=cfg.jpeg_ghost_step,
                _preloaded_pil=pil_rgb,
            )
            if ghost_result.suspicious:
                n_regions = len(ghost_result.ghost_regions)
                if n_regions >= 3 or ghost_result.quality_variance > 100:
                    severity = "high"
                    confidence = min(0.85, 0.5 + n_regions * 0.1)
                elif n_regions >= 1:
                    severity = "medium"
                    confidence = min(0.7, 0.4 + n_regions * 0.1)
                else:
                    severity = "low"
                    confidence = 0.4
                results.append(
                    {
                        "title": "JPEG ghost regions detected",
                        "analysis_type": "jpeg_ghost",
                        "method": "jpeg_ghost",
                        "severity": severity,
                        "confidence": confidence,
                        "description": ghost_result.details,
                        "figure_id": fig_id,
                        "evidence": {
                            "ghost_region_count": n_regions,
                            "dominant_quality": ghost_result.dominant_quality,
                            "quality_variance": round(ghost_result.quality_variance, 2),
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("JPEG ghost detection failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("JPEG ghost detection unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_dct() -> list[dict]:
        results: list[dict] = []
        try:
            dct_result: DCTResult = dct_analysis(
                path_str,
                periodicity_threshold=cfg.dct_periodicity_threshold,
                _preloaded_gray=cv2_gray_direct,
            )
            if dct_result.suspicious:
                score = dct_result.periodicity_score
                if score > 0.7:
                    severity = "high"
                    confidence = min(score, 0.9)
                elif score > 0.5:
                    severity = "medium"
                    confidence = min(score * 0.8, 0.75)
                else:
                    severity = "low"
                    confidence = min(score * 0.6, 0.5)

                results.append(
                    {
                        "title": "Double JPEG compression detected (DCT analysis)",
                        "analysis_type": "dct_analysis",
                        "method": "dct_analysis",
                        "severity": severity,
                        "confidence": confidence,
                        "description": dct_result.details,
                        "figure_id": fig_id,
                        "evidence": {
                            "periodicity_score": round(dct_result.periodicity_score, 3),
                            "estimated_primary_quality": dct_result.estimated_primary_quality,
                            "block_inconsistencies": dct_result.block_inconsistencies,
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("DCT analysis failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("DCT analysis unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_fft() -> list[dict]:
        results: list[dict] = []
        try:
            fft_result: FFTResult = frequency_analysis(
                path_str,
                anomaly_threshold=cfg.fft_spectral_anomaly_threshold,
                _preloaded_gray=cv2_gray_direct,
            )
            if fft_result.suspicious:
                score = fft_result.spectral_anomaly_score
                if score > 5.0:
                    severity = "high"
                    confidence = min(score / 10.0, 0.85)
                elif score > 3.0:
                    severity = "medium"
                    confidence = min(score / 8.0, 0.7)
                else:
                    severity = "low"
                    confidence = min(score / 6.0, 0.5)

                results.append(
                    {
                        "title": "Frequency domain anomaly detected (FFT)",
                        "analysis_type": "fft_analysis",
                        "method": "fft_analysis",
                        "severity": severity,
                        "confidence": confidence,
                        "description": fft_result.details,
                        "figure_id": fig_id,
                        "evidence": {
                            "spectral_anomaly_score": round(fft_result.spectral_anomaly_score, 3),
                            "periodic_peaks": fft_result.periodic_peaks[:5],
                            "high_freq_ratio": round(fft_result.high_freq_ratio, 4),
                        },
                    }
                )
        except (OSError, ValueError) as exc:
            logger.warning("FFT analysis failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("FFT analysis unexpected error on %s: %s", fig_id, exc)
        return results

    def _run_metadata() -> list[dict]:
        results: list[dict] = []
        try:
            from rosette.analysis.metadata_forensics import analyze_metadata

            meta = analyze_metadata(path_str, _preloaded_pil=pil_rgb)
            if meta.suspicious:
                for mf in meta.findings:
                    results.append(
                        {
                            "title": f"Metadata: {mf.finding_type}",
                            "analysis_type": "metadata_forensics",
                            "method": "metadata_forensics",
                            "severity": mf.severity,
                            "confidence": mf.confidence,
                            "description": mf.description,
                            "figure_id": fig_id,
                            "evidence": {
                                "software": meta.software,
                                "icc_profile": meta.icc_profile,
                            },
                        }
                    )
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Metadata forensics failed on %s: %s", fig_id, exc)
        except Exception as exc:
            logger.exception("Metadata forensics unexpected error on %s: %s", fig_id, exc)
        return results

    # Run all analysis steps sequentially
    for fn in (
        _run_ela,
        _run_clone,
        _run_block_clone,
        _run_noise,
        _run_jpeg_ghost,
        _run_dct,
        _run_fft,
        _run_metadata,
    ):
        findings.extend(fn())

    return findings


def run_intra_paper_cross_ref(
    figure_results: list[dict],
    config: AnalysisConfig | None = None,
) -> list[dict]:
    """Compare figure hashes within the same paper to find duplicates.

    Uses Union-Find to cluster matching images and emits one finding per
    cluster instead of one per pairwise match.  This prevents papers with
    many similar-looking panels from generating a combinatorial explosion
    of findings.

    Phash findings use a dedicated ``figure_id`` (``phash_cluster_N``) so
    they do not create false convergence with per-image ELA/noise findings
    in the evidence aggregation layer.

    Args:
        figure_results: List of dicts, each with 'image' (name), 'phash', and 'ahash' keys.

    Returns:
        List of finding dicts for any intra-paper hash matches.
    """
    # Union-Find helpers
    parent: dict[int, int] = {}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # Collect pairwise matches and build clusters
    edges: list[tuple[int, int, int]] = []  # (i, j, distance)

    for i, res_a in enumerate(figure_results):
        # Skip small images (icons, scale bars) that produce false hash matches
        width_a = res_a.get("width", 999)
        height_a = res_a.get("height", 999)
        if width_a < 100 or height_a < 100:
            continue

        for j, res_b in enumerate(figure_results):
            if j <= i:
                continue

            width_b = res_b.get("width", 999)
            height_b = res_b.get("height", 999)
            if width_b < 100 or height_b < 100:
                continue

            phash_a = res_a.get("phash")
            phash_b = res_b.get("phash")
            if phash_a and phash_b:
                try:
                    dist = hash_distance(phash_a, phash_b)
                except ValueError:
                    continue

                if dist <= 10:
                    edges.append((i, j, dist))
                    _union(i, j)

    if not edges:
        return []

    # Group edges by cluster root
    clusters: dict[int, list[tuple[int, int, int]]] = {}
    for i, j, dist in edges:
        root = _find(i)
        clusters.setdefault(root, []).append((i, j, dist))

    # Emit one finding per cluster
    findings: list[dict] = []
    for cluster_idx, (root, cluster_edges) in enumerate(clusters.items()):
        # Collect all images in this cluster
        members: set[int] = set()
        min_dist = min(d for _, _, d in cluster_edges)
        for i, j, _ in cluster_edges:
            members.add(i)
            members.add(j)
        member_names = sorted(figure_results[m]["image"] for m in members)

        if min_dist <= 2:
            severity = "critical"
            confidence = 0.95
        elif min_dist <= 5:
            severity = "high"
            confidence = 0.8
        else:
            severity = "medium"
            confidence = 0.6

        # Build pairwise detail for the evidence blob
        pair_details = []
        for i, j, dist in cluster_edges:
            pair_details.append(
                {
                    "figure_a": figure_results[i]["image"],
                    "figure_b": figure_results[j]["image"],
                    "hash_distance": dist,
                }
            )

        if len(member_names) == 2:
            title = f"Perceptual hash match: {member_names[0]} \u2194 {member_names[1]}"
            desc = (
                f"Figures '{member_names[0]}' and '{member_names[1]}' "
                f"have perceptual hash distance {min_dist} (possible duplicate/recycled)"
            )
        else:
            title = f"Perceptual hash cluster: {len(member_names)} similar figures"
            desc = (
                f"{len(member_names)} figures share similar perceptual hashes "
                f"(min distance {min_dist}): {', '.join(member_names[:5])}"
            )
            if len(member_names) > 5:
                desc += f" and {len(member_names) - 5} more"

        findings.append(
            {
                "title": title,
                "analysis_type": "phash",
                "method": "phash",
                "severity": severity,
                "confidence": confidence,
                "description": desc,
                "figure_id": f"phash_cluster_{cluster_idx}",
                "evidence": {
                    "cluster_members": member_names,
                    "pairs": pair_details,
                    "min_distance": min_dist,
                },
            }
        )

    return findings


def run_sprite_analysis(
    reported_mean: float,
    reported_sd: float,
    n: int,
    min_val: int = 1,
    max_val: int = 7,
    context: str = "",
) -> list[dict]:
    """Run SPRITE consistency test on reported mean/SD.

    Returns a list of standardized finding dicts.
    """
    findings: list[dict] = []

    try:
        from rosette.analysis.sprite import sprite_test

        result = sprite_test(
            reported_mean=reported_mean,
            reported_sd=reported_sd,
            n=n,
            min_val=min_val,
            max_val=max_val,
        )
        if not result.consistent:
            severity = "high" if not result.sd_achievable else "medium"
            confidence = 0.8 if not result.sd_achievable else 0.65

            findings.append(
                {
                    "title": f"SPRITE inconsistency: M={reported_mean}, SD={reported_sd}, N={n}",
                    "analysis_type": "sprite",
                    "method": "sprite",
                    "severity": severity,
                    "confidence": confidence,
                    "description": result.details,
                    "figure_id": "",
                    "evidence": {
                        "reported_mean": reported_mean,
                        "reported_sd": reported_sd,
                        "n": n,
                        "mean_achievable": result.mean_achievable,
                        "sd_achievable": result.sd_achievable,
                        "attempts": result.attempts,
                        "context": context[:200],
                    },
                }
            )
    except (ValueError, ImportError) as exc:
        logger.warning("SPRITE test failed: %s", exc)
    except Exception as exc:
        logger.exception("SPRITE test unexpected error: %s", exc)

    return findings


def run_statistical_tests(
    text: str,
    config: AnalysisConfig | None = None,
) -> list[dict]:
    """Run new statistical tests (GRIMMER, terminal digit, variance ratio) on extracted data.

    Returns a list of standardized finding dicts.
    """
    from rosette.analysis.statistical import (
        grimmer_test,
        terminal_digit_test,
        variance_ratio_test,
    )
    from rosette.extraction.stats_extractor import (
        extract_means_sds_and_ns,
        extract_numerical_values,
    )

    cfg = config or _default_config()
    findings: list[dict] = []

    # GRIMMER test on mean/SD/N reports
    try:
        mean_reports = extract_means_sds_and_ns(text)
        sd_n_pairs: list[tuple[float, int]] = []

        for mr in mean_reports:
            if mr.sd is not None and mr.n >= 2:
                sd_n_pairs.append((mr.sd, mr.n))
                result = grimmer_test(mr.mean, mr.sd, mr.n)
                if not result.consistent:
                    findings.append(
                        {
                            "title": f"GRIMMER inconsistency: M={mr.mean}, SD={mr.sd}, N={mr.n}",
                            "analysis_type": "grimmer",
                            "method": "grimmer",
                            "severity": "high",
                            "confidence": 0.8,
                            "description": result.details,
                            "figure_id": "",
                            "evidence": {
                                "mean": mr.mean,
                                "sd": mr.sd,
                                "n": mr.n,
                                "possible_sds": result.possible_sds[:5],
                                "context": mr.context[:200],
                            },
                        }
                    )
    except (ValueError, TypeError) as exc:
        logger.warning("GRIMMER test failed: %s", exc)
    except Exception as exc:
        logger.exception("GRIMMER test unexpected error: %s", exc)

    # Variance ratio test on collected SD/N pairs
    try:
        if len(sd_n_pairs) >= cfg.statistical.variance_ratio_min_sds:
            vr_result = variance_ratio_test(sd_n_pairs)
            if vr_result.suspicious:
                findings.append(
                    {
                        "title": "Variance ratio inconsistency across reported SDs",
                        "analysis_type": "variance_ratio",
                        "method": "variance_ratio",
                        "severity": "high" if vr_result.p_value < 0.01 else "medium",
                        "confidence": min(1.0 - vr_result.p_value, 0.9),
                        "description": vr_result.details,
                        "figure_id": "",
                        "evidence": {
                            "n_groups": vr_result.n_groups,
                            "observed_variance": round(vr_result.observed_variance_of_sds, 4),
                            "expected_variance": round(vr_result.expected_variance_of_sds, 4),
                            "ratio": round(vr_result.ratio, 4),
                            "p_value": round(vr_result.p_value, 6),
                        },
                    }
                )
    except (ValueError, TypeError) as exc:
        logger.warning("Variance ratio test failed: %s", exc)
    except Exception as exc:
        logger.exception("Variance ratio test unexpected error: %s", exc)

    # Terminal digit test on all extracted numerical values
    try:
        values = extract_numerical_values(text)
        if len(values) >= 20:
            td_result = terminal_digit_test(
                values, alpha=cfg.statistical.terminal_digit_uniformity_alpha
            )
            if td_result.suspicious:
                findings.append(
                    {
                        "title": "Non-uniform terminal digit distribution",
                        "analysis_type": "terminal_digit",
                        "method": "terminal_digit",
                        "severity": "medium" if td_result.p_value < 0.001 else "low",
                        "confidence": min(1.0 - td_result.p_value, 0.8),
                        "description": td_result.details,
                        "figure_id": "",
                        "evidence": {
                            "chi_squared": round(td_result.chi_squared, 2),
                            "p_value": round(td_result.p_value, 6),
                            "n_values": td_result.n_values,
                            "digit_counts": td_result.digit_counts,
                        },
                    }
                )
    except (ValueError, TypeError) as exc:
        logger.warning("Terminal digit test failed: %s", exc)
    except Exception as exc:
        logger.exception("Terminal digit test unexpected error: %s", exc)

    return findings


def run_tortured_phrases(
    text: str,
    config: AnalysisConfig | None = None,
) -> list[dict]:
    """Run tortured phrase detection on paper text.

    Returns a list of standardized finding dicts.
    """
    cfg = config or _default_config()
    findings: list[dict] = []

    try:
        from rosette.analysis.text_forensics import detect_tortured_phrases

        result = detect_tortured_phrases(
            text, min_matches=cfg.statistical.tortured_phrase_min_matches
        )
        if result.suspicious:
            if result.unique_phrases >= 5:
                severity = "critical"
                confidence = min(0.95, 0.6 + result.unique_phrases * 0.05)
            elif result.unique_phrases >= 3:
                severity = "high"
                confidence = min(0.85, 0.5 + result.unique_phrases * 0.1)
            else:
                severity = "medium"
                confidence = min(0.7, 0.4 + result.unique_phrases * 0.1)

            match_details = "; ".join(
                f"'{m.tortured_phrase}' (should be '{m.correct_phrase}')"
                for m in result.matches[:10]
            )
            findings.append(
                {
                    "title": f"Tortured phrases detected ({result.unique_phrases} unique)",
                    "analysis_type": "tortured_phrases",
                    "method": "tortured_phrases",
                    "severity": severity,
                    "confidence": confidence,
                    "description": f"{result.details}. Examples: {match_details}",
                    "figure_id": "",
                    "evidence": {
                        "match_count": result.match_count,
                        "unique_phrases": result.unique_phrases,
                        "matches": [
                            {
                                "phrase": m.tortured_phrase,
                                "correct": m.correct_phrase,
                            }
                            for m in result.matches[:20]
                        ],
                    },
                }
            )
    except (ValueError, ImportError) as exc:
        logger.warning("Tortured phrase detection failed: %s", exc)
    except Exception as exc:
        logger.exception("Tortured phrase detection unexpected error: %s", exc)

    return findings


def run_western_blot_analysis(
    image_path: str,
    figure_id: str = "",
) -> list[dict]:
    """Run western blot specific analysis on an image.

    Returns a list of finding dicts.
    """
    findings: list[dict] = []
    fig_id = figure_id or Path(image_path).name

    try:
        from rosette.analysis.western_blot import analyze_western_blot

        result = analyze_western_blot(image_path)
        if result.suspicious:
            # Duplicate lanes
            for lane_i, lane_j, corr in result.duplicate_lanes:
                findings.append(
                    {
                        "title": f"Duplicate lanes in western blot (lanes {lane_i}, {lane_j})",
                        "analysis_type": "western_blot",
                        "method": "western_blot",
                        "severity": "high",
                        "confidence": min(corr, 1.0),
                        "description": (
                            f"Lanes {lane_i} and {lane_j} have near-identical intensity profiles "
                            f"(correlation={corr:.3f})"
                        ),
                        "figure_id": fig_id,
                        "evidence": {
                            "lane_a": lane_i,
                            "lane_b": lane_j,
                            "correlation": round(corr, 4),
                        },
                    }
                )

            # Splice boundaries
            for splice in result.splice_boundaries:
                if splice.confidence > 0.5:
                    findings.append(
                        {
                            "title": "Splice boundary in western blot",
                            "analysis_type": "western_blot",
                            "method": "western_blot",
                            "severity": "high" if splice.confidence > 0.7 else "medium",
                            "confidence": splice.confidence,
                            "description": (
                                f"Splice boundary detected between lanes {splice.left_lane} and "
                                f"{splice.right_lane} at x={splice.x_position}"
                            ),
                            "figure_id": fig_id,
                            "evidence": {
                                "x_position": splice.x_position,
                                "background_discontinuity": round(
                                    splice.background_discontinuity, 2
                                ),
                                "noise_discontinuity": round(splice.noise_discontinuity, 3),
                            },
                        }
                    )

            # Uniform profiles
            if result.uniform_profiles and not result.duplicate_lanes:
                findings.append(
                    {
                        "title": "Suspiciously uniform western blot profiles",
                        "analysis_type": "western_blot",
                        "method": "western_blot",
                        "severity": "medium",
                        "confidence": 0.6,
                        "description": (
                            f"All {result.lane_count} lanes show suspiciously uniform "
                            f"intensity profiles"
                        ),
                        "figure_id": fig_id,
                        "evidence": {"lane_count": result.lane_count},
                    }
                )
    except (OSError, ValueError, ImportError) as exc:
        logger.warning("Western blot analysis failed on %s: %s", fig_id, exc)
    except Exception as exc:
        logger.exception("Western blot analysis unexpected error on %s: %s", fig_id, exc)

    return findings
