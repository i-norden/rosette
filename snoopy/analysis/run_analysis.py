"""Shared analysis functions used by both production orchestrator and demo runner.

Provides config-driven wrappers around the underlying analysis methods that
return standardized finding dicts. This eliminates the duplication between
``snoopy.pipeline.orchestrator`` and ``snoopy.demo.runner``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from snoopy.analysis.cross_reference import hash_distance
from snoopy.analysis.image_forensics import (
    CloneResult,
    ELAResult,
    NoiseResult,
    clone_detection,
    error_level_analysis,
    noise_analysis,
)
from snoopy.config import AnalysisConfig

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

    # 1. ELA
    try:
        ela: ELAResult = error_level_analysis(path_str, quality=cfg.ela_quality, output_dir=output_dir)
        if ela.suspicious:
            max_diff = ela.max_difference
            mean_diff = ela.mean_difference
            std_diff = ela.std_difference

            if max_diff >= cfg.ela_high_threshold and max_diff > mean_diff + 3 * std_diff:
                severity = "high"
                confidence = min(max_diff / 255.0 * 0.6, 1.0)
            elif max_diff >= cfg.ela_medium_threshold and max_diff > mean_diff + 3 * std_diff:
                severity = "medium"
                confidence = min(max_diff / 255.0 * 0.5, 1.0)
            elif max_diff >= cfg.ela_low_threshold and max_diff > mean_diff + 2 * std_diff:
                severity = "low"
                confidence = min(max_diff / 255.0 * 0.4, 1.0)
            else:
                severity = "low"
                confidence = min(max_diff / 255.0 * 0.3, 1.0)

            findings.append({
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
            })
    except Exception as exc:
        logger.debug("ELA failed on %s: %s", fig_id, exc)

    # 2. Clone detection
    try:
        clone: CloneResult = clone_detection(path_str, min_matches=cfg.clone_min_matches)
        if clone.suspicious:
            inliers = clone.num_matches
            ratio = clone.inlier_ratio

            if inliers >= cfg.clone_high_inliers and ratio >= cfg.clone_high_ratio:
                severity = "high"
                confidence = min(ratio, 1.0) * 0.85
            elif inliers >= cfg.clone_medium_inliers and ratio >= cfg.clone_medium_ratio:
                severity = "medium"
                confidence = min(ratio, 1.0) * 0.7
            elif inliers >= cfg.clone_low_inliers and ratio >= cfg.clone_low_ratio:
                severity = "low"
                confidence = min(ratio, 1.0) * 0.5
            else:
                severity = "low"
                confidence = min(ratio, 1.0) * 0.4

            findings.append({
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
            })
    except Exception as exc:
        logger.debug("Clone detection failed on %s: %s", fig_id, exc)

    # 3. Noise analysis
    try:
        noise_result: NoiseResult = noise_analysis(
            path_str,
            block_size=cfg.noise_block_size,
            intensity_bin_width=cfg.noise_intensity_bin_width,
        )
        if noise_result.suspicious:
            max_ratio = noise_result.max_ratio

            if max_ratio > cfg.noise_high_ratio:
                severity = "high"
                confidence = min(max_ratio / 30.0, 0.85)
            elif max_ratio > cfg.noise_medium_ratio:
                severity = "medium"
                confidence = min(max_ratio / 20.0, 0.7)
            elif max_ratio > cfg.noise_low_ratio:
                severity = "low"
                confidence = min(max_ratio / 20.0, 0.5)
            else:
                severity = "low"
                confidence = min(max_ratio / 20.0, 0.4)

            findings.append({
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
            })
    except Exception as exc:
        logger.debug("Noise analysis failed on %s: %s", fig_id, exc)

    # 4. Metadata forensics
    try:
        from snoopy.analysis.metadata_forensics import analyze_metadata

        meta = analyze_metadata(path_str)
        if meta.suspicious:
            for mf in meta.findings:
                findings.append({
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
                })
    except Exception as exc:
        logger.debug("Metadata forensics failed on %s: %s", fig_id, exc)

    return findings


def run_intra_paper_cross_ref(
    figure_results: list[dict],
    config: AnalysisConfig | None = None,
) -> list[dict]:
    """Compare figure hashes within the same paper to find duplicates.

    Args:
        figure_results: List of dicts, each with 'image' (name), 'phash', and 'ahash' keys.

    Returns:
        List of finding dicts for any intra-paper hash matches.
    """
    findings: list[dict] = []

    for i, res_a in enumerate(figure_results):
        for j, res_b in enumerate(figure_results):
            if j <= i:
                continue
            phash_a = res_a.get("phash")
            phash_b = res_b.get("phash")
            if phash_a and phash_b:
                try:
                    dist = hash_distance(phash_a, phash_b)
                except ValueError:
                    continue

                if dist <= 15:
                    if dist <= 5:
                        severity = "critical"
                        confidence = 0.95
                    elif dist <= 10:
                        severity = "high"
                        confidence = 0.8
                    else:
                        severity = "medium"
                        confidence = 0.6

                    findings.append({
                        "title": f"Perceptual hash match: {res_a['image']} \u2194 {res_b['image']}",
                        "analysis_type": "phash",
                        "method": "phash",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"Figures '{res_a['image']}' and '{res_b['image']}' "
                            f"have perceptual hash distance {dist} (possible duplicate/recycled)"
                        ),
                        "figure_id": res_a["image"],
                        "evidence": {
                            "figure_a": res_a["image"],
                            "figure_b": res_b["image"],
                            "hash_distance": dist,
                        },
                    })

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
        from snoopy.analysis.western_blot import analyze_western_blot

        result = analyze_western_blot(image_path)
        if result.suspicious:
            # Duplicate lanes
            for lane_i, lane_j, corr in result.duplicate_lanes:
                findings.append({
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
                })

            # Splice boundaries
            for splice in result.splice_boundaries:
                if splice.confidence > 0.5:
                    findings.append({
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
                            "background_discontinuity": round(splice.background_discontinuity, 2),
                            "noise_discontinuity": round(splice.noise_discontinuity, 3),
                        },
                    })

            # Uniform profiles
            if result.uniform_profiles and not result.duplicate_lanes:
                findings.append({
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
                })
    except Exception as exc:
        logger.debug("Western blot analysis failed on %s: %s", fig_id, exc)

    return findings
