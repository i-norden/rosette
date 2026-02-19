"""End-to-end demo: download fixtures, run forensics, print pretty results."""

from __future__ import annotations

import asyncio
import logging
import os
import webbrowser
from pathlib import Path

from snoopy.analysis.cross_reference import compute_ahash, compute_phash
from snoopy.analysis.evidence import aggregate_findings
from snoopy.analysis.run_analysis import run_image_forensics, run_intra_paper_cross_ref
from snoopy.config import AnalysisConfig
from snoopy.reporting.dashboard import generate_dashboard
from snoopy.reporting.pretty import (
    console,
    create_progress,
)

logger = logging.getLogger(__name__)

# Resolve project root: snoopy/demo/runner.py -> project root
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = _PACKAGE_DIR / "tests" / "fixtures"


def _find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if not directory.exists():
        return []
    return sorted(f for f in directory.iterdir() if f.suffix.lower() in extensions)


def _find_pdfs(directory: Path) -> list[Path]:
    """Find all PDF files in a directory."""
    if not directory.exists():
        return []
    return sorted(f for f in directory.iterdir() if f.suffix.lower() == ".pdf")


def _analyze_image(image_path: Path) -> dict:
    """Run all non-AI image analyses on a single image.

    Returns a result dict with findings, perceptual hashes, and metadata.
    Uses the shared run_image_forensics() from run_analysis.py for config-driven
    thresholds, eliminating duplication with the production pipeline.
    """
    path_str = str(image_path)

    # Use shared analysis function with default config
    findings = run_image_forensics(path_str, figure_id=image_path.name)

    # Compute perceptual hashes for cross-reference
    phash_value = None
    ahash_value = None
    try:
        phash_value = compute_phash(path_str)
        ahash_value = compute_ahash(path_str)
    except Exception as exc:
        logger.debug("Hash computation failed on %s: %s", image_path.name, exc)

    return {
        "image": image_path.name,
        "path": str(image_path),
        "findings": findings,
        "num_findings": len(findings),
        "phash": phash_value,
        "ahash": ahash_value,
    }


def _extract_and_analyze_figures(
    pdf_path: Path, figures_dir: Path,
) -> tuple[list[dict], list[dict], int]:
    """Extract figures from PDF and run image forensics on each.

    Returns (figure_findings, figure_results, figure_count).
    """
    figure_findings: list[dict] = []
    figure_results: list[dict] = []
    figure_count = 0

    try:
        from snoopy.extraction.figure_extractor import extract_figures

        out_dir = figures_dir / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = extract_figures(str(pdf_path), str(out_dir))
        figure_count = len(figures)

        _COMPRESSION_SENSITIVE = {"ela", "noise_analysis", "dct_analysis", "jpeg_ghost", "fft_analysis"}
        _PDF_DISCOUNT = 0.5

        for fig in figures:
            img_path = Path(fig.image_path)
            if img_path.exists():
                result = _analyze_image(img_path)
                for finding in result["findings"]:
                    finding["figure_label"] = fig.figure_label or img_path.name
                    method = finding.get("method") or finding.get("analysis_type", "")
                    if method in _COMPRESSION_SENSITIVE:
                        finding["confidence"] *= _PDF_DISCOUNT
                figure_findings.extend(result["findings"])
                figure_results.append(result)

    except Exception as exc:
        logger.warning("Figure extraction failed on %s: %s", pdf_path.name, exc)

    return figure_findings, figure_results, figure_count


def _extract_and_analyze_text(pdf_path: Path) -> tuple[list[dict], dict]:
    """Extract text from PDF and run all statistical/text analyses.

    Returns (text_findings, statistical_summary).
    """
    text_findings: list[dict] = []
    statistical_summary: dict = {}

    try:
        from snoopy.extraction.pdf_parser import extract_text
        from snoopy.extraction.stats_extractor import (
            extract_means_and_ns,
            extract_numerical_values,
            extract_p_values,
            extract_test_statistics,
        )

        pages = extract_text(str(pdf_path))
        full_text = "\n".join(p.text for p in pages)

        grim_findings: list[dict] = []
        benford_findings: list[dict] = []
        pvalue_findings: list[dict] = []
        means_and_ns: list = []
        test_stats: list = []
        numerical_values: list = []

        # 2a. GRIM test on extracted means
        try:
            from snoopy.analysis.statistical import grim_test

            means_and_ns = extract_means_and_ns(full_text)
            grim_failures = 0
            for mn in means_and_ns:
                grim_result = grim_test(mn.mean, mn.n)
                if not grim_result.consistent:
                    grim_failures += 1

            if grim_failures > 0:
                if grim_failures >= 3:
                    severity = "high"
                    confidence = 0.7
                elif grim_failures >= 2:
                    severity = "medium"
                    confidence = 0.6
                else:
                    severity = "low"
                    confidence = 0.45

                finding = {
                    "title": f"GRIM test: {grim_failures} inconsistent mean(s)",
                    "analysis_type": "grim",
                    "method": "grim",
                    "severity": severity,
                    "confidence": confidence,
                    "description": (
                        f"{grim_failures} of {len(means_and_ns)} reported means "
                        f"failed the GRIM consistency test"
                    ),
                    "figure_id": pdf_path.name,
                    "evidence": {
                        "failures": grim_failures,
                        "total_tested": len(means_and_ns),
                    },
                }
                grim_findings.append(finding)
                text_findings.append(finding)
        except Exception as exc:
            logger.debug("GRIM test failed on %s: %s", pdf_path.name, exc)

        # 2b. P-value recalculation
        try:
            from snoopy.analysis.statistical import pvalue_check

            test_stats = extract_test_statistics(full_text)
            for ts in test_stats:
                if ts.p_value is not None and ts.df:
                    try:
                        pv_result = pvalue_check(ts.test_type, ts.statistic, ts.df, ts.p_value)
                        if not pv_result.consistent:
                            if pv_result.difference > 0.05 and pv_result.significance_changed:
                                severity = "high"
                                confidence = 0.85
                            elif pv_result.significance_changed:
                                severity = "medium"
                                confidence = 0.7
                            else:
                                severity = "low"
                                confidence = 0.5

                            finding = {
                                "title": "P-value inconsistency",
                                "analysis_type": "pvalue_check",
                                "method": "pvalue_check",
                                "severity": severity,
                                "confidence": confidence,
                                "description": (
                                    f"{ts.test_type}-test: reported p={pv_result.reported_p:.4f}, "
                                    f"computed p={pv_result.computed_p:.4f}, "
                                    f"diff={pv_result.difference:.4f}"
                                    f"{', significance changed!' if pv_result.significance_changed else ''}"
                                ),
                                "figure_id": pdf_path.name,
                                "evidence": {
                                    "test_type": ts.test_type,
                                    "reported_p": round(pv_result.reported_p, 6),
                                    "computed_p": round(pv_result.computed_p, 6),
                                    "difference": round(pv_result.difference, 6),
                                    "significance_changed": pv_result.significance_changed,
                                },
                            }
                            pvalue_findings.append(finding)
                            text_findings.append(finding)
                    except Exception as exc:
                        logger.debug(
                            "P-value check failed for %s stat on %s: %s",
                            ts.test_type,
                            pdf_path.name,
                            exc,
                        )
        except Exception as exc:
            logger.debug("P-value extraction failed on %s: %s", pdf_path.name, exc)

        # 2c. Benford's Law test
        try:
            from snoopy.analysis.statistical import benford_test

            numerical_values = extract_numerical_values(full_text)
            if len(numerical_values) >= 50:
                benford_result = benford_test(numerical_values)
                if not benford_result.conforms:
                    n_vals = benford_result.n_values
                    p_val = benford_result.p_value

                    if p_val < 0.0001 and n_vals >= 100:
                        severity = "medium"
                        confidence = 0.45
                    elif p_val < 0.001 and n_vals >= 50:
                        severity = "low"
                        confidence = 0.35
                    else:
                        severity = "low"
                        confidence = 0.25

                    finding = {
                        "title": "Benford's Law non-conformity",
                        "analysis_type": "benford",
                        "method": "benford",
                        "severity": severity,
                        "confidence": confidence,
                        "description": (
                            f"Leading digit distribution deviates from Benford's Law "
                            f"(chi2={benford_result.chi_squared:.2f}, "
                            f"p={benford_result.p_value:.6f}, n={n_vals})"
                        ),
                        "figure_id": pdf_path.name,
                        "evidence": {
                            "chi_squared": round(benford_result.chi_squared, 4),
                            "p_value": round(benford_result.p_value, 6),
                            "n_values": n_vals,
                        },
                    }
                    benford_findings.append(finding)
                    text_findings.append(finding)
        except Exception as exc:
            logger.debug("Benford test failed on %s: %s", pdf_path.name, exc)

        # 2d. Advanced statistical tests (GRIMMER, variance ratio, terminal digit)
        grimmer_findings: list[dict] = []
        try:
            from snoopy.analysis.run_analysis import run_statistical_tests

            stat_findings = run_statistical_tests(full_text)
            grimmer_findings = stat_findings
            text_findings.extend(stat_findings)
        except Exception as exc:
            logger.debug("Statistical tests failed on %s: %s", pdf_path.name, exc)

        # 2e. SPRITE consistency test
        sprite_findings: list[dict] = []
        try:
            from snoopy.analysis.run_analysis import run_sprite_analysis
            from snoopy.extraction.stats_extractor import extract_means_sds_and_ns

            mean_sd_reports = extract_means_sds_and_ns(full_text)
            for mr in mean_sd_reports:
                if mr.sd is not None and mr.n >= 2:
                    sf = run_sprite_analysis(mr.mean, mr.sd, mr.n, context=mr.context)
                    sprite_findings.extend(sf)
                    text_findings.extend(sf)
        except Exception as exc:
            logger.debug("SPRITE test failed on %s: %s", pdf_path.name, exc)

        # 2f. Tortured phrase detection
        tp_findings: list[dict] = []
        try:
            from snoopy.analysis.run_analysis import run_tortured_phrases

            tp_findings = run_tortured_phrases(full_text)
            text_findings.extend(tp_findings)
        except Exception as exc:
            logger.debug("Tortured phrases failed on %s: %s", pdf_path.name, exc)

        # Collect p-value overview (reuse already-extracted values)
        try:
            p_values = extract_p_values(full_text)
            statistical_summary = {
                "means_extracted": len(means_and_ns),
                "test_stats_extracted": len(test_stats),
                "p_values_extracted": len(p_values),
                "numerical_values_extracted": len(numerical_values),
                "grim_findings": len(grim_findings),
                "pvalue_findings": len(pvalue_findings),
                "benford_findings": len(benford_findings),
                "grimmer_findings": len(grimmer_findings),
                "sprite_findings": len(sprite_findings),
                "tortured_phrase_findings": len(tp_findings),
            }
        except Exception as exc:
            logger.debug("Stats summary failed on %s: %s", pdf_path.name, exc)

    except Exception as exc:
        logger.warning("Text extraction failed on %s: %s", pdf_path.name, exc)

    return text_findings, statistical_summary


def _extract_and_analyze_tables(pdf_path: Path) -> list[dict]:
    """Extract tables from PDF and check for suspicious value patterns.

    Returns list of finding dicts.
    """
    table_findings: list[dict] = []
    try:
        from snoopy.analysis.statistical import duplicate_value_check
        from snoopy.extraction.table_extractor import extract_tables

        tables = extract_tables(str(pdf_path))
        for table in tables:
            if table.rows:
                dup_result = duplicate_value_check(table.rows)
                if dup_result.suspicious:
                    both_flags = (
                        dup_result.duplicate_ratio > 0.3 and dup_result.round_number_ratio > 0.8
                    )
                    if both_flags:
                        severity = "medium"
                        confidence = 0.5
                    else:
                        severity = "low"
                        confidence = 0.3

                    table_findings.append(
                        {
                            "title": "Suspicious value patterns in table",
                            "analysis_type": "duplicate_check",
                            "method": "duplicate_check",
                            "severity": severity,
                            "confidence": confidence,
                            "description": dup_result.details,
                            "figure_id": pdf_path.name,
                            "evidence": {
                                "page": table.page_number,
                                "table_index": table.table_index,
                                "duplicate_ratio": round(dup_result.duplicate_ratio, 4),
                                "round_number_ratio": round(dup_result.round_number_ratio, 4),
                                "total_values": dup_result.total_values,
                            },
                        }
                    )
    except Exception as exc:
        logger.debug("Table extraction failed on %s: %s", pdf_path.name, exc)
    return table_findings


def _analyze_pdf(pdf_path: Path, figures_dir: Path) -> dict:
    """Extract and analyze all content from a PDF.

    Runs figure extraction + image forensics, text extraction + statistical tests,
    table extraction + duplicate value check, and intra-paper hash cross-reference.
    Figure analysis and text/table analysis run concurrently.
    """
    phash_matches: list[dict] = []

    # Run figure analysis, text analysis, and table analysis
    figure_findings, figure_results, figure_count = _extract_and_analyze_figures(pdf_path, figures_dir)
    text_findings, statistical_summary = _extract_and_analyze_text(pdf_path)
    table_findings = _extract_and_analyze_tables(pdf_path)

    # Merge all findings
    all_findings = figure_findings + text_findings + table_findings

    # Tag all text-based findings with a common paper-level figure_id so
    # that the evidence aggregation can detect convergence among them.
    _paper_text_id = f"paper_text:{pdf_path.name}"
    _bare_pdf_name = pdf_path.name
    for f in all_findings:
        fig_id = f.get("figure_id") or ""
        if not fig_id or fig_id == _bare_pdf_name:
            f["figure_id"] = _paper_text_id

    # 4. Intra-paper cross-reference using shared function
    try:
        cross_ref_findings = run_intra_paper_cross_ref(figure_results)
        all_findings.extend(cross_ref_findings)
        for f in cross_ref_findings:
            evidence = f.get("evidence", {})
            pairs = evidence.get("pairs", [])
            if pairs:
                for pair in pairs:
                    phash_matches.append(
                        {
                            "figure_a": pair.get("figure_a", ""),
                            "figure_b": pair.get("figure_b", ""),
                            "distance": pair.get("hash_distance", 0),
                            "severity": f.get("severity", ""),
                        }
                    )
            else:
                phash_matches.append(
                    {
                        "figure_a": evidence.get("figure_a", ""),
                        "figure_b": evidence.get("figure_b", ""),
                        "distance": evidence.get("hash_distance", 0),
                        "severity": f.get("severity", ""),
                    }
                )
    except Exception as exc:
        logger.debug("Hash cross-reference failed on %s: %s", pdf_path.name, exc)

    return {
        "pdf": pdf_path.name,
        "path": str(pdf_path),
        "figures_extracted": figure_count,
        "findings": all_findings,
        "num_findings": len(all_findings),
        "statistical_summary": statistical_summary,
        "phash_matches": phash_matches,
    }


def _collect_methods(findings: list[dict]) -> list[str]:
    """Collect unique method names from findings, preserving order."""
    methods: list[str] = []
    seen: set[str] = set()
    for f in findings:
        m = f.get("method") or f.get("analysis_type", "")
        if m and m not in seen:
            methods.append(m)
            seen.add(m)
    return methods


def _build_result(
    name: str,
    category: str,
    expected: str,
    findings: list[dict],
    pass_fail: bool,
    extra: dict | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> dict:
    """Build a standardized result dict using aggregate_findings()."""
    cfg = analysis_config or AnalysisConfig()
    aggregated = aggregate_findings(
        findings,
        method_weights=cfg.method_weights,
        single_method_max_severity=cfg.single_method_max_severity,
        single_method_max_confidence=cfg.single_method_max_confidence,
        convergence_confidence_threshold=cfg.convergence_confidence_threshold,
    )

    result = {
        "name": name,
        "category": category,
        "expected": expected,
        "actual_risk": aggregated.paper_risk,
        "overall_confidence": aggregated.overall_confidence,
        "converging_evidence": aggregated.converging_evidence,
        "findings_count": aggregated.total_findings,
        "pass_fail": pass_fail,
        "findings": findings,
        "methods_used": _collect_methods(findings),
    }
    if extra:
        result.update(extra)
    return result


def _get_paper_risk(findings: list[dict], config: AnalysisConfig | None = None) -> str:
    """Compute paper risk using config-aware aggregation."""
    cfg = config or AnalysisConfig()
    return aggregate_findings(
        findings,
        method_weights=cfg.method_weights,
        single_method_max_severity=cfg.single_method_max_severity,
        single_method_max_confidence=cfg.single_method_max_confidence,
        convergence_confidence_threshold=cfg.convergence_confidence_threshold,
    ).paper_risk


def _determine_pass_fail_expected_findings(aggregated_risk: str, findings: list[dict]) -> bool:
    """For items expected to have findings, pass if risk is at least medium.

    A few low-confidence noise findings shouldn't count as "detected" —
    the risk level must be medium or higher to be a meaningful detection.
    """
    if not findings:
        return False
    return aggregated_risk in ("medium", "high", "critical")


def _determine_pass_fail_expected_clean(
    findings: list[dict],
    analysis_config: AnalysisConfig | None = None,
) -> bool:
    """For clean items, fail only if aggregated paper risk is high/critical."""
    if not findings:
        return True
    cfg = analysis_config or AnalysisConfig()
    aggregated = aggregate_findings(
        findings,
        method_weights=cfg.method_weights,
        single_method_max_severity=cfg.single_method_max_severity,
        single_method_max_confidence=cfg.single_method_max_confidence,
        convergence_confidence_threshold=cfg.convergence_confidence_threshold,
    )
    return aggregated.paper_risk not in ("critical", "high")


async def _run_llm_analysis(
    demo_results: list[dict],
    figures_dir: Path,
) -> None:
    """Run LLM-based vision analysis on figures with findings or a sampled subset.

    Modifies demo_results in place by adding LLM findings and re-aggregating.
    """
    try:
        from snoopy.analysis.llm_vision import analyze_figure_detailed, screen_figure
        from snoopy.llm.claude import ClaudeProvider

        provider = ClaudeProvider()
    except Exception as exc:
        logger.warning("Could not initialize LLM provider: %s", exc)
        return

    for result in demo_results:
        findings = result.get("findings", [])
        # Find image paths from findings
        image_paths: dict[str, str] = {}
        for f in findings:
            fig_id = f.get("figure_id", "")
            if fig_id and fig_id not in image_paths:
                # Try to find the actual image path
                # Look for image files in the figures_dir
                for candidate in figures_dir.rglob(fig_id):
                    if candidate.is_file():
                        image_paths[fig_id] = str(candidate)
                        break

        # Screen figures with findings
        for fig_id, img_path in image_paths.items():
            try:
                screening = await screen_figure(img_path, provider)
                if screening.suspicious and screening.confidence > 0.5:
                    findings.append(
                        {
                            "title": f"LLM screening flagged: {screening.reason}",
                            "analysis_type": "llm_screening",
                            "method": "llm_screening",
                            "severity": "medium" if screening.confidence > 0.7 else "low",
                            "confidence": screening.confidence,
                            "description": screening.reason,
                            "figure_id": fig_id,
                            "evidence": {
                                "model": screening.model_used,
                                "confidence": screening.confidence,
                            },
                        }
                    )

                    # Detailed analysis for flagged figures
                    try:
                        detailed = await analyze_figure_detailed(img_path, provider)
                        for vf in detailed.findings:
                            findings.append(
                                {
                                    "title": f"LLM vision: {vf.finding_type}",
                                    "analysis_type": "llm_vision",
                                    "method": "llm_vision",
                                    "severity": (
                                        "high"
                                        if vf.confidence > 0.8
                                        else "medium"
                                        if vf.confidence > 0.5
                                        else "low"
                                    ),
                                    "confidence": vf.confidence,
                                    "description": vf.description,
                                    "figure_id": fig_id,
                                    "evidence": {
                                        "location": vf.location,
                                        "model": detailed.model_used,
                                        "manipulation_likelihood": detailed.manipulation_likelihood,
                                    },
                                }
                            )
                    except Exception as exc:
                        logger.debug("LLM detailed analysis failed for %s: %s", fig_id, exc)

            except Exception as exc:
                logger.debug("LLM screening failed for %s: %s", fig_id, exc)

        # Re-aggregate findings after LLM additions
        if image_paths:
            cfg = AnalysisConfig()
            aggregated = aggregate_findings(
                findings,
                method_weights=cfg.method_weights,
                single_method_max_severity=cfg.single_method_max_severity,
                single_method_max_confidence=cfg.single_method_max_confidence,
                convergence_confidence_threshold=cfg.convergence_confidence_threshold,
            )
            result["actual_risk"] = aggregated.paper_risk
            result["overall_confidence"] = aggregated.overall_confidence
            result["converging_evidence"] = aggregated.converging_evidence
            result["findings_count"] = aggregated.total_findings
            result["methods_used"] = _collect_methods(findings)


def run_demo(
    download_only: bool = False,
    skip_llm: bool = True,
    output_dir: str | None = None,
    download_rsiil: bool = False,
) -> list[dict]:
    """Run the full demo pipeline.

    Args:
        download_only: If True, only download fixtures and exit.
        skip_llm: If True, skip LLM-based analysis.
        output_dir: Directory for HTML reports. Defaults to data/reports/demo/.
        download_rsiil: If True, download the full RSIIL dataset from Zenodo.

    Returns:
        List of result dicts for the summary dashboard.
    """
    from snoopy.demo.fixtures import download_all

    # Load analysis config for evidence aggregation (weights, thresholds)
    analysis_cfg = AnalysisConfig()

    report_dir = Path(output_dir) if output_dir else _PACKAGE_DIR / "data" / "reports" / "demo"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = _PACKAGE_DIR / "data" / "figures" / "demo"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure fixtures exist
    console.rule("[bold blue]Step 1: Check & Download Fixtures[/bold blue]")
    console.print()

    from snoopy.demo.fixtures import (
        CLEAN_PAPERS,
        RETRACTED_PAPERS,
        RETRACTION_WATCH_PAPERS,
        RSIIL_CLEAN_IMAGES,
        RSIIL_FORGERY_IMAGES,
    )

    # Build expected file sets per category so we check for specific files,
    # not just a file count (which can be inflated by unrelated local files).
    expected_files: dict[str, set[str]] = {
        "rsiil": {name for _, name in RSIIL_FORGERY_IMAGES}
        | {name for _, name in RSIIL_CLEAN_IMAGES},
        "retracted": {p["filename"] for p in RETRACTED_PAPERS},
        "retraction_watch": {p["filename"] for p in RETRACTION_WATCH_PAPERS},
        "clean": {p["filename"] for p in CLEAN_PAPERS},
    }
    # Synthetic uses a count check (files are generated, not downloaded by name)
    expected_counts: dict[str, int] = {"synthetic": 15, "survey": 1}

    incomplete = []
    for d, names in expected_files.items():
        d_path = FIXTURES_DIR / d
        existing = {f.name for f in d_path.iterdir() if f.is_file()} if d_path.exists() else set()
        missing = names - existing
        if missing:
            incomplete.append(f"{d} (missing {len(missing)}/{len(names)})")
    for d, expected in expected_counts.items():
        d_path = FIXTURES_DIR / d
        actual = sum(1 for f in d_path.iterdir() if f.is_file()) if d_path.exists() else 0
        if actual < expected:
            incomplete.append(f"{d} ({actual}/{expected})")
    if incomplete:
        console.print(
            f"[yellow]Incomplete fixture categories: {', '.join(incomplete)}. Downloading...[/yellow]"
        )
        download_all()
    else:
        console.print("[green]All fixture categories present.[/green]")

    # Download full RSIIL dataset from Zenodo if not already present.
    # Triggered automatically when data/rsiil/ is missing or empty, or
    # explicitly via --download-rsiil flag.
    from snoopy.demo.fixtures import RSIIL_DATA_DIR, download_rsiil_zenodo

    rsiil_needs_download = download_rsiil
    if not rsiil_needs_download:
        # Auto-detect: check if pristine/ and test/ dirs have any files
        pristine_dir = RSIIL_DATA_DIR / "pristine"
        test_dir = RSIIL_DATA_DIR / "test"
        has_pristine = pristine_dir.exists() and any(pristine_dir.rglob("*"))
        has_test = test_dir.exists() and any(test_dir.rglob("*"))
        rsiil_needs_download = not (has_pristine and has_test)

    if rsiil_needs_download:
        import httpx

        console.print()
        console.rule("[bold blue]Downloading Full RSIIL Dataset from Zenodo[/bold blue]")
        console.print()
        with httpx.Client(
            headers={"User-Agent": "snoopy/0.1 (academic research tool)"},
            follow_redirects=True,
            timeout=600.0,
        ) as client:
            zenodo_counts = download_rsiil_zenodo(client)
        total_files = sum(zenodo_counts.values())
        console.print(
            f"[bold green]RSIIL Zenodo dataset: {total_files} files across {len(zenodo_counts)} splits.[/bold green]"
        )
        console.print()

    if download_only:
        console.print("[dim]--download-only specified, exiting.[/dim]")
        return []

    # Step 2: Run forensics on all test items
    console.print()
    console.rule("[bold blue]Step 2: Run Analysis[/bold blue]")
    console.print()

    demo_results: list[dict] = []

    # 2a) Synthetic forgeries
    console.print("[bold]Analyzing synthetic forgeries...[/bold]")
    synthetic_images = _find_images(FIXTURES_DIR / "synthetic")
    if synthetic_images:
        with create_progress() as progress:
            task = progress.add_task("Synthetic images", total=len(synthetic_images))
            for img_path in synthetic_images:
                result = _analyze_image(img_path)
                demo_results.append(
                    _build_result(
                        name=img_path.name,
                        category="synthetic",
                        expected="findings",
                        findings=result["findings"],
                        pass_fail=_determine_pass_fail_expected_findings(
                            _get_paper_risk(result["findings"], analysis_cfg),
                            result["findings"],
                        ),
                        analysis_config=analysis_cfg,
                    )
                )
                progress.advance(task)
    console.print()

    # 2b) RSIIL benchmark images (forgery — expected to have findings)
    from snoopy.demo.fixtures import _is_ground_truth, _is_pristine_ref

    rsiil_clean_names = {name for _, name in RSIIL_CLEAN_IMAGES}

    console.print("[bold]Analyzing RSIIL benchmark images...[/bold]")
    rsiil_images = _find_images(FIXTURES_DIR / "rsiil")
    if rsiil_images:
        # Filter out ground-truth images before submitting to thread pool
        analyzable = [img for img in rsiil_images if not _is_ground_truth(img)]
        skipped = len(rsiil_images) - len(analyzable)

        with create_progress() as progress:
            task = progress.add_task("RSIIL images", total=len(rsiil_images))
            # Advance for skipped ground-truth images
            for _ in range(skipped):
                progress.advance(task)

            for img_path in analyzable:
                result = _analyze_image(img_path)
                if img_path.name in rsiil_clean_names or _is_pristine_ref(img_path):
                    demo_results.append(
                        _build_result(
                            name=img_path.name,
                            category="rsiil_clean",
                            expected="clean",
                            findings=result["findings"],
                            pass_fail=_determine_pass_fail_expected_clean(
                                result["findings"], analysis_config=analysis_cfg,
                            ),
                            analysis_config=analysis_cfg,
                        )
                    )
                else:
                    demo_results.append(
                        _build_result(
                            name=img_path.name,
                            category="rsiil",
                            expected="findings",
                            findings=result["findings"],
                            pass_fail=_determine_pass_fail_expected_findings(
                                _get_paper_risk(result["findings"], analysis_cfg),
                                result["findings"],
                            ),
                            analysis_config=analysis_cfg,
                        )
                    )
                progress.advance(task)
    else:
        console.print("[dim]No RSIIL images found (download may have failed). Skipping.[/dim]")
    console.print()

    # 2b-zenodo) Sampled RSIIL images from full Zenodo dataset (if available)
    from snoopy.demo.fixtures import sample_rsiil_images

    pristine_sample, tampered_sample = sample_rsiil_images(50)
    if tampered_sample or pristine_sample:
        console.print("[bold]Analyzing sampled RSIIL Zenodo images...[/bold]")
        total_zenodo = len(tampered_sample) + len(pristine_sample)

        # Build a mapping of path -> expected category for parallel processing
        zenodo_categories = {}
        for img_path in tampered_sample:
            zenodo_categories[img_path] = "tampered"
        for img_path in pristine_sample:
            zenodo_categories[img_path] = "pristine"

        all_zenodo = list(tampered_sample) + list(pristine_sample)
        with create_progress() as progress:
            task = progress.add_task("RSIIL Zenodo samples", total=total_zenodo)
            for img_path in all_zenodo:
                result = _analyze_image(img_path)
                if zenodo_categories[img_path] == "tampered":
                    demo_results.append(
                        _build_result(
                            name=img_path.name,
                            category="rsiil",
                            expected="findings",
                            findings=result["findings"],
                            pass_fail=_determine_pass_fail_expected_findings(
                                _get_paper_risk(result["findings"], analysis_cfg),
                                result["findings"],
                            ),
                            analysis_config=analysis_cfg,
                        )
                    )
                else:
                    demo_results.append(
                        _build_result(
                            name=img_path.name,
                            category="rsiil_clean",
                            expected="clean",
                            findings=result["findings"],
                            pass_fail=_determine_pass_fail_expected_clean(
                                result["findings"], analysis_config=analysis_cfg,
                            ),
                            analysis_config=analysis_cfg,
                        )
                    )
                progress.advance(task)
        console.print()

    # 2c) Retracted papers
    console.print("[bold]Analyzing retracted papers...[/bold]")
    retracted_pdfs = _find_pdfs(FIXTURES_DIR / "retracted")
    if retracted_pdfs:
        with create_progress() as progress:
            task = progress.add_task("Retracted papers", total=len(retracted_pdfs))
            for pdf_path in retracted_pdfs:
                result = _analyze_pdf(pdf_path, figures_dir)
                demo_results.append(
                    _build_result(
                        name=pdf_path.name,
                        category="retracted",
                        expected="findings",
                        findings=result["findings"],
                        pass_fail=_determine_pass_fail_expected_findings(
                            _get_paper_risk(result["findings"], analysis_cfg),
                            result["findings"],
                        ),
                        extra={
                            "statistical_summary": result.get("statistical_summary", {}),
                            "phash_matches": result.get("phash_matches", []),
                        },
                        analysis_config=analysis_cfg,
                    )
                )
                progress.advance(task)
    else:
        console.print("[dim]No retracted papers found. Skipping.[/dim]")
    console.print()

    # 2d) Survey paper
    console.print("[bold]Analyzing Bik survey paper...[/bold]")
    for pdf_path in _find_pdfs(FIXTURES_DIR / "survey"):
        result = _analyze_pdf(pdf_path, figures_dir)
        demo_results.append(
            _build_result(
                name=pdf_path.name,
                category="survey",
                expected="informational",
                findings=result["findings"],
                pass_fail=True,
                extra={
                    "statistical_summary": result.get("statistical_summary", {}),
                    "phash_matches": result.get("phash_matches", []),
                },
                analysis_config=analysis_cfg,
            )
        )
    console.print()

    # 2e) Retraction Watch papers
    console.print("[bold]Analyzing Retraction Watch papers...[/bold]")
    for pdf_path in _find_pdfs(FIXTURES_DIR / "retraction_watch"):
        result = _analyze_pdf(pdf_path, figures_dir)
        demo_results.append(
            _build_result(
                name=pdf_path.name,
                category="retraction_watch",
                expected="findings",
                findings=result["findings"],
                pass_fail=_determine_pass_fail_expected_findings(
                    _get_paper_risk(result["findings"], analysis_cfg),
                    result["findings"],
                ),
                extra={
                    "statistical_summary": result.get("statistical_summary", {}),
                    "phash_matches": result.get("phash_matches", []),
                },
                analysis_config=analysis_cfg,
            )
        )
    console.print()

    # 2f) Clean control papers (false positive check)
    console.print("[bold]Analyzing clean control papers (false positive check)...[/bold]")
    clean_pdfs = _find_pdfs(FIXTURES_DIR / "clean")
    if clean_pdfs:
        with create_progress() as progress:
            task = progress.add_task("Clean papers", total=len(clean_pdfs))
            for pdf_path in clean_pdfs:
                result = _analyze_pdf(pdf_path, figures_dir)
                demo_results.append(
                    _build_result(
                        name=pdf_path.name,
                        category="clean",
                        expected="clean",
                        findings=result["findings"],
                        pass_fail=_determine_pass_fail_expected_clean(
                            result["findings"], analysis_config=analysis_cfg,
                        ),
                        extra={
                            "statistical_summary": result.get("statistical_summary", {}),
                            "phash_matches": result.get("phash_matches", []),
                        },
                        analysis_config=analysis_cfg,
                    )
                )
                progress.advance(task)
    else:
        console.print("[dim]No clean papers found. Skipping.[/dim]")
    console.print()

    # Step 2.5: LLM analysis (gated on skip_llm flag)
    if not skip_llm and os.environ.get("ANTHROPIC_API_KEY"):
        console.print()
        console.rule("[bold blue]Step 2.5: LLM Vision Analysis[/bold blue]")
        console.print()
        try:
            asyncio.run(_run_llm_analysis(demo_results, figures_dir))
            console.print("[green]LLM analysis complete.[/green]")
        except Exception as exc:
            console.print(f"[yellow]LLM analysis failed: {exc}[/yellow]")
        console.print()

    # Step 3: Generate HTML reports + dashboard
    console.print()
    console.rule("[bold blue]Step 3: Generate Reports[/bold blue]")
    console.print("[dim]Generating reports...[/dim]")

    try:
        from snoopy.reporting.proof import generate_html_report

        for result in demo_results:
            if result["findings"]:
                paper = {
                    "title": result["name"],
                    "doi": "N/A",
                    "journal": result["category"],
                    "citation_count": 0,
                }
                html = generate_html_report(
                    paper=paper,
                    findings=result["findings"],
                    figures={},
                    summary=f"Demo analysis of {result['name']}",
                    overall_risk=result["actual_risk"],
                    overall_confidence=result.get("overall_confidence", 0.5),
                    converging_evidence=result.get("converging_evidence", False),
                )
                html_path = report_dir / f"{Path(result['name']).stem}_report.html"
                html_path.write_text(html)
    except Exception as exc:
        console.print(f"[yellow]HTML report generation skipped: {exc}[/yellow]")

    # Generate the dashboard
    dashboard_path = generate_dashboard(demo_results, report_dir)

    if skip_llm:
        console.print(
            "[dim]LLM analysis skipped (--skip-llm or no ANTHROPIC_API_KEY). "
            "Use --use-llm and set ANTHROPIC_API_KEY to enable.[/dim]"
        )
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[yellow]ANTHROPIC_API_KEY not set. LLM analysis skipped.[/yellow]")
    console.print()

    # Step 4: Summary + open dashboard
    passed = sum(1 for r in demo_results if r.get("pass_fail"))
    total = len(demo_results)
    style = "bold green" if passed == total else "bold yellow"
    console.print(f"[{style}]{passed}/{total} passed[/{style}]")
    console.print(f"[dim]Dashboard: {dashboard_path}[/dim]")
    webbrowser.open(dashboard_path.as_uri())

    return demo_results
