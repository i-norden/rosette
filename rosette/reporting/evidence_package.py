"""Automated evidence packages for editors and institutional review boards.

Generates structured, legally-defensible evidence packages as ZIP files
containing executive summaries, annotated figures, analysis artifacts,
and machine-readable manifests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rosette.analysis.evidence import AggregatedEvidence

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


def _file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_executive_summary(
    paper: dict,
    evidence: AggregatedEvidence,
    findings: list[dict],
) -> str:
    """Generate a Markdown executive summary."""
    lines = [
        f"# Evidence Package: {paper.get('title', 'Unknown Paper')}",
        "",
        f"**DOI:** {paper.get('doi', 'N/A')}",
        f"**Journal:** {paper.get('journal', 'N/A')}",
        f"**Analysis Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Rosette Version:** {_VERSION}",
        "",
        "## Overall Assessment",
        "",
        f"- **Risk Level:** {evidence.paper_risk.upper()}",
        f"- **Overall Confidence:** {evidence.overall_confidence:.2f}",
        f"- **Converging Evidence:** {'Yes' if evidence.converging_evidence else 'No'}",
        f"- **Total Findings:** {evidence.total_findings}",
        f"- **Critical Findings:** {evidence.critical_count}",
        "",
        "## Findings Summary",
        "",
    ]

    if not findings:
        lines.append("No integrity concerns were identified.")
    else:
        # Group by severity
        by_severity: dict[str, list[dict]] = {}
        for f in findings:
            sev = f.get("severity", "info")
            by_severity.setdefault(sev, []).append(f)

        for severity in ("critical", "high", "medium", "low"):
            if severity in by_severity:
                lines.append(f"### {severity.title()} Severity ({len(by_severity[severity])})")
                lines.append("")
                for f in by_severity[severity]:
                    lines.append(
                        f"- **{f.get('title', 'Finding')}** (confidence: {f.get('confidence', 0):.2f})"
                    )
                    desc = f.get("description", "")
                    if desc:
                        lines.append(f"  {desc}")
                    lines.append("")

    lines.extend(
        [
            "",
            "## Methods Used",
            "",
        ]
    )

    methods = set()
    for f in findings:
        m = f.get("method") or f.get("analysis_type", "")
        if m:
            methods.add(m)

    method_descriptions = {
        "ela": "Error Level Analysis (JPEG re-compression difference)",
        "clone_detection": "Copy-Move Detection (ORB + RANSAC)",
        "noise_analysis": "Noise Inconsistency Analysis (Laplacian variance)",
        "phash": "Perceptual Hash Cross-Reference",
        "grim": "GRIM Test (Granularity-Related Inconsistency of Means)",
        "pvalue_check": "P-Value Recalculation Check",
        "benford": "Benford's Law Conformity Test",
        "llm_vision": "LLM Vision Analysis (Claude)",
        "western_blot": "Western Blot Specific Analysis",
        "metadata_forensics": "EXIF/ICC Metadata Forensics",
        "sprite": "SPRITE (Sample Parameter Reconstruction)",
    }

    for m in sorted(methods):
        desc = method_descriptions.get(m, m)
        lines.append(f"- {desc}")

    lines.extend(
        [
            "",
            "---",
            "",
            "*This evidence package was generated automatically by Rosette, an open-source "
            "academic integrity analysis tool. All findings should be independently verified "
            "before taking any action.*",
        ]
    )

    return "\n".join(lines)


def generate_evidence_package(
    paper: dict,
    evidence: AggregatedEvidence,
    findings: list[dict],
    figures_dir: str | None = None,
    pdf_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Generate a ZIP evidence package.

    The package contains:
    - executive_summary.md — Markdown summary
    - manifest.json — Machine-readable chain-of-evidence metadata
    - findings.json — Raw findings data
    - figures/ — Original figures referenced in findings
    - artifacts/ — ELA difference images, etc.

    Args:
        paper: Paper metadata dict.
        evidence: Aggregated evidence result.
        findings: List of finding dicts.
        figures_dir: Directory containing extracted figures.
        pdf_path: Path to the original PDF.
        output_path: Where to write the ZIP. Auto-generated if None.

    Returns:
        Path to the generated ZIP file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    doi_slug = (paper.get("doi") or "unknown").replace("/", "_")

    if output_path is None:
        output_path = f"evidence_{doi_slug}_{timestamp}.zip"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest: dict[str, Any] = {
        "rosette_version": _VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": paper.get("title"),
            "doi": paper.get("doi"),
            "journal": paper.get("journal"),
        },
        "assessment": {
            "risk_level": evidence.paper_risk,
            "overall_confidence": evidence.overall_confidence,
            "converging_evidence": evidence.converging_evidence,
            "total_findings": evidence.total_findings,
            "critical_count": evidence.critical_count,
        },
        "files": [],
    }

    if pdf_path and Path(pdf_path).exists():
        manifest["paper"]["pdf_sha256"] = _file_sha256(pdf_path)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        # Executive summary
        summary_md = _generate_executive_summary(paper, evidence, findings)
        zf.writestr("executive_summary.md", summary_md)
        manifest["files"].append(
            {
                "path": "executive_summary.md",
                "type": "executive_summary",
                "format": "markdown",
            }
        )

        # Raw findings
        findings_json = json.dumps(findings, indent=2, default=str)
        zf.writestr("findings.json", findings_json)
        manifest["files"].append(
            {
                "path": "findings.json",
                "type": "findings_data",
                "format": "json",
            }
        )

        # Figures
        if figures_dir:
            fig_dir = Path(figures_dir)
            if fig_dir.exists():
                for fig_path in sorted(fig_dir.iterdir()):
                    if fig_path.is_file() and fig_path.suffix.lower() in (
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".tif",
                        ".tiff",
                        ".bmp",
                    ):
                        arcname = f"figures/{fig_path.name}"
                        zf.write(fig_path, arcname)
                        manifest["files"].append(
                            {
                                "path": arcname,
                                "type": "figure",
                                "sha256": _file_sha256(str(fig_path)),
                            }
                        )

        # ELA artifacts
        ela_paths = set()
        for f in findings:
            evidence_json = f.get("evidence_json", "")
            if isinstance(evidence_json, str) and evidence_json:
                try:
                    ev = json.loads(evidence_json)
                    ela_path = ev.get("ela_image_path")
                    if ela_path and Path(ela_path).exists():
                        ela_paths.add(ela_path)
                except (json.JSONDecodeError, TypeError):
                    pass

        for ela_path in ela_paths:
            p = Path(ela_path)
            arcname = f"artifacts/{p.name}"
            zf.write(p, arcname)
            manifest["files"].append(
                {
                    "path": arcname,
                    "type": "ela_difference_image",
                    "sha256": _file_sha256(ela_path),
                }
            )

        # Write manifest last (after all files are added)
        manifest_json = json.dumps(manifest, indent=2, default=str)
        zf.writestr("manifest.json", manifest_json)

    logger.info("Evidence package written to %s", output)
    return output
