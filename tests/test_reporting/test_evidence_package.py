"""Tests for evidence package ZIP generation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from snoopy.analysis.evidence import AggregatedEvidence
from snoopy.reporting.evidence_package import (
    _file_sha256,
    _generate_executive_summary,
    generate_evidence_package,
)


@pytest.fixture
def sample_paper():
    return {
        "title": "Test Paper Title",
        "doi": "10.1234/test.paper",
        "journal": "Journal of Testing",
    }


@pytest.fixture
def sample_evidence():
    return AggregatedEvidence(
        paper_risk="high",
        overall_confidence=0.85,
        converging_evidence=True,
        total_findings=3,
        critical_count=1,
        figure_evidence=[],
    )


@pytest.fixture
def sample_findings():
    return [
        {
            "analysis_type": "ela",
            "method": "ela",
            "severity": "medium",
            "confidence": 0.7,
            "title": "ELA anomaly in Figure 1",
            "description": "Inconsistent compression levels detected.",
        },
        {
            "analysis_type": "clone_detection",
            "method": "clone_detection",
            "severity": "high",
            "confidence": 0.9,
            "title": "Clone region in Figure 2",
            "description": "Copy-move detected.",
        },
        {
            "analysis_type": "grim",
            "method": "grim",
            "severity": "critical",
            "confidence": 0.8,
            "title": "GRIM inconsistency",
            "description": "Mean not possible with reported N.",
        },
    ]


@pytest.fixture
def figures_dir(tmp_path):
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    for name in ("fig1.png", "fig2.png"):
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        img.save(str(fig_dir / name))
    return str(fig_dir)


class TestFileSha256:
    def test_computes_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = _file_sha256(str(f))
        assert len(h) == 64
        assert isinstance(h, str)

    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic")
        assert _file_sha256(str(f)) == _file_sha256(str(f))


class TestExecutiveSummary:
    def test_contains_paper_info(self, sample_paper, sample_evidence, sample_findings):
        md = _generate_executive_summary(sample_paper, sample_evidence, sample_findings)
        assert "Test Paper Title" in md
        assert "10.1234/test.paper" in md
        assert "Journal of Testing" in md

    def test_contains_risk_assessment(self, sample_paper, sample_evidence, sample_findings):
        md = _generate_executive_summary(sample_paper, sample_evidence, sample_findings)
        assert "HIGH" in md
        assert "0.85" in md
        assert "Converging Evidence" in md

    def test_contains_findings(self, sample_paper, sample_evidence, sample_findings):
        md = _generate_executive_summary(sample_paper, sample_evidence, sample_findings)
        assert "ELA anomaly" in md
        assert "Clone region" in md
        assert "GRIM inconsistency" in md

    def test_clean_paper(self, sample_paper):
        clean_evidence = AggregatedEvidence(
            paper_risk="clean",
            overall_confidence=0.1,
            converging_evidence=False,
            total_findings=0,
            critical_count=0,
            figure_evidence=[],
        )
        md = _generate_executive_summary(sample_paper, clean_evidence, [])
        assert "No integrity concerns" in md

    def test_lists_methods(self, sample_paper, sample_evidence, sample_findings):
        md = _generate_executive_summary(sample_paper, sample_evidence, sample_findings)
        assert "Methods Used" in md
        assert "Error Level Analysis" in md
        assert "Copy-Move Detection" in md


class TestGenerateEvidencePackage:
    def test_creates_zip(self, tmp_path, sample_paper, sample_evidence, sample_findings):
        out = str(tmp_path / "evidence.zip")
        result = generate_evidence_package(
            sample_paper, sample_evidence, sample_findings, output_path=out
        )
        assert result == Path(out)
        assert result.exists()
        assert zipfile.is_zipfile(out)

    def test_zip_contains_required_files(
        self, tmp_path, sample_paper, sample_evidence, sample_findings
    ):
        out = str(tmp_path / "evidence.zip")
        generate_evidence_package(
            sample_paper, sample_evidence, sample_findings, output_path=out
        )
        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
            assert "executive_summary.md" in names
            assert "findings.json" in names
            assert "manifest.json" in names

    def test_manifest_structure(
        self, tmp_path, sample_paper, sample_evidence, sample_findings
    ):
        out = str(tmp_path / "evidence.zip")
        generate_evidence_package(
            sample_paper, sample_evidence, sample_findings, output_path=out
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["paper"]["doi"] == "10.1234/test.paper"
            assert manifest["assessment"]["risk_level"] == "high"
            assert manifest["assessment"]["total_findings"] == 3
            assert isinstance(manifest["files"], list)
            assert len(manifest["files"]) >= 2  # At least summary + findings

    def test_includes_figures(
        self, tmp_path, sample_paper, sample_evidence, sample_findings, figures_dir
    ):
        out = str(tmp_path / "evidence.zip")
        generate_evidence_package(
            sample_paper,
            sample_evidence,
            sample_findings,
            figures_dir=figures_dir,
            output_path=out,
        )
        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
            assert "figures/fig1.png" in names
            assert "figures/fig2.png" in names

    def test_figure_sha256_in_manifest(
        self, tmp_path, sample_paper, sample_evidence, sample_findings, figures_dir
    ):
        out = str(tmp_path / "evidence.zip")
        generate_evidence_package(
            sample_paper,
            sample_evidence,
            sample_findings,
            figures_dir=figures_dir,
            output_path=out,
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            figure_files = [f for f in manifest["files"] if f["type"] == "figure"]
            assert len(figure_files) == 2
            for ff in figure_files:
                assert "sha256" in ff
                assert len(ff["sha256"]) == 64

    def test_findings_json_content(
        self, tmp_path, sample_paper, sample_evidence, sample_findings
    ):
        out = str(tmp_path / "evidence.zip")
        generate_evidence_package(
            sample_paper, sample_evidence, sample_findings, output_path=out
        )
        with zipfile.ZipFile(out, "r") as zf:
            findings = json.loads(zf.read("findings.json"))
            assert len(findings) == 3
            assert findings[0]["analysis_type"] == "ela"

    def test_auto_generated_output_path(
        self, tmp_path, sample_paper, sample_evidence, sample_findings, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = generate_evidence_package(
            sample_paper, sample_evidence, sample_findings
        )
        assert result.exists()
        assert result.suffix == ".zip"
        assert "10.1234_test.paper" in result.name
