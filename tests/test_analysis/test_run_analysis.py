"""Tests for shared run_analysis wrapper functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from snoopy.analysis.run_analysis import (
    _default_config,
    run_dct_analysis,
    run_frequency_analysis,
    run_image_forensics,
    run_intra_paper_cross_ref,
    run_jpeg_ghost_analysis,
    run_sprite_analysis,
    run_statistical_tests,
    run_tortured_phrases,
    run_western_blot_analysis,
)
from snoopy.analysis.cross_reference import compute_phash
from snoopy.analysis.image_forensics import (
    CloneResult,
    DCTResult,
    ELAResult,
    FFTResult,
    JPEGGhostResult,
    NoiseResult,
)
from snoopy.analysis.types import FindingDict


class TestRunImageForensics:
    def test_run_image_forensics_returns_findings(self, sample_image: str) -> None:
        """Basic forensics on a sample image returns a list (possibly empty)."""
        findings = run_image_forensics(sample_image)
        assert isinstance(findings, list)
        # Each finding should be a dict with required keys
        for f in findings:
            assert isinstance(f, dict)
            assert "title" in f
            assert "analysis_type" in f
            assert "severity" in f
            assert "confidence" in f

    def test_run_image_forensics_clean_image(self, clean_image: str) -> None:
        """A clean gradient image should produce few or no high-severity findings."""
        findings = run_image_forensics(clean_image)
        assert isinstance(findings, list)
        high_findings = [f for f in findings if f["severity"] in ("high", "critical")]
        # A simple gradient image should not trigger high-severity alarms
        # Note: some minor findings may occur due to noise analysis thresholds
        assert len(high_findings) == 0

    def test_run_image_forensics_ela_exception(self, sample_image: str) -> None:
        """When ELA raises, other methods still run and produce findings."""
        with patch(
            "snoopy.analysis.run_analysis.error_level_analysis",
            side_effect=Exception("ELA boom"),
        ):
            findings = run_image_forensics(sample_image)
        assert isinstance(findings, list)
        # Should still have findings from clone/noise/metadata even if ELA fails
        methods = {f.get("analysis_type") for f in findings}
        assert "ela" not in methods

    def test_run_image_forensics_clone_exception(self, sample_image: str) -> None:
        """When clone detection raises, other methods still produce findings."""
        with patch(
            "snoopy.analysis.run_analysis.clone_detection",
            side_effect=Exception("Clone boom"),
        ):
            findings = run_image_forensics(sample_image)
        assert isinstance(findings, list)
        methods = {f.get("analysis_type") for f in findings}
        assert "clone_detection" not in methods

    def test_run_image_forensics_noise_exception(self, sample_image: str) -> None:
        """When noise analysis raises, other methods still produce findings."""
        with patch(
            "snoopy.analysis.run_analysis.noise_analysis",
            side_effect=Exception("Noise boom"),
        ):
            findings = run_image_forensics(sample_image)
        assert isinstance(findings, list)
        methods = {f.get("analysis_type") for f in findings}
        assert "noise_analysis" not in methods

    def test_run_image_forensics_all_methods_fail(self, sample_image: str) -> None:
        """When all methods raise, returns empty list without raising."""
        with (
            patch(
                "snoopy.analysis.run_analysis.error_level_analysis",
                side_effect=Exception("ELA"),
            ),
            patch(
                "snoopy.analysis.run_analysis.clone_detection",
                side_effect=Exception("Clone"),
            ),
            patch(
                "snoopy.analysis.run_analysis.noise_analysis",
                side_effect=Exception("Noise"),
            ),
            patch(
                "snoopy.analysis.metadata_forensics.analyze_metadata",
                side_effect=Exception("Metadata"),
            ),
        ):
            findings = run_image_forensics(sample_image)
        assert findings == []


class TestRunIntraPaperCrossRef:
    def test_run_intra_paper_cross_ref_no_duplicates(
        self, sample_image: str, clean_image: str
    ) -> None:
        """Distinct images within the same paper should find no duplicates."""
        hash1 = compute_phash(sample_image)
        hash2 = compute_phash(clean_image)

        figure_results = [
            {"image": "fig1.png", "phash": hash1, "ahash": None},
            {"image": "fig2.png", "phash": hash2, "ahash": None},
        ]

        findings = run_intra_paper_cross_ref(figure_results)
        assert isinstance(findings, list)
        # Very different images should not produce hash matches
        assert len(findings) == 0

    def test_run_intra_paper_cross_ref_duplicate_detected(self, sample_image: str) -> None:
        """Identical images should be detected as intra-paper duplicates."""
        hash1 = compute_phash(sample_image)

        figure_results = [
            {"image": "fig1.png", "phash": hash1, "ahash": None},
            {"image": "fig2.png", "phash": hash1, "ahash": None},
        ]

        findings = run_intra_paper_cross_ref(figure_results)
        assert isinstance(findings, list)
        assert len(findings) >= 1
        assert findings[0]["analysis_type"] == "phash"


class TestRunImageForensicsWithConfig:
    def test_custom_config(self, sample_image: str) -> None:
        """Custom config values should be passed through."""
        from snoopy.config import AnalysisConfig

        config = AnalysisConfig(ela={"quality": 50}, clone={"min_matches": 5})
        findings = run_image_forensics(sample_image, config=config)
        assert isinstance(findings, list)

    def test_custom_output_dir(self, sample_image: str, tmp_path) -> None:
        """Output dir parameter should be accepted."""
        findings = run_image_forensics(sample_image, output_dir=str(tmp_path))
        assert isinstance(findings, list)

    def test_figure_id_default(self, sample_image: str) -> None:
        """When figure_id is empty, file name is used."""
        findings = run_image_forensics(sample_image, figure_id="")
        for f in findings:
            assert f["figure_id"] != ""


class TestRunDctAnalysis:
    def test_dct_exception_returns_empty(self, sample_image: str) -> None:
        """DCT analysis exception returns empty list."""
        with patch(
            "snoopy.analysis.run_analysis.dct_analysis",
            side_effect=Exception("DCT boom"),
        ):
            findings = run_dct_analysis(sample_image)
        assert findings == []

    def test_dct_on_sample_image(self, sample_jpeg: str) -> None:
        """DCT analysis on a sample JPEG returns a list."""
        findings = run_dct_analysis(sample_jpeg)
        assert isinstance(findings, list)
        for f in findings:
            assert f["analysis_type"] == "dct_analysis"

    def test_dct_with_custom_config(self, sample_jpeg: str) -> None:
        """DCT analysis with custom config."""
        from snoopy.config import AnalysisConfig

        config = AnalysisConfig(dct_periodicity_threshold=0.1)
        findings = run_dct_analysis(sample_jpeg, config=config)
        assert isinstance(findings, list)


class TestRunJpegGhostAnalysis:
    def test_jpeg_ghost_exception_returns_empty(self, sample_image: str) -> None:
        """JPEG ghost exception returns empty list."""
        with patch(
            "snoopy.analysis.run_analysis.jpeg_ghost_detection",
            side_effect=Exception("Ghost boom"),
        ):
            findings = run_jpeg_ghost_analysis(sample_image)
        assert findings == []

    def test_jpeg_ghost_on_sample(self, sample_jpeg: str) -> None:
        """JPEG ghost on a sample JPEG returns a list."""
        findings = run_jpeg_ghost_analysis(sample_jpeg)
        assert isinstance(findings, list)


class TestRunFrequencyAnalysis:
    def test_frequency_exception_returns_empty(self, sample_image: str) -> None:
        """FFT analysis exception returns empty list."""
        with patch(
            "snoopy.analysis.run_analysis.frequency_analysis",
            side_effect=Exception("FFT boom"),
        ):
            findings = run_frequency_analysis(sample_image)
        assert findings == []

    def test_frequency_on_sample(self, sample_image: str) -> None:
        """FFT analysis on a sample image returns a list."""
        findings = run_frequency_analysis(sample_image)
        assert isinstance(findings, list)


class TestRunSpriteAnalysis:
    def test_sprite_exception_returns_empty(self) -> None:
        """SPRITE test exception returns empty list."""
        with patch(
            "snoopy.analysis.sprite.sprite_test",
            side_effect=Exception("SPRITE boom"),
        ):
            findings = run_sprite_analysis(3.5, 1.2, 10)
        assert findings == []


class TestRunStatisticalTests:
    def test_statistical_tests_exception_returns_empty(self) -> None:
        """Statistical test failures return empty list."""
        with patch(
            "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
            side_effect=Exception("Extract boom"),
        ):
            findings = run_statistical_tests("Some paper text with M=3.5, SD=1.2, N=10")
        assert findings == []


class TestRunTorturedPhrases:
    def test_tortured_phrases_exception_returns_empty(self) -> None:
        """Tortured phrase detection failure returns empty list."""
        with patch(
            "snoopy.analysis.text_forensics.detect_tortured_phrases",
            side_effect=Exception("TP boom"),
        ):
            findings = run_tortured_phrases("Some paper text")
        assert findings == []


class TestRunWesternBlotAnalysis:
    def test_run_western_blot_analysis(self, sample_image: str) -> None:
        """Western blot analysis on a sample image returns a list."""
        findings = run_western_blot_analysis(sample_image)
        assert isinstance(findings, list)
        # Each finding should be a dict with required keys
        for f in findings:
            assert isinstance(f, dict)
            assert "title" in f
            assert "analysis_type" in f
            assert f["analysis_type"] == "western_blot"

    def test_western_blot_exception_returns_empty(self, sample_image: str) -> None:
        """Western blot exception returns empty list."""
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            side_effect=Exception("WB boom"),
        ):
            findings = run_western_blot_analysis(sample_image)
        assert findings == []


class TestDefaultConfig:
    def test_default_config_returns_analysis_config(self) -> None:
        cfg = _default_config()
        assert cfg.ela.quality == 80


class TestFindingDictType:
    def test_finding_dict_instantiation(self) -> None:
        """FindingDict TypedDict can be instantiated."""
        fd: FindingDict = {
            "title": "test",
            "analysis_type": "ela",
            "method": "ela",
            "severity": "high",
            "confidence": 0.9,
            "description": "desc",
            "figure_id": "fig1",
            "evidence": {"key": "val"},
        }
        assert fd["title"] == "test"
        assert fd["confidence"] == 0.9


class TestELASeverityBranches:
    """Tests that exercise all ELA severity branches via mocking."""

    def _make_ela_result(
        self, suspicious: bool, max_diff: float, mean_diff: float, std_diff: float
    ) -> ELAResult:
        return ELAResult(
            suspicious=suspicious,
            max_difference=max_diff,
            mean_difference=mean_diff,
            std_difference=std_diff,
            ela_image_path=None,
        )

    def test_ela_high_severity(self, sample_image: str) -> None:
        """ELA with max_diff >= high threshold and > mean + 3*std -> high severity."""
        result = self._make_ela_result(True, max_diff=100.0, mean_diff=10.0, std_diff=5.0)
        with patch("snoopy.analysis.run_analysis.error_level_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        ela_findings = [f for f in findings if f["analysis_type"] == "ela"]
        assert len(ela_findings) >= 1
        assert ela_findings[0]["severity"] == "high"

    def test_ela_medium_severity(self, sample_image: str) -> None:
        """ELA with max_diff >= medium threshold -> medium severity."""
        # medium threshold default is 40, need max_diff >= 40 and > mean + 3*std
        # but max_diff < high threshold (60)
        result = self._make_ela_result(True, max_diff=45.0, mean_diff=5.0, std_diff=3.0)
        with patch("snoopy.analysis.run_analysis.error_level_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        ela_findings = [f for f in findings if f["analysis_type"] == "ela"]
        assert len(ela_findings) >= 1
        assert ela_findings[0]["severity"] == "medium"

    def test_ela_low_severity_above_threshold(self, sample_image: str) -> None:
        """ELA with max_diff >= low threshold -> low severity."""
        # low threshold default is 20, need >= 20, > mean + 2*std, but < medium threshold
        result = self._make_ela_result(True, max_diff=25.0, mean_diff=5.0, std_diff=3.0)
        with patch("snoopy.analysis.run_analysis.error_level_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        ela_findings = [f for f in findings if f["analysis_type"] == "ela"]
        assert len(ela_findings) >= 1
        assert ela_findings[0]["severity"] == "low"

    def test_ela_low_severity_fallback(self, sample_image: str) -> None:
        """ELA suspicious but below all thresholds -> low severity fallback."""
        result = self._make_ela_result(True, max_diff=10.0, mean_diff=8.0, std_diff=1.0)
        with patch("snoopy.analysis.run_analysis.error_level_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        ela_findings = [f for f in findings if f["analysis_type"] == "ela"]
        assert len(ela_findings) >= 1
        assert ela_findings[0]["severity"] == "low"

    def test_ela_not_suspicious(self, sample_image: str) -> None:
        """ELA not suspicious should produce no ELA finding."""
        result = self._make_ela_result(False, max_diff=5.0, mean_diff=3.0, std_diff=1.0)
        with patch("snoopy.analysis.run_analysis.error_level_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        ela_findings = [f for f in findings if f["analysis_type"] == "ela"]
        assert len(ela_findings) == 0


class TestCloneSeverityBranches:
    """Tests that exercise all clone detection severity branches."""

    def _make_clone_result(
        self, suspicious: bool, num_matches: int, inlier_ratio: float
    ) -> CloneResult:
        return CloneResult(
            suspicious=suspicious,
            num_matches=num_matches,
            match_clusters=[{"id": 1}],
            inlier_ratio=inlier_ratio,
        )

    def test_clone_high_severity(self, sample_image: str) -> None:
        """Clone with high inliers and ratio -> high severity."""
        result = self._make_clone_result(True, num_matches=60, inlier_ratio=0.7)
        with patch("snoopy.analysis.run_analysis.clone_detection", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        clone_findings = [f for f in findings if f["analysis_type"] == "clone_detection"]
        assert len(clone_findings) >= 1
        assert clone_findings[0]["severity"] == "high"

    def test_clone_medium_severity(self, sample_image: str) -> None:
        """Clone with medium inliers and ratio -> medium severity."""
        result = self._make_clone_result(True, num_matches=45, inlier_ratio=0.30)
        with patch("snoopy.analysis.run_analysis.clone_detection", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        clone_findings = [f for f in findings if f["analysis_type"] == "clone_detection"]
        assert len(clone_findings) >= 1
        assert clone_findings[0]["severity"] == "medium"

    def test_clone_low_severity(self, sample_image: str) -> None:
        """Clone with low inliers -> low severity."""
        result = self._make_clone_result(True, num_matches=12, inlier_ratio=0.25)
        with patch("snoopy.analysis.run_analysis.clone_detection", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        clone_findings = [f for f in findings if f["analysis_type"] == "clone_detection"]
        assert len(clone_findings) >= 1
        assert clone_findings[0]["severity"] == "low"

    def test_clone_low_severity_fallback(self, sample_image: str) -> None:
        """Clone suspicious but below low thresholds -> low severity fallback."""
        result = self._make_clone_result(True, num_matches=5, inlier_ratio=0.1)
        with patch("snoopy.analysis.run_analysis.clone_detection", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        clone_findings = [f for f in findings if f["analysis_type"] == "clone_detection"]
        assert len(clone_findings) >= 1
        assert clone_findings[0]["severity"] == "low"
        # Fallback confidence uses ratio * 0.4
        assert clone_findings[0]["confidence"] == pytest.approx(0.1 * 0.4)


class TestNoiseSeverityBranches:
    """Tests that exercise all noise analysis severity branches."""

    def _make_noise_result(self, suspicious: bool, max_ratio: float) -> NoiseResult:
        return NoiseResult(
            suspicious=suspicious,
            noise_map=[],
            mean_noise=5.0,
            noise_std=1.0,
            max_ratio=max_ratio,
        )

    def test_noise_high_severity(self, sample_image: str) -> None:
        """Noise with high max_ratio -> high severity."""
        result = self._make_noise_result(True, max_ratio=55.0)
        with patch("snoopy.analysis.run_analysis.noise_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        noise_findings = [f for f in findings if f["analysis_type"] == "noise_analysis"]
        assert len(noise_findings) >= 1
        assert noise_findings[0]["severity"] == "high"

    def test_noise_medium_severity(self, sample_image: str) -> None:
        """Noise with medium max_ratio -> medium severity."""
        result = self._make_noise_result(True, max_ratio=30.0)
        with patch("snoopy.analysis.run_analysis.noise_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        noise_findings = [f for f in findings if f["analysis_type"] == "noise_analysis"]
        assert len(noise_findings) >= 1
        assert noise_findings[0]["severity"] == "medium"

    def test_noise_low_severity(self, sample_image: str) -> None:
        """Noise with low max_ratio -> low severity."""
        result = self._make_noise_result(True, max_ratio=12.0)
        with patch("snoopy.analysis.run_analysis.noise_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        noise_findings = [f for f in findings if f["analysis_type"] == "noise_analysis"]
        assert len(noise_findings) >= 1
        assert noise_findings[0]["severity"] == "low"

    def test_noise_low_severity_fallback(self, sample_image: str) -> None:
        """Noise suspicious but below all thresholds -> low severity fallback."""
        result = self._make_noise_result(True, max_ratio=2.0)
        with patch("snoopy.analysis.run_analysis.noise_analysis", return_value=result):
            findings = run_image_forensics(sample_image, figure_id="test")
        noise_findings = [f for f in findings if f["analysis_type"] == "noise_analysis"]
        assert len(noise_findings) >= 1
        assert noise_findings[0]["severity"] == "low"


class TestMetadataSuspicious:
    """Tests for metadata forensics suspicious results."""

    def test_metadata_suspicious_produces_findings(self, sample_image: str) -> None:
        """Metadata forensics with suspicious findings produces output."""
        from snoopy.analysis.metadata_forensics import MetadataFinding, MetadataForensicsResult

        meta_result = MetadataForensicsResult(
            suspicious=True,
            findings=[
                MetadataFinding(
                    finding_type="software_mismatch",
                    description="Software does not match expected",
                    confidence=0.7,
                    severity="medium",
                )
            ],
            software="FakeEditor 1.0",
            icc_profile="sRGB",
        )
        with patch(
            "snoopy.analysis.metadata_forensics.analyze_metadata",
            return_value=meta_result,
        ):
            findings = run_image_forensics(sample_image, figure_id="test")
        meta_findings = [f for f in findings if f["analysis_type"] == "metadata_forensics"]
        assert len(meta_findings) >= 1
        assert meta_findings[0]["severity"] == "medium"
        assert meta_findings[0]["evidence"]["software"] == "FakeEditor 1.0"


class TestDCTSeverityBranches:
    """Tests that exercise all DCT severity branches via mocking."""

    def test_dct_high_severity(self, sample_image: str) -> None:
        """DCT with high periodicity score -> high severity."""
        result = DCTResult(
            suspicious=True,
            periodicity_score=0.8,
            estimated_primary_quality=75,
            block_inconsistencies=10,
            details="High periodicity detected",
        )
        with patch("snoopy.analysis.run_analysis.dct_analysis", return_value=result):
            findings = run_dct_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "high"

    def test_dct_medium_severity(self, sample_image: str) -> None:
        """DCT with medium periodicity score -> medium severity."""
        result = DCTResult(
            suspicious=True,
            periodicity_score=0.6,
            estimated_primary_quality=80,
            block_inconsistencies=5,
            details="Medium periodicity",
        )
        with patch("snoopy.analysis.run_analysis.dct_analysis", return_value=result):
            findings = run_dct_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "medium"

    def test_dct_low_severity(self, sample_image: str) -> None:
        """DCT with low periodicity score -> low severity."""
        result = DCTResult(
            suspicious=True,
            periodicity_score=0.35,
            estimated_primary_quality=90,
            block_inconsistencies=2,
            details="Low periodicity",
        )
        with patch("snoopy.analysis.run_analysis.dct_analysis", return_value=result):
            findings = run_dct_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "low"

    def test_dct_not_suspicious(self, sample_image: str) -> None:
        """DCT not suspicious returns empty."""
        result = DCTResult(
            suspicious=False,
            periodicity_score=0.1,
            estimated_primary_quality=None,
            block_inconsistencies=0,
            details="Clean",
        )
        with patch("snoopy.analysis.run_analysis.dct_analysis", return_value=result):
            findings = run_dct_analysis(sample_image)
        assert findings == []


class TestJPEGGhostSeverityBranches:
    """Tests that exercise all JPEG ghost severity branches via mocking."""

    def test_jpeg_ghost_high_severity(self, sample_image: str) -> None:
        """JPEG ghost with many regions -> high severity."""
        result = JPEGGhostResult(
            suspicious=True,
            ghost_regions=[{"r": 1}, {"r": 2}, {"r": 3}],
            quality_map=[],
            dominant_quality=75,
            quality_variance=120.0,
            details="Multiple ghost regions",
        )
        with patch("snoopy.analysis.run_analysis.jpeg_ghost_detection", return_value=result):
            findings = run_jpeg_ghost_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "high"

    def test_jpeg_ghost_medium_severity(self, sample_image: str) -> None:
        """JPEG ghost with 1-2 regions -> medium severity."""
        result = JPEGGhostResult(
            suspicious=True,
            ghost_regions=[{"r": 1}],
            quality_map=[],
            dominant_quality=80,
            quality_variance=50.0,
            details="Single ghost region",
        )
        with patch("snoopy.analysis.run_analysis.jpeg_ghost_detection", return_value=result):
            findings = run_jpeg_ghost_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "medium"

    def test_jpeg_ghost_low_severity(self, sample_image: str) -> None:
        """JPEG ghost with no regions but suspicious -> low severity."""
        result = JPEGGhostResult(
            suspicious=True,
            ghost_regions=[],
            quality_map=[],
            dominant_quality=85,
            quality_variance=10.0,
            details="Minor ghost",
        )
        with patch("snoopy.analysis.run_analysis.jpeg_ghost_detection", return_value=result):
            findings = run_jpeg_ghost_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "low"
        assert findings[0]["confidence"] == 0.4

    def test_jpeg_ghost_not_suspicious(self, sample_image: str) -> None:
        """JPEG ghost not suspicious returns empty."""
        result = JPEGGhostResult(
            suspicious=False,
            ghost_regions=[],
            quality_map=[],
            dominant_quality=85,
            quality_variance=5.0,
            details="Clean",
        )
        with patch("snoopy.analysis.run_analysis.jpeg_ghost_detection", return_value=result):
            findings = run_jpeg_ghost_analysis(sample_image)
        assert findings == []


class TestFFTSeverityBranches:
    """Tests that exercise all FFT severity branches via mocking."""

    def test_fft_high_severity(self, sample_image: str) -> None:
        """FFT with high spectral anomaly -> high severity."""
        result = FFTResult(
            suspicious=True,
            spectral_anomaly_score=6.0,
            periodic_peaks=[1.0, 2.0],
            high_freq_ratio=0.3,
            details="Major spectral anomaly",
        )
        with patch("snoopy.analysis.run_analysis.frequency_analysis", return_value=result):
            findings = run_frequency_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "high"

    def test_fft_medium_severity(self, sample_image: str) -> None:
        """FFT with medium spectral anomaly -> medium severity."""
        result = FFTResult(
            suspicious=True,
            spectral_anomaly_score=4.0,
            periodic_peaks=[1.0],
            high_freq_ratio=0.2,
            details="Medium spectral anomaly",
        )
        with patch("snoopy.analysis.run_analysis.frequency_analysis", return_value=result):
            findings = run_frequency_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "medium"

    def test_fft_low_severity(self, sample_image: str) -> None:
        """FFT with low spectral anomaly -> low severity."""
        result = FFTResult(
            suspicious=True,
            spectral_anomaly_score=2.8,
            periodic_peaks=[],
            high_freq_ratio=0.1,
            details="Minor spectral anomaly",
        )
        with patch("snoopy.analysis.run_analysis.frequency_analysis", return_value=result):
            findings = run_frequency_analysis(sample_image, figure_id="test")
        assert len(findings) == 1
        assert findings[0]["severity"] == "low"

    def test_fft_not_suspicious(self, sample_image: str) -> None:
        """FFT not suspicious returns empty."""
        result = FFTResult(
            suspicious=False,
            spectral_anomaly_score=1.0,
            periodic_peaks=[],
            high_freq_ratio=0.05,
            details="Clean spectrum",
        )
        with patch("snoopy.analysis.run_analysis.frequency_analysis", return_value=result):
            findings = run_frequency_analysis(sample_image)
        assert findings == []


class TestSpriteWithResults:
    """Tests for SPRITE analysis producing actual findings."""

    def test_sprite_inconsistent_sd_not_achievable(self) -> None:
        """SPRITE test where SD is not achievable -> high severity."""
        from snoopy.analysis.sprite import SPRITEResult

        result = SPRITEResult(
            consistent=False,
            reported_mean=3.5,
            reported_sd=0.01,
            n=10,
            min_val=1,
            max_val=7,
            mean_achievable=True,
            sd_achievable=False,
            attempts=1000,
            details="SD not achievable",
        )
        with patch("snoopy.analysis.sprite.sprite_test", return_value=result):
            findings = run_sprite_analysis(3.5, 0.01, 10, context="Test context")
        assert len(findings) == 1
        assert findings[0]["severity"] == "high"
        assert findings[0]["confidence"] == 0.8
        assert findings[0]["evidence"]["context"] == "Test context"

    def test_sprite_inconsistent_sd_achievable(self) -> None:
        """SPRITE test where SD is achievable but mean isn't -> medium severity."""
        from snoopy.analysis.sprite import SPRITEResult

        result = SPRITEResult(
            consistent=False,
            reported_mean=3.5,
            reported_sd=1.2,
            n=10,
            min_val=1,
            max_val=7,
            mean_achievable=False,
            sd_achievable=True,
            attempts=1000,
            details="Mean not achievable with this SD",
        )
        with patch("snoopy.analysis.sprite.sprite_test", return_value=result):
            findings = run_sprite_analysis(3.5, 1.2, 10)
        assert len(findings) == 1
        assert findings[0]["severity"] == "medium"
        assert findings[0]["confidence"] == 0.65

    def test_sprite_consistent(self) -> None:
        """SPRITE consistent returns empty."""
        from snoopy.analysis.sprite import SPRITEResult

        result = SPRITEResult(
            consistent=True,
            reported_mean=3.5,
            reported_sd=1.2,
            n=10,
            min_val=1,
            max_val=7,
            mean_achievable=True,
            sd_achievable=True,
            attempts=1000,
            details="Consistent",
        )
        with patch("snoopy.analysis.sprite.sprite_test", return_value=result):
            findings = run_sprite_analysis(3.5, 1.2, 10)
        assert findings == []


class TestStatisticalTestsBranches:
    """Tests for statistical test branches with mocked results."""

    def test_grimmer_inconsistent_produces_finding(self) -> None:
        """GRIMMER inconsistency produces high severity finding."""
        from snoopy.analysis.statistical import GRIMMERResult
        from snoopy.extraction.stats_extractor import MeanReport

        mean_report = MeanReport(mean=3.5, sd=1.2, n=10, context="Test context")
        grimmer_result = GRIMMERResult(
            consistent=False,
            mean=3.5,
            sd=1.2,
            n=10,
            possible_sds=[1.1, 1.3],
            details="SD inconsistent with GRIMMER",
        )
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[mean_report],
            ),
            patch(
                "snoopy.analysis.statistical.grimmer_test",
                return_value=grimmer_result,
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=[],
            ),
        ):
            findings = run_statistical_tests("text with M=3.5, SD=1.2, N=10")
        grimmer_findings = [f for f in findings if f["analysis_type"] == "grimmer"]
        assert len(grimmer_findings) == 1
        assert grimmer_findings[0]["severity"] == "high"

    def test_variance_ratio_suspicious(self) -> None:
        """Variance ratio test produces finding when suspicious."""
        from snoopy.analysis.statistical import VarianceRatioResult
        from snoopy.extraction.stats_extractor import MeanReport

        reports = [
            MeanReport(mean=3.0, sd=1.0, n=20, context="c1"),
            MeanReport(mean=4.0, sd=1.0, n=20, context="c2"),
            MeanReport(mean=5.0, sd=1.0, n=20, context="c3"),
        ]
        vr_result = VarianceRatioResult(
            suspicious=True,
            n_groups=3,
            observed_variance_of_sds=0.001,
            expected_variance_of_sds=0.1,
            ratio=0.01,
            p_value=0.005,
            details="Suspiciously uniform SDs",
        )
        grimmer_consistent = type("GRIMMERResult", (), {"consistent": True})()
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=reports,
            ),
            patch(
                "snoopy.analysis.statistical.grimmer_test",
                return_value=grimmer_consistent,
            ),
            patch(
                "snoopy.analysis.statistical.variance_ratio_test",
                return_value=vr_result,
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=[],
            ),
        ):
            findings = run_statistical_tests("text")
        vr_findings = [f for f in findings if f["analysis_type"] == "variance_ratio"]
        assert len(vr_findings) == 1
        assert vr_findings[0]["severity"] == "high"

    def test_terminal_digit_suspicious(self) -> None:
        """Terminal digit test produces finding when suspicious."""
        from snoopy.analysis.statistical import TerminalDigitResult

        td_result = TerminalDigitResult(
            suspicious=True,
            chi_squared=30.0,
            p_value=0.0001,
            n_values=100,
            digit_counts={str(i): 10 for i in range(10)},
            details="Non-uniform terminal digits",
        )
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[],
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=list(range(30)),
            ),
            patch(
                "snoopy.analysis.statistical.terminal_digit_test",
                return_value=td_result,
            ),
        ):
            findings = run_statistical_tests("text with numbers")
        td_findings = [f for f in findings if f["analysis_type"] == "terminal_digit"]
        assert len(td_findings) == 1
        assert td_findings[0]["severity"] == "medium"

    def test_terminal_digit_low_severity(self) -> None:
        """Terminal digit with higher p-value -> low severity."""
        from snoopy.analysis.statistical import TerminalDigitResult

        td_result = TerminalDigitResult(
            suspicious=True,
            chi_squared=20.0,
            p_value=0.01,
            n_values=50,
            digit_counts={str(i): 5 for i in range(10)},
            details="Slightly non-uniform",
        )
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[],
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=list(range(30)),
            ),
            patch(
                "snoopy.analysis.statistical.terminal_digit_test",
                return_value=td_result,
            ),
        ):
            findings = run_statistical_tests("text")
        td_findings = [f for f in findings if f["analysis_type"] == "terminal_digit"]
        assert len(td_findings) == 1
        assert td_findings[0]["severity"] == "low"

    def test_variance_ratio_exception(self) -> None:
        """Variance ratio exception returns findings from other tests only."""
        from snoopy.extraction.stats_extractor import MeanReport

        reports = [
            MeanReport(mean=3.0, sd=1.0, n=20, context="c1"),
            MeanReport(mean=4.0, sd=1.0, n=20, context="c2"),
            MeanReport(mean=5.0, sd=1.0, n=20, context="c3"),
        ]
        grimmer_consistent = type("GRIMMERResult", (), {"consistent": True})()
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=reports,
            ),
            patch(
                "snoopy.analysis.statistical.grimmer_test",
                return_value=grimmer_consistent,
            ),
            patch(
                "snoopy.analysis.statistical.variance_ratio_test",
                side_effect=Exception("VR boom"),
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=[],
            ),
        ):
            findings = run_statistical_tests("text")
        # No variance ratio findings, but no crash
        assert isinstance(findings, list)

    def test_terminal_digit_exception(self) -> None:
        """Terminal digit exception is caught gracefully."""
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[],
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=list(range(30)),
            ),
            patch(
                "snoopy.analysis.statistical.terminal_digit_test",
                side_effect=Exception("TD boom"),
            ),
        ):
            findings = run_statistical_tests("text")
        assert isinstance(findings, list)

    def test_too_few_values_skips_terminal_digit(self) -> None:
        """< 20 values skips terminal digit test."""
        with (
            patch(
                "snoopy.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[],
            ),
            patch(
                "snoopy.extraction.stats_extractor.extract_numerical_values",
                return_value=list(range(10)),
            ),
        ):
            findings = run_statistical_tests("text")
        assert findings == []


class TestTorturedPhrasesBranches:
    """Tests for tortured phrase detection result branches."""

    def _make_tp_result(self, unique_phrases: int):
        from snoopy.analysis.text_forensics import TorturedPhraseMatch, TorturedPhraseResult

        matches = [
            TorturedPhraseMatch(
                tortured_phrase=f"phrase_{i}",
                correct_phrase=f"correct_{i}",
                position=i * 10,
                context=f"context {i}",
            )
            for i in range(unique_phrases)
        ]
        return TorturedPhraseResult(
            suspicious=True,
            matches=matches,
            match_count=unique_phrases,
            unique_phrases=unique_phrases,
            details=f"Found {unique_phrases} tortured phrases",
        )

    def test_tortured_phrases_critical(self) -> None:
        """5+ unique tortured phrases -> critical severity."""
        result = self._make_tp_result(6)
        with patch(
            "snoopy.analysis.text_forensics.detect_tortured_phrases",
            return_value=result,
        ):
            findings = run_tortured_phrases("some text")
        assert len(findings) == 1
        assert findings[0]["severity"] == "critical"

    def test_tortured_phrases_high(self) -> None:
        """3-4 unique tortured phrases -> high severity."""
        result = self._make_tp_result(3)
        with patch(
            "snoopy.analysis.text_forensics.detect_tortured_phrases",
            return_value=result,
        ):
            findings = run_tortured_phrases("some text")
        assert len(findings) == 1
        assert findings[0]["severity"] == "high"

    def test_tortured_phrases_medium(self) -> None:
        """1-2 unique tortured phrases -> medium severity."""
        result = self._make_tp_result(2)
        with patch(
            "snoopy.analysis.text_forensics.detect_tortured_phrases",
            return_value=result,
        ):
            findings = run_tortured_phrases("some text")
        assert len(findings) == 1
        assert findings[0]["severity"] == "medium"

    def test_tortured_phrases_not_suspicious(self) -> None:
        """Not suspicious returns empty."""
        from snoopy.analysis.text_forensics import TorturedPhraseResult

        result = TorturedPhraseResult(
            suspicious=False, matches=[], match_count=0, unique_phrases=0, details=""
        )
        with patch(
            "snoopy.analysis.text_forensics.detect_tortured_phrases",
            return_value=result,
        ):
            findings = run_tortured_phrases("some text")
        assert findings == []


class TestWesternBlotBranches:
    """Tests for western blot analysis result branches."""

    def test_western_blot_duplicate_lanes(self, sample_image: str) -> None:
        """Western blot with duplicate lanes produces high severity finding."""
        from snoopy.analysis.western_blot import WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[(0, 2, 0.98)],
            splice_boundaries=[],
            uniform_profiles=False,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        dup_findings = [f for f in findings if "Duplicate lanes" in f["title"]]
        assert len(dup_findings) == 1
        assert dup_findings[0]["severity"] == "high"
        assert dup_findings[0]["evidence"]["correlation"] == 0.98

    def test_western_blot_splice_boundary_high(self, sample_image: str) -> None:
        """Western blot with high confidence splice -> high severity."""
        from snoopy.analysis.western_blot import SpliceBoundary, WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[],
            splice_boundaries=[
                SpliceBoundary(
                    x_position=100,
                    left_lane=1,
                    right_lane=2,
                    background_discontinuity=0.8,
                    noise_discontinuity=0.6,
                    confidence=0.85,
                )
            ],
            uniform_profiles=False,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        splice_findings = [f for f in findings if "Splice boundary" in f["title"]]
        assert len(splice_findings) == 1
        assert splice_findings[0]["severity"] == "high"

    def test_western_blot_splice_boundary_medium(self, sample_image: str) -> None:
        """Western blot with medium confidence splice -> medium severity."""
        from snoopy.analysis.western_blot import SpliceBoundary, WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[],
            splice_boundaries=[
                SpliceBoundary(
                    x_position=100,
                    left_lane=1,
                    right_lane=2,
                    background_discontinuity=0.3,
                    noise_discontinuity=0.2,
                    confidence=0.55,
                )
            ],
            uniform_profiles=False,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        splice_findings = [f for f in findings if "Splice boundary" in f["title"]]
        assert len(splice_findings) == 1
        assert splice_findings[0]["severity"] == "medium"

    def test_western_blot_splice_low_confidence_skipped(self, sample_image: str) -> None:
        """Western blot splice with confidence <= 0.5 is skipped."""
        from snoopy.analysis.western_blot import SpliceBoundary, WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[],
            splice_boundaries=[
                SpliceBoundary(
                    x_position=100,
                    left_lane=1,
                    right_lane=2,
                    background_discontinuity=0.1,
                    noise_discontinuity=0.1,
                    confidence=0.3,
                )
            ],
            uniform_profiles=False,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        splice_findings = [f for f in findings if "Splice boundary" in f["title"]]
        assert len(splice_findings) == 0

    def test_western_blot_uniform_profiles(self, sample_image: str) -> None:
        """Uniform profiles without duplicates produces medium finding."""
        from snoopy.analysis.western_blot import WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[],
            splice_boundaries=[],
            uniform_profiles=True,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        uniform_findings = [f for f in findings if "uniform" in f["title"].lower()]
        assert len(uniform_findings) == 1
        assert uniform_findings[0]["severity"] == "medium"

    def test_western_blot_uniform_with_duplicates_suppressed(self, sample_image: str) -> None:
        """Uniform profiles with duplicates does not produce uniform finding."""
        from snoopy.analysis.western_blot import WesternBlotResult

        result = WesternBlotResult(
            suspicious=True,
            lane_count=4,
            duplicate_lanes=[(0, 1, 0.99)],
            splice_boundaries=[],
            uniform_profiles=True,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image, figure_id="test")
        uniform_findings = [f for f in findings if "uniform" in f["title"].lower()]
        assert len(uniform_findings) == 0

    def test_western_blot_not_suspicious(self, sample_image: str) -> None:
        """Western blot not suspicious returns empty."""
        from snoopy.analysis.western_blot import WesternBlotResult

        result = WesternBlotResult(
            suspicious=False,
            lane_count=4,
            duplicate_lanes=[],
            splice_boundaries=[],
            uniform_profiles=False,
        )
        with patch(
            "snoopy.analysis.western_blot.analyze_western_blot",
            return_value=result,
        ):
            findings = run_western_blot_analysis(sample_image)
        assert findings == []


class TestIntraPaperCrossRefBranches:
    """Tests for cross-ref severity branches based on hash distance."""

    def test_critical_severity_exact_match(self) -> None:
        """Hash distance 0 (exact match) -> critical severity."""
        findings = run_intra_paper_cross_ref(
            [
                {"image": "fig1.png", "phash": "abcdef1234567890", "ahash": None},
                {"image": "fig2.png", "phash": "abcdef1234567890", "ahash": None},
            ]
        )
        assert len(findings) >= 1
        assert findings[0]["severity"] == "critical"
        assert findings[0]["confidence"] == 0.95

    def test_missing_phash_skipped(self) -> None:
        """Entries with no phash are skipped."""
        findings = run_intra_paper_cross_ref(
            [
                {"image": "fig1.png", "phash": None, "ahash": None},
                {"image": "fig2.png", "phash": None, "ahash": None},
            ]
        )
        assert findings == []

    def test_value_error_in_hash_distance(self) -> None:
        """ValueError in hash_distance is caught and skipped."""
        with patch(
            "snoopy.analysis.run_analysis.hash_distance",
            side_effect=ValueError("bad hash"),
        ):
            findings = run_intra_paper_cross_ref(
                [
                    {"image": "fig1.png", "phash": "abc", "ahash": None},
                    {"image": "fig2.png", "phash": "def", "ahash": None},
                ]
            )
        assert findings == []
