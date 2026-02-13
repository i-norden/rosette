from __future__ import annotations

from snoopy.extraction.stats_extractor import MeanReport, extract_means_sds_and_ns


class TestExtractMeanSDN:
    def test_standard_m_sd_n_format(self):
        text = "M = 3.45 (SD = 1.20), N = 30"
        results = extract_means_sds_and_ns(text)
        assert len(results) >= 1
        report = results[0]
        assert isinstance(report, MeanReport)
        assert report.mean == 3.45
        assert report.sd == 1.20
        assert report.n == 30


class TestExtractPlusMinus:
    def test_plus_minus_notation(self):
        text = "3.45 \u00b1 1.20 (N = 30)"
        results = extract_means_sds_and_ns(text)
        assert len(results) >= 1
        report = results[0]
        assert report.mean == 3.45
        assert report.sd == 1.20
        assert report.n == 30


class TestExtractLabeledMeanSD:
    def test_labeled_mean_plus_minus_sd(self):
        text = "mean \u00b1 SD: 5.67 \u00b1 2.34"
        results = extract_means_sds_and_ns(text)
        assert len(results) >= 1
        report = results[0]
        assert report.mean == 5.67
        assert report.sd == 2.34


class TestExtractMultipleReports:
    def test_multiple_mean_sd_patterns_extracted(self):
        text = (
            "Group A: M = 3.45 (SD = 1.20), N = 30. "
            "Group B: M = 4.10 (SD = 0.95), N = 25. "
            "Group C: M = 2.80 (SD = 1.55), N = 40."
        )
        results = extract_means_sds_and_ns(text)
        assert len(results) >= 3


class TestExtractNoSD:
    def test_mean_without_sd_not_extracted(self):
        text = "M = 3.45, N = 30"
        results = extract_means_sds_and_ns(text)
        assert len(results) == 0


class TestExtractEmptyText:
    def test_empty_string_returns_empty_list(self):
        results = extract_means_sds_and_ns("")
        assert results == []
