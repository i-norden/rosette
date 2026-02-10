"""Tests for statistical value extraction from text."""

from snoopy.extraction.stats_extractor import (
    extract_means_and_ns,
    extract_numerical_values,
    extract_p_values,
    extract_test_statistics,
)


class TestExtractMeansAndNs:
    def test_standard_format(self):
        text = "The results showed M = 3.45, N = 120 for the treatment group."
        results = extract_means_and_ns(text)
        assert len(results) == 1
        assert abs(results[0].mean - 3.45) < 0.001
        assert results[0].n == 120

    def test_parenthetical_format(self):
        text = "mean = 4.20 (n = 50)"
        results = extract_means_and_ns(text)
        assert len(results) == 1
        assert abs(results[0].mean - 4.20) < 0.001
        assert results[0].n == 50

    def test_no_matches(self):
        text = "This text contains no statistical values."
        results = extract_means_and_ns(text)
        assert len(results) == 0


class TestExtractTestStatistics:
    def test_t_test(self):
        text = "t(30) = 2.45, p = 0.020"
        results = extract_test_statistics(text)
        assert len(results) >= 1
        t_results = [r for r in results if r.test_type == "t"]
        assert len(t_results) == 1
        assert abs(t_results[0].statistic - 2.45) < 0.001
        assert t_results[0].df == (30,)
        assert abs(t_results[0].p_value - 0.020) < 0.001

    def test_f_test(self):
        text = "F(2, 50) = 4.56, p = 0.015"
        results = extract_test_statistics(text)
        f_results = [r for r in results if r.test_type == "F"]
        assert len(f_results) == 1
        assert abs(f_results[0].statistic - 4.56) < 0.001
        assert f_results[0].df == (2, 50)

    def test_chi_squared(self):
        text = "chi2(3) = 10.5, p = 0.015"
        results = extract_test_statistics(text)
        chi2_results = [r for r in results if r.test_type == "chi2"]
        assert len(chi2_results) == 1
        assert abs(chi2_results[0].statistic - 10.5) < 0.001

    def test_no_p_value(self):
        text = "t(25) = 1.96"
        results = extract_test_statistics(text)
        t_results = [r for r in results if r.test_type == "t"]
        assert len(t_results) == 1
        assert t_results[0].p_value is None


class TestExtractPValues:
    def test_equals(self):
        text = "p = 0.045"
        results = extract_p_values(text)
        assert len(results) >= 1
        assert results[0].comparison == "="
        assert abs(results[0].value - 0.045) < 0.001

    def test_less_than(self):
        text = "p < .001"
        results = extract_p_values(text)
        assert len(results) >= 1
        assert results[0].comparison == "<"
        assert abs(results[0].value - 0.001) < 0.001

    def test_multiple(self):
        text = "p = 0.05 and p < 0.01 and p > 0.10"
        results = extract_p_values(text)
        assert len(results) == 3


class TestExtractNumericalValues:
    def test_basic(self):
        text = "The value was 3.14 and another was 2.71."
        values = extract_numerical_values(text)
        assert len(values) == 2
        assert abs(values[0] - 3.14) < 0.001
        assert abs(values[1] - 2.71) < 0.001

    def test_negative(self):
        text = "The change was -1.5 units."
        values = extract_numerical_values(text)
        assert len(values) == 1
        assert abs(values[0] - (-1.5)) < 0.001
