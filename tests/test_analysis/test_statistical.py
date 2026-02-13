"""Tests for statistical integrity checks."""

from snoopy.analysis.statistical import (
    benford_test,
    duplicate_value_check,
    grim_test,
    pvalue_check,
)


class TestGRIM:
    def test_consistent_mean(self):
        # 3.50 * 10 = 35.0, which is an integer
        result = grim_test(3.50, 10)
        assert result.consistent is True

    def test_inconsistent_mean(self):
        # 3.47 * 12 = 41.64, not close to integer for 2 decimal places
        result = grim_test(3.47, 12)
        assert result.consistent is False

    def test_edge_case_n_1(self):
        # With n=1, tolerance is very small (0.005), so only near-integer means pass
        result = grim_test(3.00, 1)
        assert result.consistent is True

    def test_large_n_tolerance_independent(self):
        # Tolerance is 0.5 / 10^decimals = 0.005, independent of N
        # 3.33 * 3 = 9.99, difference from 10 = 0.01 > 0.005
        result = grim_test(3.33, 3)
        assert result.consistent is False


class TestBenford:
    def test_benford_conforming_data(self):
        # Generate Benford-conforming data
        import random

        random.seed(42)
        values = []
        for _ in range(1000):
            # Logarithmic distribution follows Benford's law
            values.append(10 ** (random.random() * 4))
        result = benford_test(values)
        assert result.conforms is True
        assert result.n_values == 1000

    def test_uniform_leading_digits_fail(self):
        # All values starting with 5 - definitely not Benford
        values = [50 + i * 0.1 for i in range(500)]
        result = benford_test(values)
        assert result.conforms is False

    def test_empty_values(self):
        result = benford_test([])
        assert result.conforms is True
        assert result.n_values == 0


class TestPValueCheck:
    def test_consistent_t_test(self):
        # t(30) = 2.042 → p ≈ 0.05
        from scipy import stats

        expected_p = float(stats.t.sf(2.042, 30) * 2)
        result = pvalue_check("t", 2.042, (30,), expected_p)
        assert result.consistent is True

    def test_inconsistent_t_test(self):
        # Report p=0.001 for a t(30)=2.0, which should be ~0.054
        result = pvalue_check("t", 2.0, (30,), 0.001)
        assert result.consistent is False
        assert result.significance_changed is True

    def test_f_test(self):
        from scipy import stats

        expected_p = float(stats.f.sf(4.0, 2, 50))
        result = pvalue_check("F", 4.0, (2, 50), expected_p)
        assert result.consistent is True

    def test_chi2_test(self):
        from scipy import stats

        expected_p = float(stats.chi2.sf(10.0, 3))
        result = pvalue_check("chi2", 10.0, (3,), expected_p)
        assert result.consistent is True

    def test_r_correlation(self):
        # r=0.5 with n=30
        result = pvalue_check("r", 0.5, (30,), 0.005)
        assert result.consistent is True or result.difference < 0.02


class TestDuplicateValues:
    def test_no_duplicates(self):
        # Use non-round numbers to avoid triggering round-number bias
        data = [["1.23", "2.47", "3.89"], ["4.12", "5.67", "6.31"]]
        result = duplicate_value_check(data)
        assert result.suspicious is False

    def test_high_duplicates(self):
        data = [["3.14", "3.14", "3.14"], ["3.14", "3.14", "3.14"]]
        result = duplicate_value_check(data)
        assert result.duplicate_ratio > 0.5

    def test_empty_data(self):
        result = duplicate_value_check([])
        assert result.suspicious is False
        assert result.total_values == 0
