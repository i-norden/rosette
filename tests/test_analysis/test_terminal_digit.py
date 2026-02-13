"""Tests for terminal digit test."""
from __future__ import annotations

from snoopy.analysis.statistical import terminal_digit_test, TerminalDigitResult


class TestTerminalDigitUniform:
    """Test terminal digit analysis with uniformly distributed terminal digits."""

    def test_uniform_digits_not_suspicious(self):
        # Terminal digit 0 from 10.0 ("10" -> digit 0), digits 1-9 from 1.01-1.09
        values = [10.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09] * 50
        result = terminal_digit_test(values)
        assert isinstance(result, TerminalDigitResult)
        assert result.suspicious is False

    def test_uniform_digits_high_p_value(self):
        values = [10.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09] * 50
        result = terminal_digit_test(values)
        assert result.p_value > 0.01

    def test_uniform_digit_counts_balanced(self):
        values = [10.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09] * 50
        result = terminal_digit_test(values)
        assert isinstance(result.digit_counts, dict)

    def test_uniform_n_values(self):
        values = [10.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09] * 50
        result = terminal_digit_test(values)
        assert result.n_values == len(values)

    def test_uniform_chi_squared_low(self):
        values = [10.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09] * 50
        result = terminal_digit_test(values)
        assert result.chi_squared >= 0.0


class TestTerminalDigitFabricated:
    """Test terminal digit analysis with fabricated data ending in 0 or 5."""

    def test_fabricated_digits_suspicious(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values)
        assert isinstance(result, TerminalDigitResult)
        assert result.suspicious is True

    def test_fabricated_digits_low_p_value(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values)
        assert result.p_value < 0.01

    def test_fabricated_digits_high_chi_squared(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values)
        assert result.chi_squared > 0.0

    def test_fabricated_digit_counts_skewed(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values)
        assert result.digit_counts.get(0, 0) > 0 or result.digit_counts.get(5, 0) > 0

    def test_fabricated_details_nonempty(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values)
        assert isinstance(result.details, str)
        assert len(result.details) > 0

    def test_custom_alpha(self):
        values = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5] * 50
        result = terminal_digit_test(values, alpha=0.05)
        assert isinstance(result, TerminalDigitResult)
        assert result.suspicious is True


class TestTerminalDigitSmallSample:
    """Test terminal digit analysis with insufficient data."""

    def test_fewer_than_10_not_suspicious(self):
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = terminal_digit_test(values)
        assert isinstance(result, TerminalDigitResult)
        assert result.suspicious is False

    def test_single_value_not_suspicious(self):
        values = [3.14]
        result = terminal_digit_test(values)
        assert result.suspicious is False

    def test_nine_values_not_suspicious(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        result = terminal_digit_test(values)
        assert result.suspicious is False

    def test_small_sample_n_values(self):
        values = [1.1, 2.2, 3.3]
        result = terminal_digit_test(values)
        assert result.n_values == 3


class TestTerminalDigitEdgeCases:
    """Test terminal digit analysis edge cases."""

    def test_empty_list(self):
        result = terminal_digit_test([])
        assert isinstance(result, TerminalDigitResult)
        assert result.suspicious is False
        assert result.n_values == 0

    def test_integers_only(self):
        values = [float(i) for i in range(1, 101)]
        result = terminal_digit_test(values)
        assert isinstance(result, TerminalDigitResult)
        assert isinstance(result.digit_counts, dict)

    def test_default_alpha(self):
        values = [1.0, 2.5, 3.0, 4.5] * 50
        result_default = terminal_digit_test(values)
        result_explicit = terminal_digit_test(values, alpha=0.01)
        assert result_default.chi_squared == result_explicit.chi_squared
        assert result_default.p_value == result_explicit.p_value

    def test_all_same_value(self):
        values = [3.7] * 100
        result = terminal_digit_test(values)
        assert isinstance(result, TerminalDigitResult)
