"""Tests for variance ratio test."""

from __future__ import annotations

from rosette.analysis.statistical import variance_ratio_test, VarianceRatioResult


class TestVarianceRatioRealistic:
    """Test variance ratio with realistically varying standard deviations."""

    def test_realistic_sds_not_suspicious(self):
        pairs = [(1.2, 30), (1.5, 28), (0.9, 32), (1.8, 25)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is False

    def test_realistic_result_fields(self):
        pairs = [(1.2, 30), (1.5, 28), (0.9, 32), (1.8, 25)]
        result = variance_ratio_test(pairs)
        assert result.n_groups == 4
        assert result.observed_variance_of_sds >= 0.0
        assert result.expected_variance_of_sds >= 0.0
        assert result.ratio >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_realistic_details_nonempty(self):
        pairs = [(1.2, 30), (1.5, 28), (0.9, 32), (1.8, 25)]
        result = variance_ratio_test(pairs)
        assert isinstance(result.details, str)
        assert len(result.details) > 0

    def test_realistic_high_p_value(self):
        pairs = [(1.2, 30), (1.5, 28), (0.9, 32), (1.8, 25)]
        result = variance_ratio_test(pairs)
        assert result.p_value > 0.05

    def test_many_realistic_groups(self):
        pairs = [
            (1.2, 30),
            (1.5, 28),
            (0.9, 32),
            (1.8, 25),
            (1.1, 35),
            (1.6, 27),
            (1.3, 30),
            (2.0, 22),
        ]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.n_groups == 8
        assert result.suspicious is False


class TestVarianceRatioTooSimilar:
    """Test variance ratio with implausibly similar standard deviations."""

    def test_too_similar_sds_suspicious(self):
        pairs = [(1.50, 30), (1.51, 30), (1.49, 30), (1.50, 30)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is True

    def test_too_similar_low_p_value(self):
        pairs = [(1.50, 30), (1.51, 30), (1.49, 30), (1.50, 30)]
        result = variance_ratio_test(pairs)
        assert result.p_value < 0.05

    def test_too_similar_low_observed_variance(self):
        pairs = [(1.50, 30), (1.51, 30), (1.49, 30), (1.50, 30)]
        result = variance_ratio_test(pairs)
        assert result.observed_variance_of_sds < result.expected_variance_of_sds

    def test_too_similar_ratio_below_one(self):
        pairs = [(1.50, 30), (1.51, 30), (1.49, 30), (1.50, 30)]
        result = variance_ratio_test(pairs)
        assert result.ratio < 1.0

    def test_identical_sds_suspicious(self):
        pairs = [(1.50, 30), (1.50, 30), (1.50, 30), (1.50, 30)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is True

    def test_custom_alpha(self):
        pairs = [(1.50, 30), (1.51, 30), (1.49, 30), (1.50, 30)]
        result = variance_ratio_test(pairs, alpha=0.01)
        assert isinstance(result, VarianceRatioResult)


class TestVarianceRatioTooFewGroups:
    """Test variance ratio with insufficient groups."""

    def test_two_groups_not_suspicious(self):
        pairs = [(1.50, 30), (1.51, 30)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is False

    def test_one_group_not_suspicious(self):
        pairs = [(1.50, 30)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is False

    def test_empty_list_not_suspicious(self):
        pairs = []
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is False
        assert result.n_groups == 0

    def test_two_groups_n_groups(self):
        pairs = [(1.50, 30), (1.51, 30)]
        result = variance_ratio_test(pairs)
        assert result.n_groups == 2


class TestVarianceRatioSingleN:
    """Test variance ratio when all groups have the same sample size."""

    def test_same_n_realistic_sds(self):
        pairs = [(1.2, 30), (1.5, 30), (0.9, 30), (1.8, 30)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is False

    def test_same_n_similar_sds(self):
        pairs = [(1.50, 25), (1.51, 25), (1.49, 25), (1.50, 25)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.suspicious is True

    def test_same_n_fields(self):
        pairs = [(1.2, 30), (1.5, 30), (0.9, 30), (1.8, 30)]
        result = variance_ratio_test(pairs)
        assert result.n_groups == 4
        assert result.observed_variance_of_sds >= 0.0
        assert result.expected_variance_of_sds >= 0.0

    def test_same_large_n(self):
        pairs = [(1.2, 100), (1.5, 100), (0.9, 100), (1.8, 100), (1.4, 100)]
        result = variance_ratio_test(pairs)
        assert isinstance(result, VarianceRatioResult)
        assert result.n_groups == 5
        assert result.suspicious is False
