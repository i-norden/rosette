"""Tests for GRIMMER test."""
from __future__ import annotations

from snoopy.analysis.statistical import grimmer_test, GRIMMERResult


class TestGRIMMERConsistent:
    """Test GRIMMER with known-consistent statistics."""

    def test_consistent_mean_sd(self):
        # SD=1.58 is achievable: sqrt(sum(xi-mean)^2/(n-1)) for integer data
        result = grimmer_test(mean=3.50, sd=1.58, n=10, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.consistent is True

    def test_result_fields_populated(self):
        result = grimmer_test(mean=3.50, sd=1.58, n=10, decimals=2)
        assert result.mean == 3.50
        assert result.sd == 1.58
        assert result.n == 10
        assert isinstance(result.possible_sds, list)
        assert isinstance(result.details, str)

    def test_possible_sds_nonempty(self):
        result = grimmer_test(mean=3.50, sd=1.58, n=10, decimals=2)
        assert len(result.possible_sds) > 0

    def test_reported_sd_in_possible_sds(self):
        result = grimmer_test(mean=3.50, sd=1.58, n=10, decimals=2)
        assert any(abs(s - 1.58) < 0.005 for s in result.possible_sds)


class TestGRIMMERInconsistent:
    """Test GRIMMER with known-inconsistent statistics."""

    def test_inconsistent_sd(self):
        result = grimmer_test(mean=3.50, sd=1.37, n=4, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.consistent is False

    def test_inconsistent_result_fields(self):
        result = grimmer_test(mean=3.50, sd=1.37, n=4, decimals=2)
        assert result.mean == 3.50
        assert result.sd == 1.37
        assert result.n == 4

    def test_inconsistent_sd_not_in_possible(self):
        result = grimmer_test(mean=3.50, sd=1.37, n=4, decimals=2)
        assert not any(abs(s - 1.37) < 0.005 for s in result.possible_sds)

    def test_inconsistent_details_nonempty(self):
        result = grimmer_test(mean=3.50, sd=1.37, n=4, decimals=2)
        assert len(result.details) > 0


class TestGRIMMEREdgeCases:
    """Test GRIMMER edge cases with small N and boundary values."""

    def test_n_equals_2(self):
        result = grimmer_test(mean=2.50, sd=0.71, n=2, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.n == 2

    def test_n_equals_1_returns_consistent(self):
        result = grimmer_test(mean=3.50, sd=0.00, n=1, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.consistent is True

    def test_sd_zero_all_values_equal(self):
        result = grimmer_test(mean=1.00, sd=0.00, n=5, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.consistent is True

    def test_default_decimals(self):
        result = grimmer_test(mean=3.50, sd=1.29, n=10)
        assert isinstance(result, GRIMMERResult)


class TestGRIMMERLargeN:
    """Test GRIMMER with large sample sizes."""

    def test_large_n_consistent(self):
        result = grimmer_test(mean=3.45, sd=1.50, n=100, decimals=2)
        assert isinstance(result, GRIMMERResult)
        assert result.consistent is True

    def test_large_n_uses_approximation(self):
        # Large N uses approximation mode, which may return empty possible_sds
        result = grimmer_test(mean=3.45, sd=1.50, n=100, decimals=2)
        assert result.consistent is True
        assert "approximation" in result.details.lower() or "feasible" in result.details.lower()

    def test_large_n_fields(self):
        result = grimmer_test(mean=3.45, sd=1.50, n=100, decimals=2)
        assert result.mean == 3.45
        assert result.sd == 1.50
        assert result.n == 100
