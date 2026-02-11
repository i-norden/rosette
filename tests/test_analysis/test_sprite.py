"""Tests for SPRITE (Sample Parameter Reconstruction) consistency checks."""

from __future__ import annotations

from snoopy.analysis.sprite import SPRITEResult, sprite_test


class TestSpriteConsistent:
    def test_sprite_consistent(self) -> None:
        """A known-consistent mean and SD on a 1-7 Likert scale passes SPRITE."""
        # Mean=4.0, SD=1.5, N=30 on a 1-7 scale is easily achievable
        result = sprite_test(
            reported_mean=4.0,
            reported_sd=1.5,
            n=30,
            min_val=1,
            max_val=7,
            sd_decimals=1,
        )
        assert isinstance(result, SPRITEResult)
        assert result.consistent is True
        assert result.mean_achievable is True
        assert result.sd_achievable is True


class TestSpriteImpossibleMean:
    def test_sprite_impossible_mean(self) -> None:
        """Mean outside the possible range [min_val, max_val] fails SPRITE."""
        result = sprite_test(
            reported_mean=10.0,
            reported_sd=1.0,
            n=20,
            min_val=1,
            max_val=7,
        )
        assert isinstance(result, SPRITEResult)
        assert result.consistent is False
        assert result.mean_achievable is False


class TestSpriteSmallN:
    def test_sprite_small_n(self) -> None:
        """N < 2 returns consistent (test is not applicable)."""
        result = sprite_test(
            reported_mean=3.5,
            reported_sd=0.5,
            n=1,
            min_val=1,
            max_val=7,
        )
        assert isinstance(result, SPRITEResult)
        assert result.consistent is True
        assert result.attempts == 0
        assert "too small" in result.details.lower()
