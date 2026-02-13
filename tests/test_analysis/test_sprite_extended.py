from __future__ import annotations

from snoopy.analysis.sprite import SPRITEResult, sprite_test


class TestSpriteDeterministic:
    def test_same_seed_produces_identical_results(self):
        result_a = sprite_test(
            reported_mean=3.50,
            reported_sd=1.20,
            n=30,
            seed=42,
        )
        result_b = sprite_test(
            reported_mean=3.50,
            reported_sd=1.20,
            n=30,
            seed=42,
        )
        assert result_a.consistent == result_b.consistent
        assert result_a.mean_achievable == result_b.mean_achievable
        assert result_a.sd_achievable == result_b.sd_achievable
        assert result_a.details == result_b.details


class TestSpriteDifferentSeeds:
    def test_different_seeds_agree_on_clear_cases(self):
        result_42 = sprite_test(
            reported_mean=4.00,
            reported_sd=1.50,
            n=30,
            seed=42,
        )
        result_99 = sprite_test(
            reported_mean=4.00,
            reported_sd=1.50,
            n=30,
            seed=99,
        )
        assert result_42.consistent == result_99.consistent


class TestSpriteKnownInconsistentSD:
    def test_impossibly_small_sd_flagged_inconsistent(self):
        result = sprite_test(
            reported_mean=4.0,
            reported_sd=0.01,
            n=30,
            min_val=1,
            max_val=7,
            seed=42,
        )
        assert isinstance(result, SPRITEResult)
        assert result.consistent is False
        assert result.sd_achievable is False


class TestSpriteBoundaryMean:
    def test_minimum_mean_requires_zero_sd(self):
        result = sprite_test(
            reported_mean=1.0,
            reported_sd=0.50,
            n=30,
            min_val=1,
            max_val=7,
            seed=42,
        )
        assert result.consistent is False

    def test_minimum_mean_with_zero_sd_is_consistent(self):
        result = sprite_test(
            reported_mean=1.0,
            reported_sd=0.0,
            n=30,
            min_val=1,
            max_val=7,
            seed=42,
        )
        assert result.consistent is True


class TestSpriteLargeN:
    def test_large_sample_result_fields_valid(self):
        result = sprite_test(
            reported_mean=4.0,
            reported_sd=1.50,
            n=200,
            min_val=1,
            max_val=7,
            seed=42,
        )
        assert isinstance(result, SPRITEResult)
        assert result.n == 200
        assert result.reported_mean == 4.0
        assert result.reported_sd == 1.50
        # Mean should be achievable for this range
        assert result.mean_achievable is True


class TestSpriteSeedField:
    def test_seed_value_stored_in_result(self):
        result = sprite_test(
            reported_mean=3.50,
            reported_sd=1.20,
            n=30,
            seed=42,
        )
        assert result.seed == 42

    def test_custom_seed_stored_in_result(self):
        result = sprite_test(
            reported_mean=3.50,
            reported_sd=1.20,
            n=30,
            seed=12345,
        )
        assert result.seed == 12345
