"""SPRITE (Sample Parameter Reconstruction via Iterative TEchniques) test.

Checks whether reported summary statistics (mean, SD, min, max, N) could
plausibly come from the same dataset. Complements the GRIM test by also
checking standard deviations. Well-established in fraud detection literature
but rarely automated.

Reference: Heathers & Brown (2019). SPRITE: A response to
Anaya's critique. PsyArXiv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of iterations for the reconstruction search
_MAX_ITERATIONS = 10_000
_MAX_ATTEMPTS = 1000


@dataclass
class SPRITEResult:
    """Result of the SPRITE consistency test."""

    consistent: bool
    reported_mean: float
    reported_sd: float
    n: int
    min_val: int
    max_val: int
    mean_achievable: bool
    sd_achievable: bool
    attempts: int
    details: str


def sprite_test(
    reported_mean: float,
    reported_sd: float,
    n: int,
    min_val: int = 1,
    max_val: int = 7,
    mean_decimals: int = 2,
    sd_decimals: int = 2,
) -> SPRITEResult:
    """Test whether reported mean and SD are simultaneously achievable.

    Generates random integer datasets of size N (within [min_val, max_val])
    and iteratively adjusts values to match the reported mean. Then checks
    whether the reported SD is achievable given those constraints.

    Args:
        reported_mean: The reported mean value.
        reported_sd: The reported standard deviation.
        n: Sample size.
        min_val: Minimum possible value (e.g. 1 for Likert scale).
        max_val: Maximum possible value (e.g. 7 for Likert scale).
        mean_decimals: Number of decimal places in reported mean.
        sd_decimals: Number of decimal places in reported SD.

    Returns:
        SPRITEResult indicating whether the statistics are consistent.
    """
    if n < 2:
        return SPRITEResult(
            consistent=True,
            reported_mean=reported_mean,
            reported_sd=reported_sd,
            n=n,
            min_val=min_val,
            max_val=max_val,
            mean_achievable=True,
            sd_achievable=True,
            attempts=0,
            details="N too small for SPRITE test",
        )

    # Check if mean is even possible given min/max constraints
    if reported_mean < min_val or reported_mean > max_val:
        return SPRITEResult(
            consistent=False,
            reported_mean=reported_mean,
            reported_sd=reported_sd,
            n=n,
            min_val=min_val,
            max_val=max_val,
            mean_achievable=False,
            sd_achievable=False,
            attempts=0,
            details=f"Reported mean {reported_mean} outside possible range [{min_val}, {max_val}]",
        )

    mean_tolerance = 0.5 / (10**mean_decimals)
    sd_tolerance = 0.5 / (10**sd_decimals)

    rng = np.random.default_rng()
    mean_achievable = False
    sd_achievable = False

    for attempt in range(_MAX_ATTEMPTS):
        # Start with random integers in [min_val, max_val]
        data = rng.integers(min_val, max_val + 1, size=n)

        # Iteratively adjust to match the target mean
        target_sum = round(reported_mean * n)
        # Clamp target_sum to valid range
        target_sum = max(min_val * n, min(max_val * n, target_sum))

        for _ in range(_MAX_ITERATIONS):
            current_sum = int(np.sum(data))
            diff = target_sum - current_sum

            if diff == 0:
                break

            # Pick a random element to adjust
            idx = rng.integers(0, n)
            if diff > 0 and data[idx] < max_val:
                data[idx] = min(data[idx] + 1, max_val)
            elif diff < 0 and data[idx] > min_val:
                data[idx] = max(data[idx] - 1, min_val)

        # Check if we matched the mean
        achieved_mean = np.mean(data)
        if abs(achieved_mean - reported_mean) <= mean_tolerance:
            mean_achievable = True

            # Check if SD matches
            achieved_sd = float(np.std(data, ddof=1))
            if abs(achieved_sd - reported_sd) <= sd_tolerance:
                sd_achievable = True
                return SPRITEResult(
                    consistent=True,
                    reported_mean=reported_mean,
                    reported_sd=reported_sd,
                    n=n,
                    min_val=min_val,
                    max_val=max_val,
                    mean_achievable=True,
                    sd_achievable=True,
                    attempts=attempt + 1,
                    details="Found consistent dataset",
                )

    # Construct detailed failure message
    if not mean_achievable:
        detail = (
            f"Could not construct a dataset of N={n} integers in [{min_val}, {max_val}] "
            f"with mean={reported_mean} after {_MAX_ATTEMPTS} attempts"
        )
    else:
        detail = (
            f"Mean={reported_mean} is achievable with N={n} in [{min_val}, {max_val}], "
            f"but SD={reported_sd} could not be simultaneously matched after "
            f"{_MAX_ATTEMPTS} attempts"
        )

    return SPRITEResult(
        consistent=False,
        reported_mean=reported_mean,
        reported_sd=reported_sd,
        n=n,
        min_val=min_val,
        max_val=max_val,
        mean_achievable=mean_achievable,
        sd_achievable=sd_achievable,
        attempts=_MAX_ATTEMPTS,
        details=detail,
    )
